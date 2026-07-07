#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from __future__ import annotations

import concurrent.futures
import json
import shutil
import traceback
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from queue import Queue
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlrc.whisper_types import Segment

from openlrc.config import TranscriptionConfig, TranslationConfig
from openlrc.defaults import (
    COMPARE_SUFFIX,
    NONTRANS_SUFFIX,
    PREPROCESSED_DIR,
    PREPROCESSED_SUFFIX,
    TRANSCRIBED_SUFFIX,
    TRANSLATED_SUFFIX,
    default_preprocess_options,
    default_whisper_cpp_options,
)
from openlrc.llama_resources import DEFAULT_LLAMA_IDLE_TIMEOUT, DEFAULT_LLAMA_PORT, LOCAL_LLAMA_API_KEY
from openlrc.logger import logger
from openlrc.media_utils import extract_audio, get_audio_duration, get_file_type
from openlrc.opt import SubtitleOptimizer
from openlrc.subtitle import BilingualSubtitle, Subtitle
from openlrc.utils import Timer, extend_filename, format_timestamp, get_preprocessed_path


class LRCer:
    """
    Orchestrator for audio/video transcription and translation.

    Usage::

        from openlrc import LRCer
        from openlrc.config import TranscriptionConfig, TranslationConfig
        from openlrc.models import ModelConfig, ModelProvider

        lrcer = LRCer(
            transcription=TranscriptionConfig(whisper_model="small"),
            translation=TranslationConfig(
                chatbot=ModelConfig(provider=ModelProvider.OPENAI, name='gpt-4.1-nano'),
                fee_limit=1.0,
            ),
        )

        # Or with all defaults:
        lrcer = LRCer()
    """

    def __init__(
        self, *, transcription: TranscriptionConfig | None = None, translation: TranslationConfig | None = None
    ):
        self._transcription_config = transcription or TranscriptionConfig()
        self._translation_config = translation or TranslationConfig()

        # Translation state
        self.fee_limit = self._translation_config.fee_limit
        self.api_fee = 0  # Can be updated in different thread, operation should be thread-safe
        self.from_video = set()
        self.glossary = self.parse_glossary(self._translation_config.glossary)
        self.is_force_glossary_used = self._translation_config.is_force_glossary_used
        self.translate_mode = self._translation_config.translate_mode
        self.enable_cr = self._translation_config.enable_cr
        self.chunked_guideline = self._translation_config.chunked_guideline

        self._lock = Lock()
        self.exception = None
        self.consumer_thread = self._translation_config.consumer_thread

        # Merge default options with provided options
        self.asr_options = {**default_whisper_cpp_options, **(self._transcription_config.asr_options or {})}
        self.preprocess_options = {
            **default_preprocess_options,
            **(self._transcription_config.preprocess_options or {}),
        }

        # Lazy initialization: Transcriber is created on first access via the property.
        self._transcriber_lock = Lock()
        self._transcriber = None
        self.transcribed_paths = []

        # Lazy initialization: ChatBot instances are created on first access via properties.
        self._chatbot_lock = Lock()
        self._chatbot = None
        self._retry_chatbot = None
        self._cr_chatbot = None
        self._local_llm_server = None

    @classmethod
    def local(
        cls,
        *,
        model: str = "qwen3.5-9b",
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        translate_mode: str = "lean",
        transcription: TranscriptionConfig | None = None,
    ) -> LRCer:
        """Create an LRCer configured for local whisper.cpp transcription and llama.cpp translation."""
        return cls(
            transcription=transcription,
            translation=TranslationConfig.local_qwen35_9b(
                model=model,
                idle_timeout=idle_timeout,
                port=port,
                translate_mode=translate_mode,
            ),
        )

    @property
    def transcriber(self):
        """Lazily initialize and return the Transcriber instance (thread-safe)."""
        if self._transcriber is None:
            from openlrc.transcribe import Transcriber

            with self._transcriber_lock:
                if self._transcriber is None:
                    self._transcriber = Transcriber(
                        model_name=self._transcription_config.whisper_model,
                        cli_path=self._transcription_config.cli_path,
                        vad_model=self._transcription_config.vad_model,
                        asr_options=self.asr_options,
                    )
        return self._transcriber

    @property
    def chatbot(self):
        """Lazily initialize and return the primary ChatBot instance (thread-safe)."""
        if self._chatbot is None:
            from openlrc.agents import create_chatbot
            from openlrc.models import ModelConfig, ModelProvider

            with self._chatbot_lock:
                if self._chatbot is None:
                    if self._translation_config.chatbot is not None:
                        model_config = self._translation_config.chatbot
                    elif self._local_llm_enabled():
                        assert self._translation_config.local_llm is not None
                        model_config = ModelConfig(
                            provider=ModelProvider.LOCAL_LLAMA,
                            name=self._translation_config.local_llm.alias,
                            api_key=LOCAL_LLAMA_API_KEY,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                    else:
                        model_config = ModelConfig(provider=ModelProvider.OPENAI, name="gpt-4.1-nano")
                    model_config = self._prepare_local_chatbot_config(model_config)
                    self._chatbot = create_chatbot(model_config, self.fee_limit)
        return self._chatbot

    @property
    def retry_chatbot(self):
        """Lazily initialize and return the retry ChatBot instance (thread-safe).

        Returns None if no retry_chatbot config is provided.
        """
        if self._retry_chatbot is None and self._translation_config.retry_chatbot:
            from openlrc.agents import create_chatbot

            with self._chatbot_lock:
                if self._retry_chatbot is None:
                    model_config = self._prepare_local_chatbot_config(self._translation_config.retry_chatbot)
                    self._retry_chatbot = create_chatbot(model_config, self.fee_limit)
        return self._retry_chatbot

    @property
    def cr_chatbot(self):
        """Lazily initialize and return the CR ChatBot instance (thread-safe).

        Returns None if no cr_chatbot config is provided.
        """
        if self._cr_chatbot is None and self._translation_config.cr_chatbot:
            from openlrc.agents import create_chatbot

            with self._chatbot_lock:
                if self._cr_chatbot is None:
                    model_config = self._prepare_local_chatbot_config(self._translation_config.cr_chatbot)
                    self._cr_chatbot = create_chatbot(model_config, self.fee_limit)
        return self._cr_chatbot

    def _local_llm_enabled(self) -> bool:
        return bool(self._translation_config.local_llm and self._translation_config.local_llm.enabled)

    def _local_server(self):
        if not self._local_llm_enabled():
            return None

        if self._local_llm_server is None:
            from openlrc.local_llm_server import LocalLLMServer

            config = self._translation_config.local_llm
            assert config is not None
            self._local_llm_server = LocalLLMServer(
                server_path=config.server_path,
                model_path=config.model_path,
                host=config.host,
                port=config.port,
                alias=config.alias,
                ctx_size=config.ctx_size,
                gpu_layers=config.gpu_layers,
                idle_timeout=config.idle_timeout,
                startup_timeout=config.startup_timeout,
                extra_args=config.extra_args,
            )

        return self._local_llm_server

    def _ensure_local_llm_server(self) -> str:
        server = self._local_server()
        if server is None:
            raise RuntimeError("Local LLM server is not enabled.")
        return server.ensure_running()

    def _prepare_local_chatbot_config(self, model_config):
        if not self._local_llm_enabled():
            return model_config

        from openlrc.models import ModelProvider

        if model_config.provider != ModelProvider.LOCAL_LLAMA:
            return model_config

        model_config = deepcopy(model_config)
        model_config.base_url = self._ensure_local_llm_server()
        model_config.api_key = model_config.api_key or LOCAL_LLAMA_API_KEY
        return model_config

    def _local_llm_session(self):
        server = self._local_server()
        if server is None:
            return nullcontext()
        return server.session()

    def close(self):
        """Close ChatBot connections and release resources.

        Safe to call multiple times or before any ChatBot has been created.
        """
        if self._chatbot is not None:
            self._chatbot.close()
            self._chatbot = None
        if self._retry_chatbot is not None:
            self._retry_chatbot.close()
            self._retry_chatbot = None
        if self._cr_chatbot is not None:
            self._cr_chatbot.close()
            self._cr_chatbot = None
        if self._local_llm_server is not None:
            self._local_llm_server.close()
            self._local_llm_server = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_translator(self, timestamps):
        """Create a Translator instance based on translate_mode."""
        factories = {"standard": self._create_standard_translator, "lean": self._create_lean_translator}
        factory = factories.get(self.translate_mode)
        if factory is None:
            raise ValueError(f"Unknown translate_mode: {self.translate_mode!r}. Choose from: {list(factories)}")
        return factory(timestamps)

    def _create_standard_translator(self, timestamps):
        from openlrc.translate import LLMTranslator

        if not self.enable_cr:
            logger.warning("enable_cr is only used in lean mode, ignoring.")
        return LLMTranslator(
            chatbot=self.chatbot,
            retry_chatbot=self.retry_chatbot,
            cr_chatbot=self.cr_chatbot,
            timestamps=timestamps,
            chunked_guideline=self.chunked_guideline,
        )

    def _create_lean_translator(self, timestamps):
        from openlrc.translate import LeanTranslator

        return LeanTranslator(
            chatbot=self.chatbot,
            retry_chatbot=self.retry_chatbot,
            cr_chatbot=self.cr_chatbot,
            timestamps=timestamps,
            enable_cr=self.enable_cr,
            chunked_guideline=self.chunked_guideline,
        )

    @staticmethod
    def parse_glossary(glossary: dict | str | Path | None) -> dict | None:
        if not glossary:
            return None

        if isinstance(glossary, dict):
            return glossary

        glossary_path = Path(glossary)
        if not glossary_path.exists():
            logger.warning("Glossary file not found.")
            return None

        with open(glossary_path, encoding="utf-8") as f:
            loaded: dict = json.load(f)

        return loaded

    def _transcribe_single(self, audio_path: Path, src_lang: str | None = None) -> Path:
        """
        Transcribe a single audio file and return the path to the transcribed JSON.

        If a transcribed JSON file already exists for the given audio, it is reused.

        Args:
            audio_path (Path): Path to the preprocessed audio file.
            src_lang (Optional[str]): Source language. If None, language will be auto-detected.

        Returns:
            Path: Path to the transcribed JSON file.
        """
        transcribed_path = extend_filename(audio_path, TRANSCRIBED_SUFFIX).with_suffix(".json")
        if not transcribed_path.exists():
            with Timer("Transcription process"):
                logger.info(
                    f"Audio length: {audio_path}: {format_timestamp(get_audio_duration(audio_path), fmt='srt')}"
                )
                segments, info = self.transcriber.transcribe(audio_path, language=src_lang)
                logger.info(f"Detected language: {info.language}")

            self.to_json(segments, name=transcribed_path, lang=info.language)
        else:
            logger.info(f"Found transcribed json file: {transcribed_path}")
        return transcribed_path

    def produce_transcriptions(self, transcription_queue, audio_paths, src_lang):
        """
        Sequentially produce transcriptions for given audio paths and put them in the queue.

        Args:
            transcription_queue (Queue): Queue to store transcribed paths.
            audio_paths (List[Path]): List of audio file paths to transcribe.
            src_lang (str): Source language for transcription. If None, language will be auto-detected.

        This method processes each audio file sequentially, transcribing it if necessary,
        and puts the path of the transcribed JSON file into the queue.
        """
        for audio_path in audio_paths:
            transcribed_path = self._transcribe_single(audio_path, src_lang)
            transcription_queue.put(transcribed_path)

        transcription_queue.put(None)
        logger.info("Transcription producer finished.")

    def transcribe(
        self,
        paths: str | Path | list[str | Path],
        src_lang: str | None = None,
        noise_suppress: bool = False,
        skip_preprocess: bool = False,
    ) -> list[Path]:
        """
        Transcribe audio/video files and return paths to the transcribed JSON files.

        This method runs only the preprocessing and transcription stages, without
        any translation. It can be used independently when translation is not needed,
        or to produce transcription files that can later be passed to translate().

        Args:
            paths (Union[str, Path, List[Union[str, Path]]]): Audio/Video paths.
            src_lang (Optional[str]): Language of the audio, default to auto-detect.
            noise_suppress (bool): Whether to suppress noise in the audio. Default is False.
            skip_preprocess (bool): Whether to skip the preprocessing step. Default is False.

        Returns:
            List[Path]: List of paths to the transcribed JSON files.
        """
        if not paths:
            logger.warning("No audio/video file given. Skip transcription.")
            return []

        if isinstance(paths, (str, Path)):
            paths = [paths]

        # Keep behavior aligned with pre_process(): de-duplicate repeated inputs.
        paths = [Path(p) for p in set(paths)]

        if skip_preprocess:
            audio_paths = [get_preprocessed_path(p) for p in paths]
            for p in audio_paths:
                if not p.exists():
                    raise FileNotFoundError(
                        f"Preprocessed file not found: {p}. Run pre_process() first or set skip_preprocess=False."
                    )
        else:
            audio_paths = self.pre_process(paths, noise_suppress=noise_suppress)

        logger.info(f"Transcribing {len(audio_paths)} audio files: {pformat(audio_paths)}")

        return [self._transcribe_single(p, src_lang) for p in audio_paths]

    @staticmethod
    def _get_base_name(transcribed_path: Path) -> str:
        """Extract the original audio base name from a transcribed JSON path."""
        return transcribed_path.stem.replace(f"{PREPROCESSED_SUFFIX}{TRANSCRIBED_SUFFIX}", "")

    def _is_video_transcription(self, transcribed_path: Path, base_name: str) -> bool:
        """
        Determine whether a transcribed file originated from a video source.

        In normal run() flows, self.from_video is populated during preprocessing.
        For standalone translate() usage, infer from an existing source file next to
        the output directory when possible.
        """
        source_stem = transcribed_path.parent.parent / base_name
        if source_stem in self.from_video:
            return True

        video_suffixes = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m4v", ".flv", ".wmv")
        return any(source_stem.with_suffix(suffix).exists() for suffix in video_suffixes)

    def _build_final_subtitle(self, base_name, target_lang, transcribed_opt_sub, skip_trans):
        """
        Build the final subtitle object for a single transcription.

        If skip_trans is True, copies the transcription as the final output.
        Otherwise, performs LLM translation via _translate().

        Args:
            base_name (str): Original audio base name.
            target_lang (str): Target language for translation.
            transcribed_opt_sub (Subtitle): Post-processed transcription subtitle.
            skip_trans (bool): Whether to skip translation.

        Returns:
            Subtitle: The final subtitle (translated or copied), or None on error.
        """
        translated_path = extend_filename(transcribed_opt_sub.filename, TRANSLATED_SUFFIX)
        final_json_path = translated_path.with_name(f"{base_name}.json")

        if final_json_path.exists():
            return Subtitle.from_json(final_json_path)

        if skip_trans:
            shutil.copy(transcribed_opt_sub.filename, final_json_path)
            transcribed_opt_sub.filename = final_json_path
            return transcribed_opt_sub

        try:
            with Timer("Translation process"):
                return self._translate(base_name, target_lang, transcribed_opt_sub, translated_path)
        except Exception as e:
            self.exception = e
            return None

    def _generate_subtitle_files(self, subtitle, base_name, subtitle_format):
        """
        Generate subtitle file (.lrc or .srt) and move it to the output directory.

        Args:
            subtitle (Subtitle): The subtitle object to export.
            base_name (str): Original audio base name.
            subtitle_format (str): Output format, either 'lrc' or 'srt'.
        """
        subtitle_path = getattr(subtitle, f"to_{subtitle_format}")()
        result_path = subtitle_path.parent.parent / f"{base_name}.{subtitle_format}"
        shutil.move(subtitle_path, result_path)
        self.transcribed_paths.append(result_path)

    def _handle_bilingual_subtitles(self, transcribed_path, base_name, transcribed_opt_sub, subtitle_format):
        """
        Generate bilingual subtitles and a non-translated subtitle file.

        Args:
            transcribed_path (Path): Path to the transcribed JSON file.
            base_name (str): Original audio base name.
            transcribed_opt_sub (Subtitle): Post-processed transcription subtitle.
            subtitle_format (str): Output format, either 'lrc' or 'srt'.
        """
        bilingual_subtitle = BilingualSubtitle.from_preprocessed(transcribed_path.parent, base_name)
        bilingual_optimizer = SubtitleOptimizer(bilingual_subtitle)
        bilingual_optimizer.extend_time()

        bilingual_path = getattr(bilingual_subtitle, f"to_{subtitle_format}")()
        shutil.move(bilingual_path, bilingual_path.parent.parent / bilingual_path.name)

        non_translated_subtitle = transcribed_opt_sub
        optimizer = SubtitleOptimizer(non_translated_subtitle)
        optimizer.extend_time()
        non_translated_path = getattr(non_translated_subtitle, f"to_{subtitle_format}")()
        shutil.move(
            non_translated_path, non_translated_path.parent.parent / f"{base_name}{NONTRANS_SUFFIX}.{subtitle_format}"
        )

    def _process_transcribed_file(
        self, transcribed_path: Path, target_lang: str | None, skip_trans: bool = False, bilingual_sub: bool = False
    ):
        """
        Process a single transcribed JSON file through post-processing, translation
        (or copy when skip_trans=True), and subtitle generation.

        This is the shared pipeline used by translate(), translation_worker(), and
        the skip_trans branch of run().

        Args:
            transcribed_path (Path): Path to the transcribed JSON file.
            target_lang (Optional[str]): Target language for translation.
            skip_trans (bool): Whether to skip translation.
            bilingual_sub (bool): Whether to generate bilingual subtitles.
        """
        base_name = self._get_base_name(transcribed_path)
        subtitle_format = "srt" if self._is_video_transcription(transcribed_path, base_name) else "lrc"

        transcribed_sub = Subtitle.from_json(transcribed_path)
        transcribed_opt_sub = self.post_process(transcribed_sub, update_name=True)

        final_subtitle = self._build_final_subtitle(base_name, target_lang, transcribed_opt_sub, skip_trans)

        self._generate_subtitle_files(final_subtitle, base_name, subtitle_format)

        if not skip_trans and bilingual_sub:
            self._handle_bilingual_subtitles(transcribed_path, base_name, transcribed_opt_sub, subtitle_format)

    def translate(
        self, transcribed_paths: Path | list[Path], target_lang: str = "zh-cn", bilingual_sub: bool = False
    ) -> list[Path]:
        """
        Translate previously transcribed JSON files and generate subtitle files.

        This method runs only the translation and subtitle generation stages.
        It can be used independently on transcription files produced by transcribe(),
        without needing the original audio files or the whisper model.

        Args:
            transcribed_paths (Union[Path, List[Path]]): Path(s) to transcribed JSON files.
            target_lang (str): Target language for translation. Default is 'zh-cn'.
            bilingual_sub (bool): Whether to generate bilingual subtitles. Default is False.

        Returns:
            List[Path]: List of paths to the generated subtitle files.
        """
        self.transcribed_paths = []
        self.exception = None

        if isinstance(transcribed_paths, Path):
            transcribed_paths = [transcribed_paths]

        logger.info(f"Translating {len(transcribed_paths)} transcribed files: {pformat(transcribed_paths)}")

        for transcribed_path in transcribed_paths:
            self._process_transcribed_file(transcribed_path, target_lang, bilingual_sub=bilingual_sub)

            if self.exception:
                traceback.print_exception(type(self.exception), self.exception, self.exception.__traceback__)
                raise self.exception

        logger.info(f"Total API fee used: {self.api_fee:.4f} USD")

        return self.transcribed_paths

    def consume_transcriptions(self, transcription_queue, target_lang, skip_trans, bilingual_sub):
        """
        Consume transcriptions from the queue using multiple threads for parallel processing.

        Args:
            transcription_queue (Queue): Queue containing paths of transcribed files.
            target_lang (str): Target language for translation.
            skip_trans (bool): Whether to skip the translation process.
            bilingual_sub (bool): Whether to generate bilingual subtitles.

        This method creates multiple worker threads to process transcriptions in parallel,
        handling translation and subtitle generation.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.translation_worker, transcription_queue, target_lang, skip_trans, bilingual_sub)
                for _ in range(self.consumer_thread)
            ]
            concurrent.futures.wait(futures)
        logger.info("Transcription consumer finished.")

    def translation_worker(self, transcription_queue, target_lang, skip_trans, bilingual_sub):
        """
        Worker function for parallel translation and subtitle processing.

        Args:
            transcription_queue (Queue): Queue containing paths of transcribed files.
            target_lang (str): Target language for translation.
            skip_trans (bool): Whether to skip the translation process.
            bilingual_sub (bool): Whether to generate bilingual subtitles.

        This method continuously processes transcriptions from the queue, handling translation,
        subtitle generation, and bilingual subtitle creation if required.
        """
        while True:
            logger.debug("Translation worker waiting transcription...")
            transcribed_path = transcription_queue.get()

            if transcribed_path is None:
                transcription_queue.put(None)
                logger.debug("Translation worker finished.")
                return

            logger.info(f"Got transcription: {transcribed_path}")

            self._process_transcribed_file(transcribed_path, target_lang, skip_trans, bilingual_sub)

            logger.info(f"Translation fee til now: {self.api_fee:.4f} USD")

    def _translate(self, audio_name, target_lang, transcribed_opt_sub, translated_path):
        """
        Perform translation of transcribed subtitles.

        Args:
            audio_name (str): Name of the audio file.
            target_lang (str): Target language for translation.
            transcribed_opt_sub (Subtitle): Optimized transcribed subtitle object.
            translated_path (Path): Path to save the translated subtitle.

        Returns:
            Subtitle: Final translated and post-processed subtitle object.

        This method handles the translation process, including context preparation,
        actual translation, and post-processing of the translated subtitles.
        """
        from openlrc.context import TranslateInfo

        context = TranslateInfo(
            title=audio_name, audio_type="Movie", glossary=self.glossary, forced_glossary=self.is_force_glossary_used
        )

        json_filename = Path(translated_path.parent / (audio_name + ".json"))
        compare_path = Path(translated_path.parent, f"{audio_name}{COMPARE_SUFFIX}.json")
        if not translated_path.exists():
            with self._local_llm_session():
                timestamps = [(seg.start, seg.end) for seg in transcribed_opt_sub.segments]
                translator = self._create_translator(timestamps)

                target_texts = translator.translate(
                    transcribed_opt_sub.texts,
                    src_lang=transcribed_opt_sub.lang,
                    target_lang=target_lang,
                    info=context,
                    compare_path=compare_path,
                )

            with self._lock:
                self.api_fee += translator.api_fee  # Ensure thread-safe

            translated_sub = deepcopy(transcribed_opt_sub)
            translated_sub.set_texts(target_texts, lang=target_lang)

            # xxx_transcribed_optimized_translated.json
            translated_sub.save(translated_path, update_name=True)
        else:
            logger.info(f"Found translated json file: {translated_path}")
        translated_sub = Subtitle.from_json(translated_path)

        final_subtitle = self.post_process(
            translated_sub, output_name=json_filename, update_name=True, extend_time=True
        )  # xxx.json

        return final_subtitle

    def run(
        self,
        paths: str | Path | list[str | Path],
        src_lang: str | None = None,
        target_lang="zh-cn",
        skip_trans=False,
        noise_suppress=False,
        bilingual_sub=False,
        clear_temp=False,
        skip_preprocess=False,
    ) -> list[str]:
        """
        Run the entire transcription and translation process.

        This method orchestrates the entire workflow of transcribing audio/video files and translating the transcriptions.
        It operates in two parallel phases: transcription (producer) and translation (consumer).

        Detailed process:
        1. Pre-processing:
           - Convert input paths to Path objects.
           - Extract audio from video files if necessary.
           - Apply noise suppression if requested.

        2. Transcription (Producer):
           - Sequentially process each audio file.
           - For each file, either transcribe it or use existing transcription.
           - Put transcribed file paths into a queue.

        3. Translation (Consumer):
           - Create multiple worker threads to process transcriptions in parallel.
           - Each worker:
             a. Retrieves a transcription from the queue.
             b. Translates the transcription (if not skipped).
             c. Generates subtitle files.
             d. Creates bilingual subtitles if requested.

        4. Post-processing:
           - Clear temporary files if requested.

        Args:
            paths (Union[str, Path, List[Union[str, Path]]]): Audio/Video paths, can be a list or a single path.
            src_lang (Optional[str]): Language of the audio, default to auto-detect.
            target_lang (str): Target language for translation, default to Mandarin Chinese ('zh-cn').
            skip_trans (bool): Whether to skip the translation process. Default is False.
            noise_suppress (bool): Whether to suppress noise in the audio. Default is False.
            bilingual_sub (bool): Whether to generate bilingual subtitles. Default is False.
            clear_temp (bool): Whether to clear all temporary files, including generated .wav from video.
                               Set to False to keep intermediate results if errors occur. Default is False.
            skip_preprocess (bool): Whether to skip the preprocessing step. When True, assumes that
                               preprocessed files already exist at the expected locations (as returned by
                               get_preprocessed_path()). This is useful when preprocessing and transcription
                               are run in separate stages. Default is False.

        Returns:
            List[str]: List of paths to the generated subtitle files.

        Raises:
            Exception: If an error occurs during the transcription or translation process.

        Note:
            - The method uses a producer-consumer pattern with a queue to manage parallel processing.
            - It tracks and logs API usage fees for translation services.
            - Temporary files are managed and can be cleared based on the clear_temp parameter.
        """
        self.transcribed_paths = []

        if not paths:
            logger.warning("No audio/video file given. Skip LRCer.run()")
            return []

        if isinstance(paths, (str, Path)):
            paths = [paths]

        paths = list(map(Path, paths))

        if skip_preprocess:
            # Use preprocessed files directly without running preprocessing
            audio_paths = [get_preprocessed_path(p) for p in paths]
            for p in audio_paths:
                if not p.exists():
                    raise FileNotFoundError(
                        f"Preprocessed file not found: {p}. Run pre_process() first or set skip_preprocess=False."
                    )
        else:
            audio_paths = self.pre_process(paths, noise_suppress=noise_suppress)

        if skip_trans:
            # Transcribe-only: no translation threads needed
            transcribed_paths = self.transcribe(paths, src_lang=src_lang, skip_preprocess=True)
            for transcribed_path in transcribed_paths:
                self._process_transcribed_file(transcribed_path, target_lang=None, skip_trans=True)

            if clear_temp:
                logger.info("Clearing temporary folder...")
                self.clear_temp_files(audio_paths)

            return self.transcribed_paths

        logger.info(f"Working on {len(audio_paths)} audio files: {pformat(audio_paths)}")

        transcription_queue = Queue()

        with Timer("Transcription (Producer) and Translation (Consumer) process"):
            consumer = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="Consumer").submit(
                self.consume_transcriptions, transcription_queue, target_lang, skip_trans, bilingual_sub
            )
            producer = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="Producer").submit(
                self.produce_transcriptions, transcription_queue, audio_paths, src_lang
            )

            producer.result()
            consumer.result()

            if self.exception:
                traceback.print_exception(type(self.exception), self.exception, self.exception.__traceback__)
                raise self.exception

        logger.info(f"Total API fee used: {self.api_fee:.4f} USD")

        if clear_temp:
            logger.info("Clearing temporary folder...")
            self.clear_temp_files(audio_paths)

        return self.transcribed_paths

    def clear_temp_files(self, paths):
        """
        Clear the temporary files generated during the transcription and translation process.

        Args:
            paths (List[Path]): List of paths to the processed audio files.

        This method removes temporary folders and generated wave files from video processing.
        """
        temp_folders = {path.parent for path in paths}
        for folder in temp_folders:
            if folder.name != PREPROCESSED_DIR:
                raise ValueError(f"Not a temporary folder: {folder}")

            shutil.rmtree(folder)
            logger.debug(f"Removed {folder}")

        for input_video_path in self.from_video:
            generated_wave = input_video_path.with_suffix(".wav")
            if generated_wave.exists():
                generated_wave.unlink()
                logger.debug(f"Removed generated wav (from video): {generated_wave}")

    @staticmethod
    def to_json(segments: list[Segment], name, lang):
        """
        Convert transcription segments to JSON format and save to file.

        Args:
            segments (List[Segment]): List of transcription segments.
            name (str): Name of the output JSON file.
            lang (str): Language of the transcription.

        Returns:
            dict: The JSON representation of the transcription.

        This method creates a JSON structure from the transcription segments and saves it to a file.
        """
        result = {"language": lang, "segments": []}

        if not segments:
            result["segments"].append({"start": 0.0, "end": 5.0, "text": "no speech found"})
        else:
            for segment in segments:
                result["segments"].append({"start": segment.start, "end": segment.end, "text": segment.text})

        with open(name, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        logger.info(f"File saved to {name}")

        return result

    def pre_process(self, paths, noise_suppress=False):
        """
        Preprocess input audio/video files.

        Args:
            paths (List[Path]): Input file paths
            noise_suppress (bool): Apply noise suppression if True

        Returns:
            List[Path]: Preprocessed audio file paths
        """
        paths = [Path(p) for p in set(paths)]

        for i, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")

            if get_file_type(path) == "video":
                self.from_video.add(path.with_suffix(""))
                audio_path = path.with_suffix(".wav")
                if not audio_path.exists():
                    extract_audio(path)
                paths[i] = audio_path

        from openlrc.preprocess import Preprocessor

        return Preprocessor(paths, options=self.preprocess_options).run(noise_suppress)

    @staticmethod
    def post_process(
        transcribed_sub: Path | Subtitle | BilingualSubtitle,
        output_name: str | Path | None = None,
        remove_files: list[Path] | None = None,
        update_name: bool = False,
        extend_time: bool = False,
    ):
        """
        Post-process the transcribed subtitles.

        Args:
            transcribed_sub: Path or Subtitle object to post-process.
            output_name: Path for the output file.
            remove_files: List of files to remove after processing.
            update_name: Whether to update the subtitle name.
            extend_time: Whether to extend the time of subtitles.

        Returns:
            Subtitle: The post-processed subtitle object.

        This method applies various optimizations to the transcribed subtitles and saves the result.
        """
        optimizer = SubtitleOptimizer(transcribed_sub)
        optimizer.perform_all(extend_time=extend_time)
        optimizer.save(output_name, update_name=update_name)

        # Remove intermediate files
        if remove_files:
            _ = [file.unlink() for file in remove_files if file.is_file()]

        return optimizer.subtitle
