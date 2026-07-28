#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from __future__ import annotations

import concurrent.futures
import json
import shutil
import time
import traceback
import uuid
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from threading import Event, Lock
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from openlrc.context import TranslationBriefInput
    from openlrc.editing import EditResult
    from openlrc.glossary import GlossaryCatalog
    from openlrc.whisper_types import Segment
    from openlrc.workflow import ExecutionContext, RunExecutionStrategy

from openlrc.config import (
    ContextAssistance,
    ContextLLMConfig,
    EditConfig,
    GlossaryOptions,
    HyMT2Mode,
    SubtitleOptimizationMode,
    TranscriptionConfig,
    TranslationConfig,
    context_model_required,
    normalize_hymt2_mode,
)
from openlrc.defaults import (
    BILINGUAL_SUFFIX,
    COMPARE_SUFFIX,
    EDIT_REPORT_SUFFIX,
    EDIT_SESSION_SUFFIX,
    NONTRANS_SUFFIX,
    OPTIMIZED_SUFFIX,
    PREPROCESSED_DIR,
    PREPROCESSED_SUFFIX,
    RELAXED_OPTIMIZED_SUFFIX,
    TRANSCRIBED_SUFFIX,
    TRANSLATED_SUFFIX,
    default_preprocess_options,
    default_whisper_cpp_options,
)
from openlrc.llama_resources import (
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_PORT,
    HY_MT2_7B_PROFILE,
    HY_MT2_PROMPT_PROFILE,
    LOCAL_LLAMA_API_KEY,
)
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
        self,
        *,
        transcription: TranscriptionConfig | None = None,
        translation: TranslationConfig | None = None,
        subtitle_optimization: SubtitleOptimizationMode | str = SubtitleOptimizationMode.AGGRESSIVE,
        execution_context: ExecutionContext | None = None,
    ):
        self._transcription_config = transcription or TranscriptionConfig()
        self._translation_config = translation or TranslationConfig()
        self.subtitle_optimization = SubtitleOptimizationMode(subtitle_optimization)
        self._execution_context = execution_context

        # Translation state
        self.fee_limit = self._translation_config.fee_limit
        self.api_fee = 0  # Can be updated in different thread, operation should be thread-safe
        self.from_video = set()
        self._owned_temp_paths: set[Path] = set()
        from openlrc.glossary import GlossaryService

        self.glossary_options = deepcopy(self._translation_config.glossary_options)
        if self._translation_config.is_force_glossary_used:
            self.glossary_options.force = True
        self.glossary_service = GlossaryService(
            strict=self.glossary_options.strict,
            force=self.glossary_options.force,
            report_matches=self.glossary_options.report_matches,
        )
        self.glossary_catalog, self._glossary_load_conflicts = self.glossary_service.load(
            self._translation_config.glossary
        )
        self.glossary_state = self.glossary_service.merge(
            self.glossary_catalog, load_conflicts=self._glossary_load_conflicts
        )
        self.glossary = self.glossary_state.prompt_mapping() or None
        self.is_force_glossary_used = self.glossary_options.force
        self._translator_engine = self._translation_config._translator_engine
        self.enable_cr = self._translation_config.enable_cr
        self.chunked_guideline = self._translation_config.chunked_guideline
        self.prompt_profile = self._translation_config.prompt_profile
        self.hy_mt2_mode = normalize_hymt2_mode(self._translation_config.hy_mt2_mode)
        from openlrc.context import normalize_translation_brief_input

        self.translation_brief_input = normalize_translation_brief_input(self._translation_config.translation_brief)
        self.context_llm = self._translation_config.context_llm
        self.context_assistance = ContextAssistance(self._translation_config.context_assistance)
        self.edit_config = deepcopy(self._translation_config.edit_config)
        if self.hy_mt2_mode in {HyMT2Mode.NORMAL_PLUS, HyMT2Mode.PRO}:
            self.edit_config.enabled = True

        self._lock = Lock()
        self.exception = None
        self.consumer_thread = self._translation_config.consumer_thread
        self.review_statuses: dict[str, dict] = {}
        self._keep_completed_checkpoint = False
        self._transcription_artifact_primary: bool | None = None
        self._model_load_count = 0
        self._task_started_at = time.perf_counter()

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
        transcription: TranscriptionConfig | None = None,
        subtitle_optimization: SubtitleOptimizationMode | str = SubtitleOptimizationMode.AGGRESSIVE,
        glossary: dict | str | Path | GlossaryCatalog | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
    ) -> LRCer:
        """Create an LRCer configured for local whisper.cpp transcription and llama.cpp translation."""
        return cls(
            transcription=transcription,
            translation=TranslationConfig.local_qwen35_9b(
                model=model,
                idle_timeout=idle_timeout,
                port=port,
                glossary=glossary,
                glossary_options=glossary_options,
                edit_config=edit_config,
            ),
            subtitle_optimization=subtitle_optimization,
        )

    @classmethod
    def local_hy_mt2(
        cls,
        *,
        size: str = HY_MT2_7B_PROFILE,
        model: str | None = None,
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        transcription: TranscriptionConfig | None = None,
        mode: HyMT2Mode | str = HyMT2Mode.FAST,
        context_llm: ContextLLMConfig | None = None,
        context_assistance: ContextAssistance | str = ContextAssistance.AUTO,
        subtitle_optimization: SubtitleOptimizationMode | str = SubtitleOptimizationMode.AGGRESSIVE,
        glossary: dict | str | Path | GlossaryCatalog | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
        translation_brief: TranslationBriefInput | dict | None = None,
    ) -> LRCer:
        """Create an LRCer configured for local Hy-MT2 translation via llama.cpp."""
        return cls(
            transcription=transcription,
            translation=TranslationConfig.local_hy_mt2(
                size=size,
                model=model,
                idle_timeout=idle_timeout,
                port=port,
                mode=mode,
                context_llm=context_llm,
                context_assistance=context_assistance,
                glossary=glossary,
                glossary_options=glossary_options,
                edit_config=edit_config,
                translation_brief=translation_brief,
            ),
            subtitle_optimization=subtitle_optimization,
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
                        execution_context=self._execution_context,
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
                    self._chatbot = create_chatbot(
                        model_config,
                        self.fee_limit,
                        cancellation_token=(
                            self._execution_context.cancellation_token if self._execution_context else None
                        ),
                    )
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
                    self._retry_chatbot = create_chatbot(
                        model_config,
                        self.fee_limit,
                        cancellation_token=(
                            self._execution_context.cancellation_token if self._execution_context else None
                        ),
                    )
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
                    self._cr_chatbot = create_chatbot(
                        model_config,
                        self.fee_limit,
                        cancellation_token=(
                            self._execution_context.cancellation_token if self._execution_context else None
                        ),
                    )
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
                allow_external=self.hy_mt2_mode is HyMT2Mode.FAST,
                execution_context=self._execution_context,
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

    def _check_cancelled(self) -> None:
        if self._execution_context is not None:
            self._execution_context.check_cancelled()

    def _workflow_stage(self, stage: str, *, item: str | Path | None = None):
        if self._execution_context is None:
            return nullcontext()
        from openlrc.workflow import WorkflowStage

        return self._execution_context.stage(WorkflowStage(stage), item=item)

    def _workflow_item(self, item: str | Path | None):
        if self._execution_context is None:
            return nullcontext()
        return self._execution_context.item(item)

    @contextmanager
    def _workflow_file(self, transcribed_path: Path, base_name: str):
        """Scope typed events and review metadata to one unambiguous input."""
        item = str(transcribed_path.resolve())
        with self._workflow_item(item):
            try:
                yield
            finally:
                status = self.review_statuses.get(base_name)
                if status is not None:
                    status["_workflow_item"] = item

    def _workflow_progress(
        self, stage: str, completed: float, total: float, *, item: str | Path | None = None, message: str | None = None
    ) -> None:
        if self._execution_context is not None:
            from openlrc.workflow import WorkflowStage

            self._execution_context.stage_progress(WorkflowStage(stage), completed, total, item=item, message=message)

    def _workflow_artifact(
        self, path: str | Path, kind: str, *, item: str | None = None, primary: bool = False
    ) -> None:
        if self._execution_context is not None:
            from openlrc.workflow import ArtifactKind, WorkflowArtifact

            item = item or self._execution_context.current_item
            self._execution_context.artifact_created(
                WorkflowArtifact(Path(path), ArtifactKind(kind), item=item, primary=primary)
            )

    def _register_owned_path(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve(strict=False)
        self._owned_temp_paths.add(resolved)
        if self._execution_context is not None:
            self._execution_context.register_owned_path(resolved)

    def _owns_path(self, path: str | Path) -> bool:
        resolved = Path(path).expanduser().resolve(strict=False)
        if self._execution_context is not None:
            return self._execution_context.owns_path(resolved)
        return resolved in self._owned_temp_paths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_translator(self, timestamps):
        """Create the classic translator or Hy-MT2's internal lean translator."""
        if self._local_llm_enabled():
            self._model_load_count += 1
        mode = self._translator_engine
        if self.prompt_profile == HY_MT2_PROMPT_PROFILE:
            mode = "lean"
        factories = {"classic": self._create_classic_translator, "lean": self._create_lean_translator}
        factory = factories.get(mode)
        if factory is None:
            raise ValueError(f"Unknown translator engine: {mode!r}. Choose from: {list(factories)}")
        return factory(timestamps)

    def _create_classic_translator(self, timestamps):
        from openlrc.translate import LLMTranslator

        if not self.enable_cr:
            logger.warning("enable_cr is only used in lean mode, ignoring.")
        return LLMTranslator(
            chatbot=self.chatbot,
            retry_chatbot=self.retry_chatbot,
            cr_chatbot=self.cr_chatbot,
            timestamps=timestamps,
            chunked_guideline=self.chunked_guideline,
            execution_context=self._execution_context,
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
            prompt_profile=self.prompt_profile,
            execution_context=self._execution_context,
        )

    @staticmethod
    def parse_glossary(glossary: dict | str | Path | None) -> dict | None:
        """Compatibility wrapper returning the effective task prompt mapping."""
        from openlrc.glossary import GlossaryService

        service = GlossaryService()
        catalog, conflicts = service.load(glossary)
        return service.merge(catalog, load_conflicts=conflicts).prompt_mapping() or None

    def _build_glossary_state(self, *, brief=None, source_language: str, target_language: str):
        catalog = self.glossary_catalog
        if catalog.source_language and catalog.source_language.lower() != source_language.lower():
            raise ValueError(
                f"Glossary source_language {catalog.source_language!r} does not match task language "
                f"{source_language!r}."
            )
        if catalog.target_language and catalog.target_language.lower() != target_language.lower():
            raise ValueError(
                f"Glossary target_language {catalog.target_language!r} does not match task language "
                f"{target_language!r}."
            )
        self.glossary_state = self.glossary_service.merge(
            catalog, brief=brief, load_conflicts=self._glossary_load_conflicts
        )
        self.glossary = self.glossary_state.prompt_mapping() or None
        return self.glossary_state

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
            with self._workflow_stage("transcribe", item=transcribed_path.resolve()), Timer("Transcription process"):
                logger.info(
                    f"Audio length: {audio_path}: {format_timestamp(get_audio_duration(audio_path), fmt='srt')}"
                )
                segments, info = self.transcriber.transcribe(audio_path, language=src_lang)
                logger.info(f"Detected language: {info.language}")
                self._check_cancelled()
                self.to_json(segments, name=transcribed_path, lang=info.language)
                self._register_owned_path(transcribed_path)
        else:
            logger.info(f"Found transcribed json file: {transcribed_path}")
            if self._execution_context is not None:
                from openlrc.workflow import StageOutcome, WorkflowStage

                self._execution_context.stage_completed(
                    WorkflowStage.TRANSCRIBE,
                    outcome=StageOutcome.SKIPPED,
                    item=transcribed_path.resolve(),
                    message="Using cached transcription.",
                )
        transcription_is_primary = self._transcription_artifact_primary
        if transcription_is_primary is None and self._execution_context is not None:
            from openlrc.workflow import WorkflowKind

            transcription_is_primary = self._execution_context.workflow is WorkflowKind.TRANSCRIBE
        self._workflow_artifact(
            transcribed_path,
            "transcription",
            item=str(transcribed_path.resolve()),
            primary=bool(transcription_is_primary),
        )
        return transcribed_path

    def produce_transcriptions(
        self, transcription_queue, audio_paths, src_lang, sentinel_count: int = 1, stop_event: Event | None = None
    ):
        """
        Sequentially produce transcriptions for given audio paths and put them in the queue.

        Args:
            transcription_queue (Queue): Queue to store transcribed paths.
            audio_paths (List[Path]): List of audio file paths to transcribe.
            src_lang (str): Source language for transcription. If None, language will be auto-detected.

        This method processes each audio file sequentially, transcribing it if necessary,
        and puts the path of the transcribed JSON file into the queue.
        """
        try:
            for audio_path in audio_paths:
                self._check_cancelled()
                if stop_event is not None and stop_event.is_set():
                    return
                transcribed_path = self._transcribe_single(audio_path, src_lang)
                transcription_queue.put(transcribed_path)
        finally:
            # The queue is intentionally unbounded, so termination signals cannot
            # block even when a producer or consumer fails.
            for _ in range(sentinel_count):
                transcription_queue.put_nowait(None)
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

        self._transcription_artifact_primary = None

        if isinstance(paths, (str, Path)):
            paths = [paths]

        # Keep behavior aligned with pre_process(): de-duplicate repeated inputs.
        paths = list(dict.fromkeys(Path(p) for p in paths))

        with self._workflow_stage("preprocess"):
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

        if skip_trans:
            shutil.copy(transcribed_opt_sub.filename, final_json_path)
            transcribed_opt_sub.filename = final_json_path
            return transcribed_opt_sub

        try:
            with Timer("Translation process"):
                return self._translate(base_name, target_lang, transcribed_opt_sub, translated_path)
        except Exception as e:
            self.exception = e
            if self._execution_context is not None:
                raise
            return None

    def _has_incomplete_hymt2_review(self, compare_path: Path) -> bool:
        if self.prompt_profile != HY_MT2_PROMPT_PROFILE or self.hy_mt2_mode not in {
            HyMT2Mode.NORMAL_PLUS,
            HyMT2Mode.PRO,
        }:
            return False
        from openlrc.hymt2_pipeline import TranslationBriefAgent, load_checkpoint

        checkpoint = load_checkpoint(compare_path)
        edit_session = checkpoint.get("edit_session") or {}
        resumable_stage = (
            bool(checkpoint.get("review_incomplete"))
            or checkpoint.get("pipeline_stage") in {"review", "editing", "edit_incomplete"}
            or edit_session.get("status") in {"checking", "editing", "incomplete", "failed"}
        )
        return bool(
            resumable_stage
            and checkpoint.get("brief_prompt_version")
            == self._brief_prompt_version(TranslationBriefAgent.PROMPT_VERSION)
            and checkpoint.get("brief_input_fingerprint", "") == self._brief_input_fingerprint()
            and checkpoint.get("translation_brief")
            and checkpoint.get("raw_hymt2_translations")
        )

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
        self._workflow_artifact(result_path, "subtitle", primary=True)

    def _handle_bilingual_subtitles(self, transcribed_path, base_name, transcribed_opt_sub, subtitle_format):
        """
        Generate bilingual subtitles and a non-translated subtitle file.

        Args:
            transcribed_path (Path): Path to the transcribed JSON file.
            base_name (str): Original audio base name.
            transcribed_opt_sub (Subtitle): Post-processed transcription subtitle.
            subtitle_format (str): Output format, either 'lrc' or 'srt'.
        """
        optimized_suffix = (
            RELAXED_OPTIMIZED_SUFFIX
            if self.subtitle_optimization is SubtitleOptimizationMode.RELAXED
            else OPTIMIZED_SUFFIX
        )
        bilingual_subtitle = BilingualSubtitle.from_preprocessed(
            transcribed_path.parent, base_name, optimized_suffix=optimized_suffix
        )
        bilingual_optimizer = SubtitleOptimizer(bilingual_subtitle)
        if self.subtitle_optimization is SubtitleOptimizationMode.AGGRESSIVE:
            bilingual_optimizer.extend_time()

        bilingual_path = getattr(bilingual_subtitle, f"to_{subtitle_format}")()
        bilingual_result = bilingual_path.parent.parent / bilingual_path.name
        shutil.move(bilingual_path, bilingual_result)
        self._workflow_artifact(bilingual_result, "bilingual-subtitle")

        non_translated_subtitle = transcribed_opt_sub
        optimizer = SubtitleOptimizer(non_translated_subtitle)
        if self.subtitle_optimization is SubtitleOptimizationMode.AGGRESSIVE:
            optimizer.extend_time()
        non_translated_path = getattr(non_translated_subtitle, f"to_{subtitle_format}")()
        source_result = non_translated_path.parent.parent / f"{base_name}{NONTRANS_SUFFIX}.{subtitle_format}"
        shutil.move(non_translated_path, source_result)
        self._workflow_artifact(source_result, "source-subtitle")

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
        temporary_dir = transcribed_path.parent
        existing_temporary_paths = set(temporary_dir.iterdir()) if temporary_dir.is_dir() else set()
        try:
            with self._workflow_file(transcribed_path, base_name):
                subtitle_format = "srt" if self._is_video_transcription(transcribed_path, base_name) else "lrc"

                with self._workflow_stage("source-optimize"):
                    transcribed_sub = Subtitle.from_json(transcribed_path)
                    optimized_output = (
                        extend_filename(transcribed_path, RELAXED_OPTIMIZED_SUFFIX)
                        if self.subtitle_optimization is SubtitleOptimizationMode.RELAXED
                        else None
                    )
                    transcribed_opt_sub = self.post_process(
                        transcribed_sub,
                        output_name=optimized_output,
                        update_name=True,
                        mode=self.subtitle_optimization,
                        stage="source",
                    )

                final_subtitle = self._build_final_subtitle(base_name, target_lang, transcribed_opt_sub, skip_trans)
                if final_subtitle is None:
                    return

                with self._workflow_stage("export"):
                    self._generate_subtitle_files(final_subtitle, base_name, subtitle_format)

                    if not skip_trans and bilingual_sub:
                        self._handle_bilingual_subtitles(
                            transcribed_path, base_name, transcribed_opt_sub, subtitle_format
                        )
        finally:
            if temporary_dir.is_dir():
                for candidate in set(temporary_dir.iterdir()) - existing_temporary_paths:
                    if candidate.is_file():
                        self._register_owned_path(candidate)

    def translate(
        self,
        transcribed_paths: Path | list[Path],
        target_lang: str = "zh-cn",
        bilingual_sub: bool = False,
        clear_checkpoint: bool = True,
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
            clear_checkpoint (bool): Remove a completed translation checkpoint.
                Incomplete review checkpoints are always retained. Default is True.

        Returns:
            List[Path]: List of paths to the generated subtitle files.
        """
        self.transcribed_paths = []
        self.exception = None
        self.review_statuses = {}
        # Defer checkpoint cleanup until every input has completed. This keeps
        # earlier files recoverable if a later file fails or is cancelled.
        self._keep_completed_checkpoint = True

        if isinstance(transcribed_paths, Path):
            transcribed_paths = [transcribed_paths]

        logger.info(f"Translating {len(transcribed_paths)} transcribed files: {pformat(transcribed_paths)}")

        completed_checkpoints: list[Path] = []
        has_incomplete_review = False
        for transcribed_path in transcribed_paths:
            self._process_transcribed_file(transcribed_path, target_lang, bilingual_sub=bilingual_sub)

            if self.exception:
                traceback.print_exception(type(self.exception), self.exception, self.exception.__traceback__)
                raise self.exception

            base_name = self._get_base_name(transcribed_path)
            status = self.review_statuses.get(base_name, {})
            has_incomplete_review = has_incomplete_review or bool(status.get("incomplete"))
            completed_checkpoints.append(transcribed_path.parent / f"{base_name}{COMPARE_SUFFIX}.json")

        if clear_checkpoint and not has_incomplete_review:
            for checkpoint in completed_checkpoints:
                if self._owns_path(checkpoint):
                    checkpoint.unlink(missing_ok=True)

        logger.info(f"Total API fee used: {self.api_fee:.4f} USD")

        return self.transcribed_paths

    def edit(
        self,
        source_path: str | Path,
        target_path: str | Path,
        *,
        action: str = "verify",
        segment_ids: list[int] | None = None,
        restore_round: int | None = None,
        session_path: str | Path | None = None,
        output_path: str | Path | None = None,
        markdown_report: bool = False,
    ) -> EditResult:
        """Verify or edit an existing translation without rerunning transcription.

        ``verify`` and ``restore`` are offline. ``review`` uses only the
        explicitly configured context model, while ``retranslate`` uses the
        explicitly configured Hy-MT2 model and, for contextual modes, its
        context model. No model path or credential is inferred from history.
        """
        self._model_load_count = 0
        self._task_started_at = time.perf_counter()
        from openlrc.checkpoint import (
            STANDALONE_EDIT_CHECKPOINT_KIND,
            STANDALONE_EDIT_CHECKPOINT_SCHEMA_VERSION,
            save_json_checkpoint,
        )
        from openlrc.chunking import chunk_plan_signature, plan_translation_chunks
        from openlrc.context import ContextTimeline, TranslateInfo, TranslationBrief
        from openlrc.edit_pipeline import EditPipeline
        from openlrc.edit_validators import DeterministicValidatorSuite
        from openlrc.editing import (
            EditAction,
            EditIssue,
            EditIssueStatus,
            EditPatch,
            EditResult,
            EditRound,
            EditRoundStatus,
            EditSession,
            EditSessionStatus,
            EditSeverity,
            EditStopReason,
            apply_patch_transaction,
            atomic_save_subtitle,
            load_edit_session,
            save_edit_session,
            subtitle_fingerprint,
            translations_at_round,
            write_edit_report,
        )
        from openlrc.hymt2_pipeline import load_checkpoint

        edit_action = EditAction(action)
        if self.translation_brief_input is not None and edit_action in {EditAction.VERIFY, EditAction.RESTORE}:
            raise ValueError(f"{edit_action.value} does not use a Translation Brief.")
        if (
            self.translation_brief_input is not None
            and edit_action is EditAction.RETRANSLATE
            and (self.prompt_profile != HY_MT2_PROMPT_PROFILE or self.hy_mt2_mode is HyMT2Mode.FAST)
        ):
            raise ValueError("A manual Translation Brief requires contextual Hy-MT2 retranslation.")
        source_file = Path(source_path)
        target_file = Path(target_path)
        output_file = Path(output_path) if output_path is not None else target_file
        source = Subtitle.from_json(source_file)
        target = Subtitle.from_json(target_file)
        source_timestamps = [(segment.start, segment.end) for segment in source.segments]
        target_timestamps = [(segment.start, segment.end) for segment in target.segments]
        if len(source) != len(target):
            raise ValueError("Source and target subtitles must contain the same number of segments.")
        if source_timestamps != target_timestamps:
            raise ValueError("Source and target subtitle timelines must match exactly.")

        selected_ids = sorted(set(segment_ids or range(1, len(source) + 1)))
        if not selected_ids or selected_ids[0] < 1 or selected_ids[-1] > len(source):
            raise ValueError("Edit segment IDs must be within the subtitle range.")

        artifact_base = output_file.stem.removesuffix(TRANSLATED_SUFFIX)
        report_suffix = "md" if markdown_report else "json"
        report_path = output_file.parent / f"{artifact_base}{EDIT_REPORT_SUFFIX}.{report_suffix}"
        stable_session_path = (
            Path(session_path)
            if session_path is not None
            else output_file.parent / f"{artifact_base}{EDIT_SESSION_SUFFIX}.json"
        )
        process_checkpoint = output_file.parent / f"{artifact_base}.edit-checkpoint.json"

        if edit_action is EditAction.RESTORE:
            if restore_round is None:
                raise ValueError("restore requires an explicit restore_round.")
            session = load_edit_session(stable_session_path)
            expected_source = subtitle_fingerprint(source.texts, source_timestamps, language=source.lang)
            if session.source_fingerprint != expected_source:
                raise ValueError("Edit session source fingerprint does not match the current source subtitle.")
            if session.source_timestamps != source_timestamps:
                raise ValueError("Edit session timeline does not match the current source subtitle.")
            current_fingerprint = subtitle_fingerprint(target.texts, target_timestamps)
            if session.translation_fingerprint != current_fingerprint:
                raise ValueError("Current translation has changed since the edit session was saved.")
            restored = translations_at_round(session, restore_round)
            changed_ids = [
                index for index, (before, after) in enumerate(zip(target.texts, restored), 1) if before != after
            ]
            atomic_save_subtitle(output_file, language=target.lang, timestamps=target_timestamps, texts=restored)
            restored_session = session.model_copy(
                update={
                    "current_translations": restored,
                    "translation_fingerprint": subtitle_fingerprint(restored, target_timestamps),
                    "status": EditSessionStatus.COMPLETE,
                }
            )
            save_edit_session(stable_session_path, restored_session)
            write_edit_report(report_path, restored_session, markdown=markdown_report)
            return EditResult(
                action=edit_action,
                output_path=output_file,
                report_path=report_path,
                session_path=stable_session_path,
                changed_ids=changed_ids,
                rounds=restored_session.rounds,
                unresolved_issues=restored_session.unresolved_issues,
            )

        context_checkpoint = output_file.parent / f"{artifact_base}.edit-context.json"
        plans = plan_translation_chunks(source.texts, timestamps=source_timestamps)
        plan_signature = chunk_plan_signature(plans, timestamps=source_timestamps)
        expected_source_fingerprint = subtitle_fingerprint(source.texts, source_timestamps, language=source.lang)
        current_target_fingerprint = subtitle_fingerprint(target.texts, target_timestamps)
        process_state = load_checkpoint(process_checkpoint)
        if process_state:
            if process_state.get("checkpoint_kind") != STANDALONE_EDIT_CHECKPOINT_KIND:
                raise ValueError(
                    f"Existing standalone edit checkpoint has an incompatible format: {process_checkpoint}"
                )
            expected = {
                "schema_version": STANDALONE_EDIT_CHECKPOINT_SCHEMA_VERSION,
                "action": edit_action.value,
                "scope_ids": selected_ids,
                "source_fingerprint": expected_source_fingerprint,
                "chunk_signature": plan_signature,
            }
            if self.translation_brief_input is not None:
                expected["brief_input_fingerprint"] = self._brief_input_fingerprint()
            mismatched = [key for key, value in expected.items() if process_state.get(key) != value]
            accepted_target_fingerprints = {
                process_state.get("initial_target_fingerprint"),
                process_state.get("target_fingerprint"),
                process_state.get("snapshot_fingerprint"),
            }
            if process_state.get("edit_session"):
                checkpoint_session = EditSession.model_validate(process_state["edit_session"])
                accepted_target_fingerprints.update(
                    {
                        subtitle_fingerprint(checkpoint_session.raw_translations, target_timestamps),
                        subtitle_fingerprint(checkpoint_session.current_translations, target_timestamps),
                    }
                )
            if current_target_fingerprint not in accepted_target_fingerprints:
                mismatched.append("target_fingerprint")
            if mismatched:
                raise ValueError(
                    "Standalone edit checkpoint does not match the current action/input: " + ", ".join(mismatched)
                )

        def persist_process(session: EditSession | None = None, **updates) -> None:
            nonlocal process_state
            initial_target_fingerprint = process_state.get("initial_target_fingerprint", current_target_fingerprint)
            process_state.update(
                checkpoint_kind=STANDALONE_EDIT_CHECKPOINT_KIND,
                schema_version=STANDALONE_EDIT_CHECKPOINT_SCHEMA_VERSION,
                action=edit_action.value,
                scope_ids=selected_ids,
                source_fingerprint=expected_source_fingerprint,
                initial_target_fingerprint=initial_target_fingerprint,
                target_fingerprint=initial_target_fingerprint,
                snapshot_fingerprint=(
                    session.translation_fingerprint
                    if session is not None
                    else process_state.get("snapshot_fingerprint", current_target_fingerprint)
                ),
                chunk_signature=plan_signature,
                brief_origin=self._brief_origin(),
                brief_input_fields=(
                    self.translation_brief_input.provided_fields if self.translation_brief_input is not None else []
                ),
                brief_input_fingerprint=self._brief_input_fingerprint(),
                **updates,
            )
            if session is not None:
                process_state["edit_session"] = session.model_dump(mode="json")
            save_json_checkpoint(process_checkpoint, process_state)

        def finalize_session(session: EditSession) -> EditResult:
            """Durably finalize a terminal standalone round, including crash recovery."""
            session.translation_fingerprint = subtitle_fingerprint(session.current_translations, target_timestamps)
            changed_ids = [
                index
                for index, (before, after) in enumerate(zip(target.texts, session.current_translations), 1)
                if before != after
            ]
            session.metrics.update(
                elapsed_seconds=round(time.perf_counter() - self._task_started_at, 3),
                model_load_count=self._model_load_count,
                changed_ids=changed_ids,
                modification_ratio=(len(changed_ids) / len(source) if len(source) else 0.0),
            )
            try:
                import resource

                parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                child_peak = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
                session.metrics.update(
                    peak_rss_bytes=max(parent_peak, child_peak),
                    peak_parent_rss_bytes=parent_peak,
                    peak_model_process_rss_bytes=child_peak,
                )
            except (ImportError, OSError):
                session.metrics["peak_rss_bytes"] = None
            model_incomplete = bool(
                session.rounds
                and session.rounds[-1].status is EditRoundStatus.FAILED
                and session.rounds[-1].stop_reason is EditStopReason.MODEL_FAILURE
            )
            if not model_incomplete:
                process_state.pop("model_progress", None)
            # This checkpoint must be durable before the user translation is replaced.
            persist_process(
                session,
                phase=(
                    "complete"
                    if session.status is EditSessionStatus.COMPLETE
                    else ("model-incomplete" if model_incomplete else "finished-incomplete")
                ),
            )
            if edit_action is not EditAction.VERIFY:
                atomic_save_subtitle(
                    output_file, language=target.lang, timestamps=target_timestamps, texts=session.current_translations
                )
            save_edit_session(stable_session_path, session)
            write_edit_report(report_path, session, markdown=markdown_report)
            if session.status is EditSessionStatus.COMPLETE:
                process_checkpoint.unlink(missing_ok=True)
                context_checkpoint.unlink(missing_ok=True)
            return EditResult(
                action=edit_action,
                output_path=output_file,
                report_path=report_path,
                session_path=stable_session_path,
                changed_ids=changed_ids,
                rounds=session.rounds,
                unresolved_issues=session.unresolved_issues,
            )

        brief = (
            TranslationBrief.model_validate(process_state["translation_brief"])
            if process_state.get("translation_brief")
            else None
        )
        timeline = (
            ContextTimeline.model_validate(process_state["context_timeline"])
            if process_state.get("context_timeline")
            else None
        )
        if edit_action is EditAction.REVIEW:
            if self.context_llm is None:
                raise ValueError("review requires an explicitly configured context model.")
            if brief is None:
                from openlrc.hymt2_pipeline import TranslationBriefAgent

                if self.translation_brief_input is not None and self.translation_brief_input.is_complete:
                    brief = TranslationBriefAgent(chatbot=None, src_lang=source.lang, target_lang=target.lang).build(
                        source.texts,
                        title=artifact_base,
                        glossary=self.glossary,
                        translation_brief=self.translation_brief_input,
                    )
                    persist_process(translation_brief=brief.model_dump(mode="json"))
                else:
                    chatbot, server = self._create_context_chatbot()
                    try:
                        brief = TranslationBriefAgent(
                            chatbot=chatbot, src_lang=source.lang, target_lang=target.lang
                        ).build(
                            source.texts,
                            title=artifact_base,
                            glossary=self.glossary,
                            translation_brief=self.translation_brief_input,
                        )
                        self.api_fee += sum(chatbot.api_fees)
                        persist_process(translation_brief=brief.model_dump(mode="json"))
                    finally:
                        chatbot.close()
                        if server is not None:
                            server.close()
        elif edit_action is EditAction.RETRANSLATE:
            if self.prompt_profile != HY_MT2_PROMPT_PROFILE or not self._local_llm_enabled():
                raise ValueError("retranslate requires an explicitly configured local Hy-MT2 model.")
            if self._translation_context_model_required() and self.context_llm is None:
                raise ValueError(f"Hy-MT2 {self.hy_mt2_mode.value} retranslate requires a context model.")
            if self.hy_mt2_mode is HyMT2Mode.PRO and (brief is None or timeline is None):
                brief, timeline, plans, _ = self._prepare_hymt2_pro_context(
                    source.texts,
                    source_timestamps,
                    src_lang=source.lang,
                    target_lang=target.lang,
                    info=TranslateInfo(title=artifact_base, glossary=self.glossary),
                    compare_path=context_checkpoint,
                )
                plan_signature = chunk_plan_signature(plans, timestamps=source_timestamps)
                persist_process(
                    translation_brief=brief.model_dump(mode="json"), context_timeline=timeline.model_dump(mode="json")
                )
            elif self.hy_mt2_mode is not HyMT2Mode.FAST and brief is None:
                brief, _ = self._prepare_hymt2_brief(
                    source.texts,
                    src_lang=source.lang,
                    target_lang=target.lang,
                    info=TranslateInfo(title=artifact_base, glossary=self.glossary),
                    compare_path=context_checkpoint,
                )
                persist_process(translation_brief=brief.model_dump(mode="json"))

        glossary_state = self._build_glossary_state(
            brief=brief, source_language=source.lang, target_language=target.lang
        )
        validators = DeterministicValidatorSuite(
            source, glossary_service=self.glossary_service, glossary_state=glossary_state, brief=brief
        )
        pipeline = EditPipeline(
            source,
            validators,
            glossary_fingerprint=glossary_state.fingerprint,
            max_rounds=self.edit_config.max_rounds,
            restore_enabled=True,
            report_matches=self.glossary_options.report_matches,
            checkpoint_hook=lambda value: persist_process(value),
        )
        if process_state.get("edit_session"):
            session = EditSession.model_validate(process_state["edit_session"])
            if session.source_fingerprint != expected_source_fingerprint:
                raise ValueError("Standalone edit checkpoint session has a stale source fingerprint.")
            if session.source_timestamps != source_timestamps:
                raise ValueError("Standalone edit checkpoint session has a stale timeline.")
            session_snapshot_fingerprint = subtitle_fingerprint(session.current_translations, target_timestamps)
            if session.translation_fingerprint != session_snapshot_fingerprint:
                raise ValueError("Standalone edit checkpoint session has a corrupt translation snapshot.")
            session_input_fingerprint = subtitle_fingerprint(session.raw_translations, target_timestamps)
            if current_target_fingerprint not in {session_input_fingerprint, session_snapshot_fingerprint}:
                raise ValueError("Standalone edit checkpoint session does not match the current translation.")
        else:
            session = pipeline.new_session(target.texts)
            persist_process(session, phase="initialized")

        checkpoint_phase = process_state.get("phase")
        last_round = session.rounds[-1] if session.rounds else None
        terminal_round_was_saved = bool(
            checkpoint_phase == "model-running"
            and last_round is not None
            and not (
                last_round.status is EditRoundStatus.FAILED and last_round.stop_reason is EditStopReason.MODEL_FAILURE
            )
        )
        if checkpoint_phase in {"complete", "finished-incomplete"} or terminal_round_was_saved:
            # No model is loaded for finalization, but the action must still have
            # an explicit current model configuration as required by the API.
            if edit_action is EditAction.REVIEW and self.context_llm is None:
                raise ValueError("review requires an explicitly configured context model.")
            if edit_action is EditAction.RETRANSLATE:
                if self.prompt_profile != HY_MT2_PROMPT_PROFILE or not self._local_llm_enabled():
                    raise ValueError("retranslate requires an explicitly configured local Hy-MT2 model.")
                if self._translation_context_model_required() and self.context_llm is None:
                    raise ValueError(f"Hy-MT2 {self.hy_mt2_mode.value} retranslate requires a context model.")
            return finalize_session(session)

        if edit_action is EditAction.VERIFY:
            session = pipeline.run_deterministic_repair(session, repair=None)
            session.status = (
                EditSessionStatus.INCOMPLETE
                if any(issue.severity is EditSeverity.ERROR for issue in session.unresolved_issues)
                else EditSessionStatus.COMPLETE
            )
        else:
            working_texts = list(session.current_translations)
            before_issues = validators.validate(working_texts)
            progress = process_state.get("model_progress") or {}
            selected_id_set = set(selected_ids)
            patches = [EditPatch.model_validate(item) for item in progress.get("patches", [])]
            semantic_issues = [EditIssue.model_validate(item) for item in progress.get("issues", [])]
            completed_chunks = {int(item) for item in progress.get("completed_chunks", [])}
            failed_chunks: set[int] = set()
            failure_issues: list[EditIssue] = []
            metadata: dict[str, object] = {"action": edit_action.value}
            if edit_action is EditAction.REVIEW:
                from openlrc.hymt2_pipeline import HyMT2RiskReviewAgent

                chatbot, server = self._create_context_chatbot()
                try:
                    reviewer = HyMT2RiskReviewAgent(chatbot=chatbot, src_lang=source.lang, target_lang=target.lang)
                    selected = selected_id_set
                    for plan in plans:
                        active = [line_id for line_id in plan.segment_ids if line_id in selected]
                        if not active or plan.chunk_id in completed_chunks:
                            continue
                        chunk = [(line_id, source.texts[line_id - 1]) for line_id in active]
                        try:
                            chunk_issues, chunk_patches, chunk_metadata = reviewer.review_patches(
                                chunk,
                                {line_id: working_texts[line_id - 1] for line_id in active},
                                brief=brief or TranslationBrief(summary=""),
                                neighboring_context=self._review_neighboring_context(source.texts, chunk),
                                fallback_metadata={},
                            )
                            if any(item.status is EditIssueStatus.FAILED for item in chunk_issues):
                                failed_chunks.add(plan.chunk_id)
                                failure_issues.extend(chunk_issues)
                            else:
                                completed_chunks.add(plan.chunk_id)
                                semantic_issues.extend(chunk_issues)
                                patches.extend(chunk_patches)
                                metadata.update(chunk_metadata)
                        except Exception as exc:
                            failed_chunks.add(plan.chunk_id)
                            failure_issues.append(
                                EditIssue.create(
                                    segment_ids=active,
                                    category="semantic",
                                    severity=EditSeverity.ERROR,
                                    source="standalone-review",
                                    message=f"Standalone review failed for chunk {plan.chunk_id}: {exc}",
                                    evidence={"chunk_id": plan.chunk_id},
                                    status=EditIssueStatus.FAILED,
                                )
                            )
                        persist_process(
                            session,
                            phase="model-running",
                            model_progress={
                                "completed_chunks": sorted(completed_chunks),
                                "failed_chunks": sorted(failed_chunks),
                                "issues": [item.model_dump(mode="json") for item in semantic_issues],
                                "patches": [item.model_dump(mode="json") for item in patches],
                            },
                        )
                    self.api_fee += sum(chatbot.api_fees)
                finally:
                    chatbot.close()
                    if server is not None:
                        server.close()
            else:
                try:
                    with self._local_llm_session():
                        translator = self._create_translator(source_timestamps)
                        completed_results = {
                            int(line_id): value for line_id, value in progress.get("targeted_results", {}).items()
                        }

                        def save_targeted_progress(results: dict[int, str], chunk_id: int) -> None:
                            persist_process(
                                session,
                                phase="model-running",
                                model_progress={
                                    "targeted_results": {str(key): value for key, value in results.items()},
                                    "completed_chunks": sorted(
                                        {
                                            plan.chunk_id
                                            for plan in plans
                                            if all(
                                                line_id in results
                                                for line_id in plan.segment_ids
                                                if line_id in selected_id_set
                                            )
                                            and any(line_id in selected_id_set for line_id in plan.segment_ids)
                                        }
                                    ),
                                    "last_chunk_id": chunk_id,
                                },
                            )

                        try:
                            replacements = translator.translate_targeted(
                                source.texts,
                                working_texts,
                                selected_ids,
                                src_lang=source.lang,
                                target_lang=target.lang,
                                info=TranslateInfo(title=artifact_base, glossary=self.glossary),
                                translation_brief=brief,
                                chunk_plans=plans,
                                context_timeline=timeline,
                                resolved_glossary=glossary_state.merged_entries,
                                completed_results=completed_results,
                                checkpoint_hook=save_targeted_progress,
                            )
                            patches = [
                                EditPatch.create(
                                    issue_ids=[],
                                    segment_id=line_id,
                                    before=working_texts[line_id - 1],
                                    after=replacements[line_id],
                                    reason="Explicit targeted retranslation",
                                    action=EditAction.RETRANSLATE,
                                )
                                for line_id in selected_ids
                            ]
                        except Exception as exc:
                            failure_issues.append(
                                EditIssue.create(
                                    segment_ids=selected_ids,
                                    category="translation",
                                    severity=EditSeverity.ERROR,
                                    source="standalone-retranslate",
                                    message=f"Standalone retranslation failed: {exc}",
                                    evidence={},
                                    status=EditIssueStatus.FAILED,
                                )
                            )
                        metadata["hymt2_metrics"] = dict(translator.metrics)
                finally:
                    self._close_primary_local_stage()

            all_issues = [*before_issues, *semantic_issues, *failure_issues]
            transaction = None
            if failure_issues or failed_chunks:
                stop_reason = EditStopReason.MODEL_FAILURE
                session.current_translations = working_texts
                session.unresolved_issues = all_issues
                session.status = EditSessionStatus.INCOMPLETE
                round_status = EditRoundStatus.FAILED
            elif edit_action is EditAction.REVIEW and not any(
                issue.severity is EditSeverity.ERROR for issue in semantic_issues
            ):
                stop_reason = EditStopReason.NO_HIGH_RISK
                session.current_translations = working_texts
                session.unresolved_issues = before_issues
                session.status = (
                    EditSessionStatus.INCOMPLETE
                    if any(issue.severity is EditSeverity.ERROR for issue in before_issues)
                    else EditSessionStatus.COMPLETE
                )
                round_status = EditRoundStatus.COMPLETED
            elif not patches:
                stop_reason = EditStopReason.NO_EFFECTIVE_PATCH
                session.unresolved_issues = all_issues
                session.status = EditSessionStatus.INCOMPLETE
                round_status = EditRoundStatus.FAILED
            else:
                transaction = apply_patch_transaction(
                    working_texts,
                    patches,
                    scope_ids=selected_ids,
                    baseline_issues=before_issues,
                    validate=validators.validate,
                )
                session.current_translations = transaction.translations
                session.unresolved_issues = [
                    *transaction.validation_after,
                    *[
                        issue.model_copy(update={"status": EditIssueStatus.DEFERRED})
                        for issue in semantic_issues
                        if issue.issue_id
                        not in {issue_id for patch in transaction.applied_patches for issue_id in patch.issue_ids}
                    ],
                ]
                session.status = (
                    EditSessionStatus.COMPLETE
                    if transaction.committed
                    and not any(issue.severity is EditSeverity.ERROR for issue in session.unresolved_issues)
                    else EditSessionStatus.INCOMPLETE
                )
                stop_reason = (
                    EditStopReason.CHECKS_PASSED
                    if transaction.committed and session.status is EditSessionStatus.COMPLETE
                    else EditStopReason.VALIDATION_FAILURE
                )
                round_status = EditRoundStatus.COMMITTED if transaction.committed else EditRoundStatus.FAILED
            session.rounds.append(
                EditRound(
                    round_index=1,
                    status=round_status,
                    scope_ids=selected_ids,
                    issues_before=all_issues,
                    proposed_patches=patches,
                    applied_patches=transaction.applied_patches if transaction is not None else [],
                    rejected_patches=transaction.rejected_patches if transaction is not None else [],
                    validation_after=transaction.validation_after if transaction is not None else before_issues,
                    model_metadata=metadata,
                    stop_reason=stop_reason,
                )
            )
            pipeline._save(session)

        return finalize_session(session)

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
            for future in futures:
                future.result()
        logger.info("Transcription consumer finished.")

    def translation_worker(
        self, transcription_queue, target_lang, skip_trans, bilingual_sub, stop_event: Event | None = None
    ):
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
            try:
                transcribed_path = transcription_queue.get(timeout=0.1)
            except Empty:
                self._check_cancelled()
                if stop_event is not None and stop_event.is_set():
                    return
                continue

            if transcribed_path is None:
                if stop_event is None:
                    # Compatibility path for callers of consume_transcriptions(),
                    # which historically used one self-propagating sentinel.
                    transcription_queue.put_nowait(None)
                logger.debug("Translation worker finished.")
                return

            logger.info(f"Got transcription: {transcribed_path}")

            try:
                self._process_transcribed_file(transcribed_path, target_lang, skip_trans, bilingual_sub)
                if self.exception:
                    raise self.exception
            except Exception:
                if stop_event is not None:
                    stop_event.set()
                raise

            logger.info(f"Translation fee til now: {self.api_fee:.4f} USD")

    def _create_context_chatbot(self):
        """Create the explicitly configured context model and optional owned local server."""
        if self.context_llm is None:
            raise ValueError(f"Hy-MT2 {self.hy_mt2_mode.value!r} mode requires context_llm.")

        from openlrc.agents import create_chatbot

        config = self.context_llm
        model_config = deepcopy(config.chatbot)
        server = None
        if config.local_llm is not None:
            from openlrc.local_llm_server import LocalLLMServer

            local = config.local_llm
            server = LocalLLMServer(
                server_path=local.server_path,
                model_path=local.model_path,
                host=local.host,
                port=local.port,
                alias=local.alias,
                ctx_size=local.ctx_size,
                gpu_layers=local.gpu_layers,
                idle_timeout=0,
                startup_timeout=local.startup_timeout,
                extra_args=local.extra_args,
                allow_external=False,
                execution_context=self._execution_context,
                role="context",
            )
            try:
                model_config.base_url = server.ensure_running(schedule_idle=False)
                model_config.api_key = model_config.api_key or LOCAL_LLAMA_API_KEY
                self._model_load_count += 1
            except Exception:
                server.close()
                raise

        try:
            chatbot = create_chatbot(
                model_config,
                config.fee_limit,
                cancellation_token=(self._execution_context.cancellation_token if self._execution_context else None),
            )
        except Exception:
            if server is not None:
                server.close()
            raise
        return chatbot, server

    def _close_primary_local_stage(self) -> None:
        """Immediately unload an owned Hy-MT2 model between staged pipeline phases."""
        if self._chatbot is not None:
            self._chatbot.close()
            self._chatbot = None
        if self._retry_chatbot is not None:
            self._retry_chatbot.close()
            self._retry_chatbot = None
        if self._local_llm_server is not None:
            self._local_llm_server.close()
            self._local_llm_server = None

    @staticmethod
    def _dump_model(model) -> dict:
        return model.model_dump() if hasattr(model, "model_dump") else model.dict()

    def _brief_origin(self) -> str:
        if self.translation_brief_input is None:
            return "auto"
        return self.translation_brief_input.origin

    def _brief_input_fingerprint(self) -> str:
        return self.translation_brief_input.fingerprint() if self.translation_brief_input is not None else ""

    def _brief_prompt_version(self, automatic_version: int) -> int:
        if self.translation_brief_input is not None and self.translation_brief_input.is_complete:
            return 0
        return automatic_version

    def _brief_checkpoint_metadata(self, automatic_prompt_version: int) -> dict:
        return {
            "brief_origin": self._brief_origin(),
            "brief_input_fields": (
                self.translation_brief_input.provided_fields if self.translation_brief_input is not None else []
            ),
            "brief_input_fingerprint": self._brief_input_fingerprint(),
            "brief_prompt_version": self._brief_prompt_version(automatic_prompt_version),
        }

    def _translation_context_model_required(self) -> bool:
        return context_model_required(
            mode=self.hy_mt2_mode,
            translation_brief=self.translation_brief_input,
            edit_config=self.edit_config,
            context_assistance=self.context_assistance,
        )

    def _prepare_hymt2_brief(self, texts: list[str], *, src_lang: str, target_lang: str, info, compare_path: Path):
        """Build or restore a structured brief before starting Hy-MT2."""
        from openlrc.checkpoint import (
            HYM_T2_CHECKPOINT_KIND,
            HYM_T2_CHECKPOINT_SCHEMA_VERSION,
            migrate_hymt2_checkpoint,
        )
        from openlrc.context import TranslationBrief
        from openlrc.hymt2_pipeline import TranslationBriefAgent, load_checkpoint, save_checkpoint, source_fingerprint

        context_model = (
            self._model_identity(self.context_llm.chatbot, self.context_llm.local_llm)
            if self.context_llm is not None and self._translation_context_model_required()
            else ""
        )
        translation_model = self._model_identity(self._translation_config.chatbot, self._translation_config.local_llm)
        fingerprint = source_fingerprint(
            texts,
            src_lang=src_lang,
            target_lang=target_lang,
            glossary=info.glossary,
            mode=self.hy_mt2_mode.value,
            context_model=context_model,
            translation_model=translation_model,
            translation_brief=self.translation_brief_input,
        )
        legacy_mode = {HyMT2Mode.NORMAL: "context", HyMT2Mode.NORMAL_PLUS: "context-plus"}.get(self.hy_mt2_mode)
        legacy_fingerprints = set()
        if legacy_mode and self.translation_brief_input is None and self.context_llm is not None:
            legacy_context_model = str(self.context_llm.chatbot)
            legacy_translation_model = str(self._translation_config.chatbot)
            legacy_fingerprints.add(
                source_fingerprint(
                    texts,
                    src_lang=src_lang,
                    target_lang=target_lang,
                    glossary=info.glossary,
                    mode=legacy_mode,
                    context_model=legacy_context_model,
                    translation_model=legacy_translation_model,
                    normalize_mode=False,
                )
            )
        metadata = {
            "checkpoint_kind": HYM_T2_CHECKPOINT_KIND,
            "schema_version": HYM_T2_CHECKPOINT_SCHEMA_VERSION,
            "source_fingerprint": fingerprint,
            "mode": self.hy_mt2_mode.value,
            "hymt2_mode": self.hy_mt2_mode.value,
            "glossary_fingerprint": self.glossary_state.fingerprint,
            "edit_protocol_version": 1,
            "context_model": context_model,
            "translation_model": translation_model,
            **self._brief_checkpoint_metadata(TranslationBriefAgent.PROMPT_VERSION),
            "pipeline_stage": "translation",
        }
        checkpoint = migrate_hymt2_checkpoint(
            load_checkpoint(compare_path),
            canonical_mode=self.hy_mt2_mode.value,
            canonical_fingerprint=fingerprint,
            accepted_legacy_fingerprints=legacy_fingerprints,
            glossary_fingerprint=self.glossary_state.fingerprint,
        )
        if (
            checkpoint.get("source_fingerprint") == fingerprint
            and checkpoint.get("brief_prompt_version")
            == self._brief_prompt_version(TranslationBriefAgent.PROMPT_VERSION)
            and checkpoint.get("brief_input_fingerprint", "") == self._brief_input_fingerprint()
            and checkpoint.get("translation_brief")
        ):
            brief = TranslationBrief(**checkpoint["translation_brief"])
            checkpoint.update(metadata, translation_brief=self._dump_model(brief))
            save_checkpoint(compare_path, checkpoint)
            if self._execution_context is not None:
                from openlrc.workflow import StageOutcome, WorkflowStage

                self._execution_context.stage_completed(
                    WorkflowStage.BRIEF,
                    outcome=StageOutcome.RESUMED,
                    message="Translation Brief restored from checkpoint.",
                )
            return brief, {key: value for key, value in checkpoint.items() if key != "compare"}

        with self._workflow_stage("brief"):
            if self.translation_brief_input is not None and self.translation_brief_input.is_complete:
                brief = TranslationBriefAgent(
                    chatbot=None, src_lang=src_lang, target_lang=target_lang, execution_context=self._execution_context
                ).build(
                    texts,
                    title=info.title or "",
                    glossary=info.glossary,
                    translation_brief=self.translation_brief_input,
                )
            else:
                if self.context_llm is None:
                    raise ValueError("Automatic or Partial Translation Brief completion requires a context model.")
                chatbot, server = self._create_context_chatbot()
                try:
                    brief = TranslationBriefAgent(
                        chatbot=chatbot,
                        src_lang=src_lang,
                        target_lang=target_lang,
                        execution_context=self._execution_context,
                    ).build(
                        texts,
                        title=info.title or "",
                        glossary=info.glossary,
                        translation_brief=self.translation_brief_input,
                    )
                    self.api_fee += sum(chatbot.api_fees)
                finally:
                    chatbot.close()
                    if server is not None:
                        server.close()

        checkpoint = {"compare": [], **metadata, "translation_brief": self._dump_model(brief)}
        save_checkpoint(compare_path, checkpoint)
        return brief, {key: value for key, value in checkpoint.items() if key != "compare"}

    @staticmethod
    def _model_identity(chatbot_config, local_config=None) -> str:
        chatbot = asdict(chatbot_config) if chatbot_config is not None else {}
        chatbot.pop("api_key", None)
        local = asdict(local_config) if local_config is not None else None
        return json.dumps({"chatbot": chatbot, "local": local}, ensure_ascii=False, sort_keys=True, default=str)

    def _prepare_hymt2_pro_context(
        self,
        texts: list[str],
        timestamps: list[tuple[float, float | None]],
        *,
        src_lang: str,
        target_lang: str,
        info,
        compare_path: Path,
    ):
        """Build or restore Pro Brief and Timeline before loading Hy-MT2."""
        from openlrc.checkpoint import (
            HYM_T2_CHECKPOINT_KIND,
            HYM_T2_CHECKPOINT_SCHEMA_VERSION,
            migrate_hymt2_checkpoint,
        )
        from openlrc.chunking import CHUNK_PLANNER_VERSION, chunk_plan_signature, plan_translation_chunks
        from openlrc.context import ContextTimeline, TranslationBrief
        from openlrc.hymt2_pipeline import (
            ContextTimelineAgent,
            TranslationBriefAgent,
            load_checkpoint,
            save_checkpoint,
            source_fingerprint,
        )

        assert self.context_llm is not None
        plans = plan_translation_chunks(texts, timestamps=timestamps)
        signature = chunk_plan_signature(plans, timestamps=timestamps)
        context_model = self._model_identity(self.context_llm.chatbot, self.context_llm.local_llm)
        translation_model = self._model_identity(self._translation_config.chatbot, self._translation_config.local_llm)
        fingerprint = source_fingerprint(
            texts,
            src_lang=src_lang,
            target_lang=target_lang,
            glossary=info.glossary,
            mode=self.hy_mt2_mode.value,
            context_model=context_model,
            translation_model=translation_model,
            translation_brief=self.translation_brief_input,
        )
        metadata = {
            "checkpoint_kind": HYM_T2_CHECKPOINT_KIND,
            "schema_version": HYM_T2_CHECKPOINT_SCHEMA_VERSION,
            "source_fingerprint": fingerprint,
            "chunk_signature": signature,
            "chunk_planner_version": CHUNK_PLANNER_VERSION,
            "mode": self.hy_mt2_mode.value,
            "hymt2_mode": self.hy_mt2_mode.value,
            "glossary_fingerprint": self.glossary_state.fingerprint,
            "edit_protocol_version": 1,
            "context_model": context_model,
            "translation_model": translation_model,
            **self._brief_checkpoint_metadata(TranslationBriefAgent.PROMPT_VERSION),
            "timeline_schema_version": ContextTimelineAgent.SCHEMA_VERSION,
            "timeline_prompt_version": ContextTimelineAgent.PROMPT_VERSION,
        }
        checkpoint = migrate_hymt2_checkpoint(
            load_checkpoint(compare_path),
            canonical_mode=self.hy_mt2_mode.value,
            canonical_fingerprint=fingerprint,
            glossary_fingerprint=self.glossary_state.fingerprint,
        )
        is_valid = all(
            (
                checkpoint.get("checkpoint_kind") == HYM_T2_CHECKPOINT_KIND,
                checkpoint.get("schema_version") == HYM_T2_CHECKPOINT_SCHEMA_VERSION,
                checkpoint.get("source_fingerprint") == fingerprint,
                checkpoint.get("chunk_signature") == signature,
                checkpoint.get("chunk_planner_version") == CHUNK_PLANNER_VERSION,
                checkpoint.get("hymt2_mode") == HyMT2Mode.PRO.value,
                checkpoint.get("brief_prompt_version")
                == self._brief_prompt_version(TranslationBriefAgent.PROMPT_VERSION),
                checkpoint.get("brief_input_fingerprint", "") == self._brief_input_fingerprint(),
                checkpoint.get("timeline_schema_version") == ContextTimelineAgent.SCHEMA_VERSION,
                checkpoint.get("timeline_prompt_version") == ContextTimelineAgent.PROMPT_VERSION,
            )
        )
        if not is_valid:
            checkpoint = {"compare": [], **metadata, "pipeline_stage": "context_plan"}
            save_checkpoint(compare_path, checkpoint)
        else:
            checkpoint.update(metadata)

        brief = None
        if checkpoint.get("translation_brief"):
            try:
                brief = TranslationBrief(**checkpoint["translation_brief"])
            except (ValueError, TypeError):
                checkpoint.pop("translation_brief", None)
                checkpoint.pop("context_timeline", None)
                checkpoint.pop("timeline_completed_chunks", None)

        if brief is None:
            checkpoint.pop("context_timeline", None)
            checkpoint.pop("timeline_completed_chunks", None)

        timeline = None
        if brief is not None and checkpoint.get("context_timeline"):
            try:
                candidate = ContextTimeline(**checkpoint["context_timeline"])
                aligned = len(candidate.chunks) == len(plans) and all(
                    context.chunk_id == plan.chunk_id and context.segment_ids == plan.segment_ids
                    for plan, context in zip(plans, candidate.chunks)
                )
                if candidate.chunk_signature == signature and aligned:
                    timeline = candidate
            except (ValueError, TypeError):
                timeline = None

        if brief is None or timeline is None:
            chatbot, server = self._create_context_chatbot()
            try:
                if brief is None:
                    with self._workflow_stage("brief"):
                        brief = TranslationBriefAgent(
                            chatbot=chatbot,
                            src_lang=src_lang,
                            target_lang=target_lang,
                            execution_context=self._execution_context,
                        ).build(
                            texts,
                            title=info.title or "",
                            glossary=info.glossary,
                            translation_brief=self.translation_brief_input,
                        )
                    checkpoint.update(translation_brief=self._dump_model(brief), pipeline_stage="context_plan")
                    save_checkpoint(compare_path, checkpoint)
                elif self._execution_context is not None:
                    from openlrc.workflow import StageOutcome, WorkflowStage

                    self._execution_context.stage_completed(
                        WorkflowStage.BRIEF, outcome=StageOutcome.RESUMED, message="Brief restored."
                    )
                if timeline is None:
                    with self._workflow_stage("timeline"):
                        timeline = ContextTimelineAgent(
                            chatbot=chatbot, src_lang=src_lang, execution_context=self._execution_context
                        ).build(
                            plans,
                            texts=texts,
                            brief=brief,
                            chunk_signature=signature,
                            checkpoint_path=compare_path,
                            checkpoint=checkpoint,
                        )
            finally:
                self.api_fee += sum(chatbot.api_fees)
                chatbot.close()
                if server is not None:
                    server.close()
        elif self._execution_context is not None:
            from openlrc.workflow import StageOutcome, WorkflowStage

            self._execution_context.stage_completed(
                WorkflowStage.BRIEF, outcome=StageOutcome.RESUMED, message="Brief restored."
            )
            self._execution_context.stage_completed(
                WorkflowStage.TIMELINE, outcome=StageOutcome.RESUMED, message="Timeline restored."
            )

        assert brief is not None and timeline is not None
        checkpoint.update(
            metadata,
            translation_brief=self._dump_model(brief),
            context_timeline=self._dump_model(timeline),
            timeline_completed_chunks=[plan.chunk_id for plan in plans],
            pipeline_stage="translation",
        )
        save_checkpoint(compare_path, checkpoint)
        checkpoint_metadata = {key: value for key, value in checkpoint.items() if key != "compare"}
        return brief, timeline, plans, checkpoint_metadata

    @staticmethod
    def _review_neighboring_context(texts: list[str], chunk: list[tuple[int, str]], radius: int = 2) -> str:
        first = chunk[0][0] - 1
        last = chunk[-1][0] - 1
        start = max(0, first - radius)
        end = min(len(texts), last + radius + 1)
        return "\n".join(
            f"[{index + 1}] {texts[index]}" for index in range(start, end) if index < first or index > last
        )

    @staticmethod
    def _validate_review_chunk_ids(value: object, text_count: int) -> list[list[int]]:
        if not isinstance(value, list):
            raise ValueError("review_chunks must be a list.")
        chunks: list[list[int]] = []
        for chunk in value:
            if not isinstance(chunk, list) or not chunk:
                raise ValueError("Every review chunk must be a non-empty ID list.")
            if any(not isinstance(line_id, int) or isinstance(line_id, bool) for line_id in chunk):
                raise ValueError("Review chunk IDs must be integers.")
            chunks.append(list(chunk))
        flattened = [line_id for chunk in chunks for line_id in chunk]
        if flattened != list(range(1, text_count + 1)):
            raise ValueError("Review chunks must cover every source ID exactly once in order.")
        return chunks

    @staticmethod
    def _validate_translation_array(value: object, text_count: int, label: str) -> list[str]:
        if not isinstance(value, list) or len(value) != text_count or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{label} must contain exactly {text_count} string translations.")
        return list(value)

    def _create_edit_pipeline(
        self, source_subtitle: Subtitle, *, target_language: str, brief, compare_path: Path, translations: list[str]
    ):
        """Build the shared validator/edit pipeline and restore compatible session state."""
        from openlrc.checkpoint import HYM_T2_CHECKPOINT_KIND, HYM_T2_CHECKPOINT_SCHEMA_VERSION
        from openlrc.edit_pipeline import EditPipeline
        from openlrc.edit_validators import DeterministicValidatorSuite, validation_fingerprint
        from openlrc.editing import EditSession
        from openlrc.hymt2_pipeline import load_checkpoint, save_checkpoint

        state = self._build_glossary_state(
            brief=brief, source_language=source_subtitle.lang, target_language=target_language
        )
        validators = DeterministicValidatorSuite(
            source_subtitle, glossary_service=self.glossary_service, glossary_state=state, brief=brief
        )

        def persist(session: EditSession) -> None:
            effective_state = validators.effective_glossary_state
            if effective_state is not None:
                self.glossary_state = effective_state
            checkpoint = load_checkpoint(compare_path)
            incomplete = session.status.value in {"incomplete", "failed"}
            checkpoint.update(
                checkpoint_kind=HYM_T2_CHECKPOINT_KIND,
                schema_version=HYM_T2_CHECKPOINT_SCHEMA_VERSION,
                mode=self.hy_mt2_mode.value,
                hymt2_mode=self.hy_mt2_mode.value,
                glossary_state=self.glossary_service.report_state(self.glossary_state).model_dump(mode="json"),
                glossary_fingerprint=self.glossary_state.fingerprint,
                check_fingerprint=validation_fingerprint(
                    glossary_fingerprint=self.glossary_state.fingerprint, brief=brief
                ),
                edit_protocol_version=1,
                edit_session=session.model_dump(mode="json"),
                final_translations=session.current_translations,
                current_snapshot=session.current_translations,
                review_incomplete=incomplete,
                pipeline_stage="edit_incomplete" if incomplete else "editing",
            )
            save_checkpoint(compare_path, checkpoint)

        pipeline = EditPipeline(
            source_subtitle,
            validators,
            glossary_fingerprint=state.fingerprint,
            max_rounds=self.edit_config.max_rounds,
            restore_enabled=self.edit_config.restore_enabled,
            report_matches=self.glossary_options.report_matches,
            checkpoint_hook=persist,
        )
        fresh = pipeline.new_session(translations)
        checkpoint = load_checkpoint(compare_path)
        session = fresh
        if checkpoint.get("edit_session"):
            try:
                candidate = EditSession.model_validate(checkpoint["edit_session"])
                source_is_compatible = candidate.source_fingerprint == fresh.source_fingerprint or (
                    checkpoint.get("checkpoint_kind") == HYM_T2_CHECKPOINT_KIND
                    and candidate.source_fingerprint == checkpoint.get("source_fingerprint")
                )
                if (
                    source_is_compatible
                    and candidate.glossary_fingerprint == fresh.glossary_fingerprint
                    and len(candidate.current_translations) == len(translations)
                ):
                    session = candidate.model_copy(
                        update={
                            "source_fingerprint": fresh.source_fingerprint,
                            "translation_fingerprint": fresh.translation_fingerprint,
                            "source_timestamps": fresh.source_timestamps,
                            "max_rounds": self.edit_config.max_rounds,
                            "restore_enabled": self.edit_config.restore_enabled,
                        }
                    )
            except (TypeError, ValueError):
                logger.warning("Discarding an incompatible edit session from the checkpoint.")
        return pipeline, session

    def _run_deterministic_hymt2_edit(
        self,
        audio_name: str,
        source_subtitle: Subtitle,
        translations: list[str],
        *,
        target_lang: str,
        brief,
        compare_path: Path,
        translator,
        chunk_plans,
        context_timeline,
    ) -> list[str]:
        """Run one batched offline-driven Hy-MT2 repair while that model is resident."""
        from openlrc.context import TranslateInfo

        pipeline, session = self._create_edit_pipeline(
            source_subtitle,
            target_language=target_lang,
            brief=brief,
            compare_path=compare_path,
            translations=translations,
        )

        def repair(ids: list[int], current: list[str]) -> dict[int, str]:
            return translator.translate_targeted(
                source_subtitle.texts,
                current,
                ids,
                src_lang=source_subtitle.lang,
                target_lang=target_lang,
                info=TranslateInfo(
                    title=audio_name,
                    audio_type="Movie",
                    glossary=self.glossary_catalog
                    and {entry.source: entry.target for entry in self.glossary_catalog.entries if entry.enabled},
                    forced_glossary=self.is_force_glossary_used,
                ),
                translation_brief=brief,
                chunk_plans=chunk_plans,
                context_timeline=context_timeline,
                resolved_glossary=self.glossary_state.merged_entries,
            )

        session = pipeline.run_deterministic_repair(
            session, repair=repair if self.edit_config.deterministic_checks else None
        )
        effective_state = pipeline.validators.effective_glossary_state
        if effective_state is not None:
            effective_state = self.glossary_service.check(
                effective_state,
                source_subtitle.texts,
                session.current_translations,
                glossary_removed_retry_chunks=translator.metrics.get("glossary_removed_retry_chunks", []),
            )
            assert pipeline.validators.glossary is not None
            pipeline.validators.glossary.state = effective_state
            self.glossary_state = effective_state
            pipeline._save(session)
        return session.current_translations

    def _run_semantic_hymt2_edit(
        self,
        audio_name: str,
        source_subtitle: Subtitle,
        translations: list[str],
        *,
        src_lang: str,
        target_lang: str,
        brief,
        compare_path: Path,
        translator=None,
        chunk_plans=None,
        context_timeline=None,
    ) -> list[str]:
        """Run bounded semantic rounds with per-chunk progress and transactional patches."""
        from openlrc.editing import EditIssue, EditIssueStatus, EditSessionStatus, EditSeverity
        from openlrc.hymt2_pipeline import HyMT2RiskReviewAgent, load_checkpoint, save_checkpoint

        pipeline, session = self._create_edit_pipeline(
            source_subtitle,
            target_language=target_lang,
            brief=brief,
            compare_path=compare_path,
            translations=translations,
        )
        if not session.current_translations:
            session.current_translations = list(translations)

        plans = chunk_plans
        if plans is None:
            from openlrc.chunking import plan_translation_chunks

            timestamps = [(segment.start, segment.end) for segment in source_subtitle.segments]
            plans = plan_translation_chunks(source_subtitle.texts, timestamps=timestamps)

        if not self.edit_config.semantic_review or self.edit_config.max_rounds == 0:
            session = pipeline.run_semantic_rounds(session, review=None)
            checkpoint = load_checkpoint(compare_path)
            incomplete = session.status is EditSessionStatus.INCOMPLETE
            checkpoint.update(
                edit_session=session.model_dump(mode="json"),
                final_translations=session.current_translations,
                current_snapshot=session.current_translations,
                review_incomplete=incomplete,
                pipeline_stage="edit_incomplete" if incomplete else "complete",
            )
            save_checkpoint(compare_path, checkpoint)
            self.review_statuses[audio_name] = {
                "incomplete": incomplete,
                "failed_chunks": [],
                "reviewed_chunks": 0,
                "total_chunks": len(plans),
                "unresolved_issues": len(session.unresolved_issues),
                "checkpoint": str(compare_path),
            }
            return session.current_translations

        chatbot = None
        server = None
        failed_chunks: set[int] = set()
        try:
            chatbot, server = self._create_context_chatbot()
            reviewer = HyMT2RiskReviewAgent(chatbot=chatbot, src_lang=src_lang, target_lang=target_lang)

            def review(scope: list[int], current: list[str], round_index: int):
                scope_set = set(scope)
                saved_progress = load_checkpoint(compare_path).get("edit_round_progress") or {}
                if saved_progress.get("round_index") == round_index:
                    try:
                        issues = [EditIssue.model_validate(item) for item in saved_progress.get("issues", [])]
                        from openlrc.editing import EditPatch

                        patches = [EditPatch.model_validate(item) for item in saved_progress.get("patches", [])]
                        completed = [int(item) for item in saved_progress.get("completed_chunks", [])]
                    except (TypeError, ValueError):
                        issues, patches, completed = [], [], []
                else:
                    issues, patches, completed = [], [], []
                if completed and self._execution_context is not None:
                    from openlrc.workflow import StageOutcome, WorkflowStage

                    self._execution_context.stage_completed(
                        WorkflowStage.SEMANTIC_REVIEW,
                        outcome=StageOutcome.RESUMED,
                        message=f"Restored {len(completed)} semantic-review chunks for round {round_index}.",
                    )
                failed_chunks.clear()
                for plan in plans:
                    self._check_cancelled()
                    active_ids = [line_id for line_id in plan.segment_ids if line_id in scope_set]
                    if not active_ids or plan.chunk_id in completed:
                        continue
                    chunk = [(line_id, source_subtitle.texts[line_id - 1]) for line_id in active_ids]
                    mapping = {line_id: current[line_id - 1] for line_id in active_ids}
                    try:
                        chunk_context = (
                            context_timeline.context_for(plan.chunk_id) if context_timeline is not None else None
                        )
                        chunk_issues, chunk_patches, metadata = reviewer.review_patches(
                            chunk,
                            mapping,
                            brief=brief,
                            neighboring_context=self._review_neighboring_context(source_subtitle.texts, chunk),
                            fallback_metadata=(
                                translator.metrics
                                if translator is not None
                                else load_checkpoint(compare_path).get("hymt2_metrics", {})
                            ),
                            chunk_context=chunk_context,
                        )
                        issues.extend(chunk_issues)
                        patches.extend(chunk_patches)
                        completed.append(plan.chunk_id)
                    except Exception as exc:
                        self._check_cancelled()
                        failed_chunks.add(plan.chunk_id)
                        issues.append(
                            EditIssue.create(
                                segment_ids=active_ids,
                                category="semantic",
                                severity=EditSeverity.ERROR,
                                source="semantic-review",
                                message=f"Semantic review failed for chunk {plan.chunk_id}: {exc}",
                                evidence={"chunk_id": plan.chunk_id},
                                status=EditIssueStatus.FAILED,
                            )
                        )
                    progress = load_checkpoint(compare_path)
                    progress.update(
                        edit_round_progress={
                            "round_index": round_index,
                            "completed_chunks": completed,
                            "failed_chunks": sorted(failed_chunks),
                            "issues": [
                                item.model_dump(mode="json")
                                for item in issues
                                if item.status is not EditIssueStatus.FAILED
                            ],
                            "patches": [item.model_dump(mode="json") for item in patches],
                        }
                    )
                    save_checkpoint(compare_path, progress)
                    self._workflow_progress(
                        "semantic-review",
                        len(completed) + len(failed_chunks),
                        len(plans),
                        message=f"Semantic round {round_index}",
                    )
                return (
                    issues,
                    patches,
                    {
                        "review_protocol_version": HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION,
                        "model": getattr(chatbot, "model_name", None),
                        "completed_chunks": completed,
                        "failed_chunks": sorted(failed_chunks),
                    },
                )

            session = pipeline.run_semantic_rounds(session, review=review if self.edit_config.semantic_review else None)
            self.api_fee += sum(chatbot.api_fees)
        except Exception as exc:
            self._check_cancelled()
            logger.warning(f"Hy-MT2 semantic editing failed; keeping the current translation: {exc}")
            session.status = EditSessionStatus.INCOMPLETE
            session.unresolved_issues.append(
                EditIssue.create(
                    segment_ids=[],
                    category="semantic",
                    severity=EditSeverity.ERROR,
                    source="semantic-review",
                    message=f"Semantic editing stage failed: {exc}",
                    evidence={},
                    status=EditIssueStatus.FAILED,
                )
            )
            pipeline._save(session)
        finally:
            if chatbot is not None:
                chatbot.close()
            if server is not None:
                server.close()

        checkpoint = load_checkpoint(compare_path)
        incomplete = session.status is EditSessionStatus.INCOMPLETE or bool(failed_chunks)
        if not incomplete:
            checkpoint.pop("edit_round_progress", None)
        checkpoint.update(
            edit_session=session.model_dump(mode="json"),
            final_translations=session.current_translations,
            review_incomplete=incomplete,
            pipeline_stage="edit_incomplete" if incomplete else "complete",
        )
        save_checkpoint(compare_path, checkpoint)
        self.review_statuses[audio_name] = {
            "incomplete": incomplete,
            "failed_chunks": sorted(failed_chunks),
            "reviewed_chunks": sum(
                len(item.model_metadata.get("completed_chunks", [])) for item in session.rounds if item.round_index > 0
            ),
            "total_chunks": len(plans),
            "unresolved_issues": len(session.unresolved_issues),
            "checkpoint": str(compare_path),
        }
        return session.current_translations

    def _finalize_edit_artifacts(
        self,
        audio_name: str,
        source_subtitle: Subtitle,
        target_texts: list[str],
        *,
        target_lang: str,
        brief,
        compare_path: Path,
        output_dir: Path,
    ) -> None:
        """Persist stable edit artifacts outside the temporary preprocessing directory."""
        from openlrc.edit_pipeline import EditPipeline
        from openlrc.edit_validators import DeterministicValidatorSuite
        from openlrc.editing import EditSession, EditSessionStatus, save_edit_session, write_edit_report
        from openlrc.hymt2_pipeline import load_checkpoint

        checkpoint = load_checkpoint(compare_path)
        session = None
        if checkpoint.get("edit_session"):
            try:
                session = EditSession.model_validate(checkpoint["edit_session"])
            except (TypeError, ValueError) as exc:
                logger.warning(f"Could not write edit artifacts from an invalid checkpoint session: {exc}")
        elif self.glossary_catalog.entries:
            state = self._build_glossary_state(
                brief=brief, source_language=source_subtitle.lang, target_language=target_lang
            )
            validators = DeterministicValidatorSuite(
                source_subtitle, glossary_service=self.glossary_service, glossary_state=state, brief=brief
            )
            pipeline = EditPipeline(
                source_subtitle,
                validators,
                glossary_fingerprint=state.fingerprint,
                max_rounds=0,
                report_matches=self.glossary_options.report_matches,
            )
            session = pipeline.run_deterministic_repair(pipeline.new_session(target_texts), repair=None)
            self.glossary_state = validators.effective_glossary_state or state

        if session is None:
            return

        changed_ids = {patch.segment_id for edit_round in session.rounds for patch in edit_round.applied_patches}
        session.metrics.update(
            elapsed_seconds=round(time.perf_counter() - self._task_started_at, 3),
            model_load_count=self._model_load_count,
            changed_ids=sorted(changed_ids),
            modification_ratio=(len(changed_ids) / len(source_subtitle) if len(source_subtitle) else 0.0),
        )
        glossary_metrics = session.metrics.get("glossary") or {}
        required_total = glossary_metrics.get("required_compliant", 0) + glossary_metrics.get(
            "required_noncompliant", 0
        )
        session.metrics["glossary_compliance_rate"] = (
            glossary_metrics.get("required_compliant", 0) / required_total if required_total else 1.0
        )
        try:
            import resource

            parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            child_peak = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            session.metrics.update(
                peak_rss_bytes=max(parent_peak, child_peak),
                peak_parent_rss_bytes=parent_peak,
                peak_model_process_rss_bytes=child_peak,
            )
        except (ImportError, OSError):
            session.metrics["peak_rss_bytes"] = None

        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{audio_name}{EDIT_REPORT_SUFFIX}.json"
        write_edit_report(report_path, session)
        incomplete = session.status in {EditSessionStatus.INCOMPLETE, EditSessionStatus.FAILED}
        status = self.review_statuses.setdefault(audio_name, {})
        status.update(
            incomplete=incomplete,
            unresolved_issues=len(session.unresolved_issues),
            report=str(report_path),
            checkpoint=str(compare_path),
        )
        if self.edit_config.restore_enabled and not incomplete:
            session_path = output_dir / f"{audio_name}{EDIT_SESSION_SUFFIX}.json"
            save_edit_session(session_path, session)
            status["session"] = str(session_path)
            if not self._keep_completed_checkpoint:
                compare_path.unlink(missing_ok=True)
                status["checkpoint"] = None

    def _review_hymt2_translations(
        self,
        audio_name: str,
        texts: list[str],
        translations: list[str],
        *,
        src_lang: str,
        target_lang: str,
        brief,
        compare_path: Path,
        translator=None,
        chunk_plans=None,
        context_timeline=None,
        source_subtitle: Subtitle | None = None,
    ) -> list[str]:
        """Run conservative whole-document risk review and resume per completed chunk."""
        from openlrc.checkpoint import HYM_T2_CHECKPOINT_KIND
        from openlrc.hymt2_pipeline import load_checkpoint

        active_checkpoint = load_checkpoint(compare_path)
        if active_checkpoint.get("checkpoint_kind") == HYM_T2_CHECKPOINT_KIND and source_subtitle is not None:
            return self._run_semantic_hymt2_edit(
                audio_name,
                source_subtitle,
                translations,
                src_lang=src_lang,
                target_lang=target_lang,
                brief=brief,
                compare_path=compare_path,
                translator=translator,
                chunk_plans=chunk_plans,
                context_timeline=context_timeline,
            )
        from openlrc.context import ContextTimeline
        from openlrc.hymt2_pipeline import ContextTimelineAgent, HyMT2RiskReviewAgent, load_checkpoint, save_checkpoint

        checkpoint = load_checkpoint(compare_path)
        if context_timeline is None and checkpoint.get("context_timeline"):
            context_timeline = ContextTimeline(**checkpoint["context_timeline"])
        text_count = len(texts)
        supplied_translations = self._validate_translation_array(translations, text_count, "Review input translations")
        if "raw_hymt2_translations" in checkpoint:
            raw_translations = self._validate_translation_array(
                checkpoint["raw_hymt2_translations"], text_count, "Saved raw Hy-MT2 translations"
            )
        else:
            raw_translations = supplied_translations

        saved_state_invalid = False
        canonical_chunk_ids: list[list[int]] | None = None
        if chunk_plans is not None:
            canonical_chunk_ids = self._validate_review_chunk_ids(
                [[line_id for line_id, _ in plan.pairs()] for plan in chunk_plans], text_count
            )
        elif context_timeline is not None:
            if context_timeline.schema_version != ContextTimelineAgent.SCHEMA_VERSION:
                raise ValueError("ContextTimeline payload schema version is incompatible with review.")
            canonical_chunk_ids = self._validate_review_chunk_ids(
                [list(item.segment_ids) for item in context_timeline.chunks], text_count
            )
        elif translator is not None:
            planned_chunks = translator.make_chunks_by_tokens(texts)
            canonical_chunk_ids = self._validate_review_chunk_ids(
                [[line_id for line_id, _ in chunk] for chunk in planned_chunks], text_count
            )

        saved_chunk_ids = checkpoint.get("review_chunks")
        if canonical_chunk_ids is None and saved_chunk_ids is not None:
            try:
                canonical_chunk_ids = self._validate_review_chunk_ids(saved_chunk_ids, text_count)
            except ValueError:
                saved_state_invalid = True
        elif canonical_chunk_ids is not None and saved_chunk_ids is not None:
            try:
                if self._validate_review_chunk_ids(saved_chunk_ids, text_count) != canonical_chunk_ids:
                    saved_state_invalid = True
            except ValueError:
                saved_state_invalid = True

        if canonical_chunk_ids is None:
            grouped: dict[int, list[int]] = {}
            try:
                for item in checkpoint.get("compare", []):
                    chunk_id = int(item["chunk"])
                    line_id = int(item["idx"])
                    grouped.setdefault(chunk_id, []).append(line_id)
                compare_chunk_ids = [grouped[index] for index in sorted(grouped)]
                canonical_chunk_ids = self._validate_review_chunk_ids(compare_chunk_ids, text_count)
            except (KeyError, TypeError, ValueError):
                canonical_chunk_ids = [list(range(1, text_count + 1))]
                saved_state_invalid = True

        chunks = [[(line_id, texts[line_id - 1]) for line_id in chunk_ids] for chunk_ids in canonical_chunk_ids]
        if context_timeline is not None:
            if context_timeline.schema_version != ContextTimelineAgent.SCHEMA_VERSION:
                raise ValueError("ContextTimeline payload schema version is incompatible with review.")
            timeline_ids = [list(item.segment_ids) for item in context_timeline.chunks]
            if timeline_ids != canonical_chunk_ids or [item.chunk_id for item in context_timeline.chunks] != list(
                range(1, len(chunks) + 1)
            ):
                raise ValueError("ContextTimeline payload does not align with review chunks.")

        expected_chunk_indexes = set(range(1, len(chunks) + 1))
        review_results: dict[str, list[dict]] = {}
        reviewed_chunks: set[int] = set()
        failed_chunks: set[int] = set()
        has_saved_review_state = any(
            key in checkpoint
            for key in ("review_results", "reviewed_chunks", "review_failed_chunks", "final_translations")
        )
        if (
            has_saved_review_state
            and checkpoint.get("review_protocol_version") != HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION
        ):
            saved_state_invalid = True

        if has_saved_review_state and not saved_state_invalid:
            try:
                saved_reviewed = checkpoint.get("reviewed_chunks", [])
                saved_failed = checkpoint.get("review_failed_chunks", [])
                if not isinstance(saved_reviewed, list) or not isinstance(saved_failed, list):
                    raise ValueError("Saved review chunk states must be lists.")
                if any(
                    not isinstance(item, int) or isinstance(item, bool) for item in [*saved_reviewed, *saved_failed]
                ):
                    raise ValueError("Saved review chunk states must contain integer indexes.")
                reviewed_chunks = set(saved_reviewed)
                failed_chunks = set(saved_failed)
                if not reviewed_chunks <= expected_chunk_indexes or not failed_chunks <= expected_chunk_indexes:
                    raise ValueError("Saved review chunk index is out of range.")
                if reviewed_chunks & failed_chunks:
                    raise ValueError("A review chunk cannot be both completed and failed.")

                saved_results = checkpoint.get("review_results", {})
                if not isinstance(saved_results, dict):
                    raise ValueError("Saved review_results must be an object.")
                for raw_chunk_index, items in saved_results.items():
                    chunk_index = int(raw_chunk_index)
                    if chunk_index not in expected_chunk_indexes:
                        raise ValueError("Saved review result chunk index is out of range.")
                    review_results[str(chunk_index)] = HyMT2RiskReviewAgent.validate_saved_items(
                        items, canonical_chunk_ids[chunk_index - 1]
                    )
                if {int(index) for index in review_results} != reviewed_chunks:
                    raise ValueError("Saved review results do not match reviewed_chunks.")

                if "final_translations" in checkpoint:
                    self._validate_translation_array(
                        checkpoint["final_translations"], text_count, "Saved final translations"
                    )
            except (TypeError, ValueError):
                saved_state_invalid = True

        if saved_state_invalid:
            logger.warning("Discarding invalid Hy-MT2 review checkpoint state and re-running review.")
            review_results = {}
            reviewed_chunks = set()
            failed_chunks = set()
            checkpoint.pop("final_translations", None)

        checkpoint["review_chunks"] = canonical_chunk_ids
        checkpoint["review_protocol_version"] = HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION
        output = list(raw_translations)
        checkpoint["raw_hymt2_translations"] = raw_translations

        # Reapply already completed revisions before continuing a resumed review.
        for chunk_index in sorted(reviewed_chunks):
            for item in review_results[str(chunk_index)]:
                if item["risk"] == "high":
                    output[int(item["id"]) - 1] = item["revised_translation"]

        if reviewed_chunks and self._execution_context is not None:
            from openlrc.workflow import StageOutcome, WorkflowStage

            self._execution_context.stage_completed(
                WorkflowStage.SEMANTIC_REVIEW,
                outcome=StageOutcome.RESUMED,
                message=f"Restored {len(reviewed_chunks)} reviewed chunks.",
            )

        if reviewed_chunks == expected_chunk_indexes and not failed_chunks:
            checkpoint.update(review_incomplete=False, pipeline_stage="complete", final_translations=output)
            save_checkpoint(compare_path, checkpoint)
            self.review_statuses[audio_name] = {
                "incomplete": False,
                "failed_chunks": [],
                "reviewed_chunks": len(reviewed_chunks),
                "total_chunks": len(chunks),
                "checkpoint": str(compare_path),
            }
            return output

        chatbot, server = self._create_context_chatbot()
        try:
            reviewer = HyMT2RiskReviewAgent(chatbot=chatbot, src_lang=src_lang, target_lang=target_lang)
            for chunk_index, chunk in enumerate(chunks, 1):
                self._check_cancelled()
                if chunk_index in reviewed_chunks:
                    continue
                mapping = {line_id: output[line_id - 1] for line_id, _ in chunk}
                try:
                    chunk_context = context_timeline.context_for(chunk_index) if context_timeline is not None else None
                    if chunk_context is not None and chunk_context.segment_ids != [line_id for line_id, _ in chunk]:
                        raise ValueError(f"Review chunk {chunk_index} does not align with ContextTimeline.")
                    result = reviewer.review(
                        chunk,
                        mapping,
                        brief=brief,
                        neighboring_context=self._review_neighboring_context(texts, chunk),
                        fallback_metadata=(
                            translator.metrics if translator is not None else checkpoint.get("hymt2_metrics", {})
                        ),
                        chunk_context=chunk_context,
                    )
                    dumped_items = [self._dump_model(item) for item in result.items]
                    review_results[str(chunk_index)] = dumped_items
                    for item in result.items:
                        if item.risk == "high":
                            assert item.revised_translation is not None
                            output[item.id - 1] = item.revised_translation
                    reviewed_chunks.add(chunk_index)
                    failed_chunks.discard(chunk_index)
                except Exception as exc:
                    self._check_cancelled()
                    failed_chunks.add(chunk_index)
                    logger.warning(f"Hy-MT2 risk review failed for chunk {chunk_index}; keeping draft: {exc}")

                incomplete = bool(failed_chunks)
                checkpoint.update(
                    review_results=review_results,
                    reviewed_chunks=sorted(reviewed_chunks),
                    review_failed_chunks=sorted(failed_chunks),
                    review_incomplete=incomplete,
                    pipeline_stage="review",
                    final_translations=output,
                )
                save_checkpoint(compare_path, checkpoint)
                self._workflow_progress(
                    "semantic-review",
                    len(reviewed_chunks) + len(failed_chunks),
                    len(chunks),
                    message="Risk-review chunk",
                )
            self.api_fee += sum(chatbot.api_fees)
        finally:
            chatbot.close()
            if server is not None:
                server.close()

        incomplete = bool(failed_chunks)
        checkpoint.update(
            review_results=review_results,
            reviewed_chunks=sorted(reviewed_chunks),
            review_failed_chunks=sorted(failed_chunks),
            review_incomplete=incomplete,
            pipeline_stage="complete" if not incomplete else "review_incomplete",
            final_translations=output,
        )
        save_checkpoint(compare_path, checkpoint)
        self.review_statuses[audio_name] = {
            "incomplete": incomplete,
            "failed_chunks": sorted(failed_chunks),
            "reviewed_chunks": len(reviewed_chunks),
            "total_chunks": len(chunks),
            "checkpoint": str(compare_path),
        }
        if incomplete:
            logger.warning(f"Hy-MT2 {self.hy_mt2_mode.value} completed with review_incomplete status.")
        return output

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

        self._model_load_count = 0
        self._task_started_at = time.perf_counter()

        if self.translation_brief_input is not None and (
            self.prompt_profile != HY_MT2_PROMPT_PROFILE or self.hy_mt2_mode is HyMT2Mode.FAST
        ):
            raise ValueError("A manual Translation Brief is only supported by contextual Hy-MT2 translation.")
        if (
            self.prompt_profile == HY_MT2_PROMPT_PROFILE
            and self._translation_context_model_required()
            and self.context_llm is None
        ):
            raise ValueError(f"Hy-MT2 {self.hy_mt2_mode.value} requires an explicit context model.")

        self._build_glossary_state(source_language=transcribed_opt_sub.lang, target_language=target_lang)
        context = TranslateInfo(
            title=audio_name, audio_type="Movie", glossary=self.glossary, forced_glossary=self.is_force_glossary_used
        )

        json_filename = Path(translated_path.parent / (audio_name + ".json"))
        compare_path = Path(translated_path.parent, f"{audio_name}{COMPARE_SUFFIX}.json")
        timestamps = [(seg.start, seg.end) for seg in transcribed_opt_sub.segments]
        translation_brief = None
        resume_review = self._has_incomplete_hymt2_review(compare_path)
        resume_plans = None
        force_retranslate = False
        if resume_review:
            from openlrc.chunking import CHUNK_PLANNER_VERSION, chunk_plan_signature, plan_translation_chunks
            from openlrc.hymt2_pipeline import load_checkpoint

            checkpoint = load_checkpoint(compare_path)
            resume_plans = plan_translation_chunks(transcribed_opt_sub.texts, timestamps=timestamps)
            signature = chunk_plan_signature(resume_plans, timestamps=timestamps)
            planner_state_is_valid = (
                checkpoint.get("chunk_signature") == signature
                and checkpoint.get("chunk_planner_version") == CHUNK_PLANNER_VERSION
            )
            try:
                self._validate_translation_array(
                    checkpoint.get("raw_hymt2_translations"),
                    len(transcribed_opt_sub.texts),
                    "Saved raw Hy-MT2 translations",
                )
            except ValueError:
                planner_state_is_valid = False

            if self.hy_mt2_mode is HyMT2Mode.PRO:
                from openlrc.checkpoint import (
                    HYM_T2_CHECKPOINT_KIND,
                    HYM_T2_CHECKPOINT_SCHEMA_VERSION,
                    migrate_hymt2_checkpoint,
                )
                from openlrc.context import ContextTimeline
                from openlrc.hymt2_pipeline import (
                    ContextTimelineAgent,
                    TranslationBriefAgent,
                    save_checkpoint,
                    source_fingerprint,
                )

                assert self.context_llm is not None
                context_model = self._model_identity(self.context_llm.chatbot, self.context_llm.local_llm)
                translation_model = self._model_identity(
                    self._translation_config.chatbot, self._translation_config.local_llm
                )
                fingerprint = source_fingerprint(
                    transcribed_opt_sub.texts,
                    src_lang=transcribed_opt_sub.lang,
                    target_lang=target_lang,
                    glossary=context.glossary,
                    mode=HyMT2Mode.PRO.value,
                    context_model=context_model,
                    translation_model=translation_model,
                    translation_brief=self.translation_brief_input,
                )
                checkpoint = migrate_hymt2_checkpoint(
                    checkpoint,
                    canonical_mode=HyMT2Mode.PRO.value,
                    canonical_fingerprint=fingerprint,
                    glossary_fingerprint=self.glossary_state.fingerprint,
                )
                if checkpoint:
                    save_checkpoint(compare_path, checkpoint)
                timeline_is_valid = False
                try:
                    timeline = ContextTimeline(**checkpoint["context_timeline"])
                    timeline_is_valid = (
                        timeline.chunk_signature == signature
                        and len(timeline.chunks) == len(resume_plans)
                        and all(
                            context_item.chunk_id == plan.chunk_id and context_item.segment_ids == plan.segment_ids
                            for plan, context_item in zip(resume_plans, timeline.chunks)
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    timeline_is_valid = False
                resume_review = (
                    planner_state_is_valid
                    and checkpoint.get("checkpoint_kind") == HYM_T2_CHECKPOINT_KIND
                    and checkpoint.get("schema_version") == HYM_T2_CHECKPOINT_SCHEMA_VERSION
                    and checkpoint.get("hymt2_mode") == HyMT2Mode.PRO.value
                    and checkpoint.get("source_fingerprint") == fingerprint
                    and checkpoint.get("brief_prompt_version")
                    == self._brief_prompt_version(TranslationBriefAgent.PROMPT_VERSION)
                    and checkpoint.get("brief_input_fingerprint", "") == self._brief_input_fingerprint()
                    and checkpoint.get("timeline_schema_version") == ContextTimelineAgent.SCHEMA_VERSION
                    and checkpoint.get("timeline_prompt_version") == ContextTimelineAgent.PROMPT_VERSION
                    and timeline_is_valid
                )
            else:
                from openlrc.checkpoint import (
                    HYM_T2_CHECKPOINT_KIND,
                    HYM_T2_CHECKPOINT_SCHEMA_VERSION,
                    migrate_hymt2_checkpoint,
                )
                from openlrc.hymt2_pipeline import TranslationBriefAgent, save_checkpoint, source_fingerprint

                assert self.context_llm is not None
                context_model = self._model_identity(self.context_llm.chatbot, self.context_llm.local_llm)
                translation_model = self._model_identity(
                    self._translation_config.chatbot, self._translation_config.local_llm
                )
                fingerprint = source_fingerprint(
                    transcribed_opt_sub.texts,
                    src_lang=transcribed_opt_sub.lang,
                    target_lang=target_lang,
                    glossary=context.glossary,
                    mode=self.hy_mt2_mode.value,
                    context_model=context_model,
                    translation_model=translation_model,
                    translation_brief=self.translation_brief_input,
                )
                legacy_fingerprints = set()
                if self.translation_brief_input is None:
                    legacy_fingerprints.add(
                        source_fingerprint(
                            transcribed_opt_sub.texts,
                            src_lang=transcribed_opt_sub.lang,
                            target_lang=target_lang,
                            glossary=context.glossary,
                            mode="context-plus",
                            context_model=str(self.context_llm.chatbot),
                            translation_model=str(self._translation_config.chatbot),
                            normalize_mode=False,
                        )
                    )
                checkpoint = migrate_hymt2_checkpoint(
                    checkpoint,
                    canonical_mode=self.hy_mt2_mode.value,
                    canonical_fingerprint=fingerprint,
                    accepted_legacy_fingerprints=legacy_fingerprints,
                    glossary_fingerprint=self.glossary_state.fingerprint,
                )
                if checkpoint:
                    save_checkpoint(compare_path, checkpoint)
                resume_review = bool(
                    planner_state_is_valid
                    and checkpoint.get("checkpoint_kind") == HYM_T2_CHECKPOINT_KIND
                    and checkpoint.get("schema_version") == HYM_T2_CHECKPOINT_SCHEMA_VERSION
                    and checkpoint.get("hymt2_mode") == HyMT2Mode.NORMAL_PLUS.value
                    and checkpoint.get("source_fingerprint") == fingerprint
                    and checkpoint.get("brief_prompt_version")
                    == self._brief_prompt_version(TranslationBriefAgent.PROMPT_VERSION)
                    and checkpoint.get("brief_input_fingerprint", "") == self._brief_input_fingerprint()
                )
            force_retranslate = not resume_review
        if resume_review:
            from openlrc.context import ContextTimeline, TranslationBrief
            from openlrc.hymt2_pipeline import load_checkpoint

            checkpoint = load_checkpoint(compare_path)
            translation_brief = TranslationBrief(**checkpoint["translation_brief"])
            self._build_glossary_state(
                brief=translation_brief, source_language=transcribed_opt_sub.lang, target_language=target_lang
            )
            context_timeline = (
                ContextTimeline(**checkpoint["context_timeline"]) if checkpoint.get("context_timeline") else None
            )
            saved_session = checkpoint.get("edit_session") or {}
            draft = (
                saved_session.get("current_translations")
                or checkpoint.get("final_translations")
                or checkpoint.get("raw_hymt2_translations")
                or Subtitle.from_json(translated_path).texts
            )
            with self._workflow_stage("semantic-review"):
                target_texts = self._review_hymt2_translations(
                    audio_name,
                    transcribed_opt_sub.texts,
                    draft,
                    src_lang=transcribed_opt_sub.lang,
                    target_lang=target_lang,
                    brief=translation_brief,
                    compare_path=compare_path,
                    context_timeline=context_timeline,
                    chunk_plans=resume_plans,
                    source_subtitle=transcribed_opt_sub,
                )
            translated_sub = deepcopy(transcribed_opt_sub)
            translated_sub.set_texts(target_texts, lang=target_lang)
            translated_sub.save(translated_path, update_name=True)
        elif not translated_path.exists() or force_retranslate:
            context_timeline = None
            chunk_plans = None
            checkpoint_metadata: dict = {}
            if self.prompt_profile == HY_MT2_PROMPT_PROFILE and self.hy_mt2_mode is HyMT2Mode.PRO:
                translation_brief, context_timeline, chunk_plans, checkpoint_metadata = self._prepare_hymt2_pro_context(
                    transcribed_opt_sub.texts,
                    timestamps,
                    src_lang=transcribed_opt_sub.lang,
                    target_lang=target_lang,
                    info=context,
                    compare_path=compare_path,
                )
            elif self.prompt_profile == HY_MT2_PROMPT_PROFILE and self.hy_mt2_mode is not HyMT2Mode.FAST:
                translation_brief, checkpoint_metadata = self._prepare_hymt2_brief(
                    transcribed_opt_sub.texts,
                    src_lang=transcribed_opt_sub.lang,
                    target_lang=target_lang,
                    info=context,
                    compare_path=compare_path,
                )

            if self.prompt_profile == HY_MT2_PROMPT_PROFILE and self.hy_mt2_mode is not HyMT2Mode.FAST:
                from openlrc.chunking import plan_translation_chunks
                from openlrc.hymt2_pipeline import load_checkpoint, save_checkpoint

                chunk_plans = chunk_plans or plan_translation_chunks(transcribed_opt_sub.texts, timestamps=timestamps)
                assert translation_brief is not None
                glossary_state = self._build_glossary_state(
                    brief=translation_brief, source_language=transcribed_opt_sub.lang, target_language=target_lang
                )
                glossary_checkpoint = load_checkpoint(compare_path)
                glossary_checkpoint.update(
                    glossary_state=self.glossary_service.report_state(glossary_state).model_dump(mode="json"),
                    glossary_fingerprint=glossary_state.fingerprint,
                )
                save_checkpoint(compare_path, glossary_checkpoint)

            translator = None
            target_texts: list[str] = []
            translation_complete = False
            if self.hy_mt2_mode is HyMT2Mode.PRO:
                from openlrc.hymt2_pipeline import load_checkpoint

                translation_checkpoint = load_checkpoint(compare_path)
                raw_translations = translation_checkpoint.get("raw_hymt2_translations") or []
                translated_ids = [int(item["idx"]) for item in translation_checkpoint.get("compare", [])]
                translation_complete = len(raw_translations) == len(
                    transcribed_opt_sub.texts
                ) and translated_ids == list(range(1, len(transcribed_opt_sub.texts) + 1))
                if translation_complete:
                    target_texts = list(raw_translations)
                    logger.info("Resuming Pro from completed Hy-MT2 translation; skipping Hy-MT2 model load.")
                    if self._execution_context is not None:
                        from openlrc.workflow import StageOutcome, WorkflowStage

                        self._execution_context.stage_completed(
                            WorkflowStage.TRANSLATE,
                            outcome=StageOutcome.RESUMED,
                            message="Translation restored from checkpoint.",
                        )

            if not translation_complete:
                try:
                    with self._local_llm_session():
                        translator = self._create_translator(timestamps)

                        translate_kwargs = {
                            "src_lang": transcribed_opt_sub.lang,
                            "target_lang": target_lang,
                            "info": context,
                            "compare_path": compare_path,
                        }
                        if self.prompt_profile == HY_MT2_PROMPT_PROFILE:
                            translate_kwargs.update(
                                translation_brief=translation_brief,
                                checkpoint_metadata=checkpoint_metadata,
                                chunk_plans=chunk_plans,
                                context_timeline=context_timeline,
                                resolved_glossary=self.glossary_state.merged_entries,
                            )
                        with self._workflow_stage("translate"):
                            target_texts = translator.translate(transcribed_opt_sub.texts, **translate_kwargs)
                        if (
                            self.prompt_profile == HY_MT2_PROMPT_PROFILE
                            and self.hy_mt2_mode is not HyMT2Mode.FAST
                            and self.edit_config.enabled
                        ):
                            from openlrc.hymt2_pipeline import load_checkpoint, save_checkpoint

                            edit_checkpoint = load_checkpoint(compare_path)
                            edit_checkpoint.update(
                                raw_hymt2_translations=list(target_texts),
                                current_snapshot=list(target_texts),
                                pipeline_stage="deterministic_edit",
                            )
                            save_checkpoint(compare_path, edit_checkpoint)
                            with self._workflow_stage("deterministic-repair"):
                                target_texts = self._run_deterministic_hymt2_edit(
                                    audio_name,
                                    transcribed_opt_sub,
                                    target_texts,
                                    target_lang=target_lang,
                                    brief=translation_brief,
                                    compare_path=compare_path,
                                    translator=translator,
                                    chunk_plans=chunk_plans,
                                    context_timeline=context_timeline,
                                )
                finally:
                    if self.prompt_profile == HY_MT2_PROMPT_PROFILE and self.hy_mt2_mode is not HyMT2Mode.FAST:
                        self._close_primary_local_stage()

            elif (
                self.prompt_profile == HY_MT2_PROMPT_PROFILE
                and self.hy_mt2_mode is not HyMT2Mode.FAST
                and self.edit_config.enabled
            ):
                # A completed translation checkpoint may predate the deterministic
                # stage. Load Hy-MT2 once to finish only that missing work.
                from openlrc.hymt2_pipeline import load_checkpoint

                if not load_checkpoint(compare_path).get("edit_session"):
                    try:
                        with self._local_llm_session():
                            translator = self._create_translator(timestamps)
                            with self._workflow_stage("deterministic-repair"):
                                target_texts = self._run_deterministic_hymt2_edit(
                                    audio_name,
                                    transcribed_opt_sub,
                                    target_texts,
                                    target_lang=target_lang,
                                    brief=translation_brief,
                                    compare_path=compare_path,
                                    translator=translator,
                                    chunk_plans=chunk_plans,
                                    context_timeline=context_timeline,
                                )
                    finally:
                        self._close_primary_local_stage()

            if self.prompt_profile == HY_MT2_PROMPT_PROFILE and self.hy_mt2_mode in {
                HyMT2Mode.NORMAL_PLUS,
                HyMT2Mode.PRO,
            }:
                from openlrc.hymt2_pipeline import load_checkpoint, save_checkpoint

                review_checkpoint = load_checkpoint(compare_path)
                review_checkpoint.setdefault("raw_hymt2_translations", list(target_texts))
                review_checkpoint.update(current_snapshot=list(target_texts), pipeline_stage="review")
                save_checkpoint(compare_path, review_checkpoint)
                assert translation_brief is not None
                with self._workflow_stage("semantic-review"):
                    target_texts = self._review_hymt2_translations(
                        audio_name,
                        transcribed_opt_sub.texts,
                        target_texts,
                        src_lang=transcribed_opt_sub.lang,
                        target_lang=target_lang,
                        brief=translation_brief,
                        compare_path=compare_path,
                        translator=translator,
                        chunk_plans=chunk_plans,
                        context_timeline=context_timeline,
                        source_subtitle=transcribed_opt_sub,
                    )

            if translator is not None:
                with self._lock:
                    self.api_fee += translator.api_fee  # Ensure thread-safe

            translated_sub = deepcopy(transcribed_opt_sub)
            translated_sub.set_texts(target_texts, lang=target_lang)

            # xxx_transcribed_optimized_translated.json
            translated_sub.save(translated_path, update_name=True)
        else:
            logger.info(f"Found translated json file: {translated_path}")
            if self._execution_context is not None:
                from openlrc.workflow import StageOutcome, WorkflowStage

                self._execution_context.stage_completed(
                    WorkflowStage.TRANSLATE, outcome=StageOutcome.SKIPPED, message="Using cached translation."
                )
        translated_sub = Subtitle.from_json(translated_path)

        self._finalize_edit_artifacts(
            audio_name,
            transcribed_opt_sub,
            translated_sub.texts,
            target_lang=target_lang,
            brief=translation_brief,
            compare_path=compare_path,
            output_dir=translated_path.parent.parent,
        )

        with self._workflow_stage("target-optimize"):
            final_subtitle = self.post_process(
                translated_sub,
                output_name=json_filename,
                update_name=True,
                extend_time=True,
                mode=self.subtitle_optimization,
                stage="target",
            )  # xxx.json

        return final_subtitle

    def run(
        self,
        paths: str | Path | list[str | Path],
        src_lang: str | None = None,
        target_lang: str = "zh-cn",
        skip_trans: bool = False,
        noise_suppress: bool = False,
        bilingual_sub: bool = False,
        clear_temp: bool = True,
        clear_checkpoint: bool | None = None,
        skip_preprocess: bool = False,
        execution_strategy: RunExecutionStrategy | str | None = None,
    ) -> list[Path]:
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
            clear_temp (bool): Whether to clear temporary files after complete success.
                               Incomplete review checkpoints are retained. Default is True.
            clear_checkpoint (Optional[bool]): Whether to remove completed translation checkpoints.
                               ``None`` preserves the historical behavior where checkpoint cleanup
                               follows ``clear_temp``. Incomplete review checkpoints are always retained.
            skip_preprocess (bool): Whether to skip the preprocessing step. When True, assumes that
                               preprocessed files already exist at the expected locations (as returned by
                               get_preprocessed_path()). This is useful when preprocessing and transcription
                               are run in separate stages. Default is False.
            execution_strategy: Optional Workflow scheduling strategy. ``None`` preserves the
                               historical producer/consumer behavior used by direct API calls.

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
        self.review_statuses = {}
        self._transcription_artifact_primary = False
        # Successful Run cleanup removes the whole per-input temporary tree.
        # Until that global decision, never delete a completed file's checkpoint.
        self._keep_completed_checkpoint = True

        if not paths:
            logger.warning("No audio/video file given. Skip LRCer.run()")
            return []

        if isinstance(paths, (str, Path)):
            paths = [paths]

        input_paths: list[Path] = list(dict.fromkeys(Path(path) for path in paths))

        with self._workflow_stage("preprocess"):
            if skip_preprocess:
                # Use preprocessed files directly without running preprocessing
                audio_paths = [get_preprocessed_path(p) for p in input_paths]
                for p in audio_paths:
                    if not p.exists():
                        raise FileNotFoundError(
                            f"Preprocessed file not found: {p}. Run pre_process() first or set skip_preprocess=False."
                        )
            else:
                audio_paths = self.pre_process(input_paths, noise_suppress=noise_suppress)

        if skip_trans:
            # Transcribe-only: no translation threads needed
            transcribed_paths = [self._transcribe_single(path, src_lang) for path in audio_paths]
            for transcribed_path in transcribed_paths:
                self._process_transcribed_file(transcribed_path, target_lang=None, skip_trans=True)

            if clear_temp and not skip_preprocess:
                with self._workflow_stage("cleanup"):
                    logger.info("Clearing temporary folder...")
                    self.clear_temp_files(audio_paths)

            return self.transcribed_paths

        logger.info(f"Working on {len(audio_paths)} audio files: {pformat(audio_paths)}")

        from openlrc.workflow import RunExecutionStrategy

        strategy = (
            RunExecutionStrategy.PIPELINE if execution_strategy is None else RunExecutionStrategy(execution_strategy)
        )
        if strategy is RunExecutionStrategy.MEMORY_SAVER:
            with Timer("Memory-saving transcription then translation process"):
                transcribed_paths = [self._transcribe_single(path, src_lang) for path in audio_paths]
                for transcribed_path in transcribed_paths:
                    self._check_cancelled()
                    self._process_transcribed_file(transcribed_path, target_lang, skip_trans, bilingual_sub)
                    if self.exception:
                        raise self.exception
        else:
            transcription_queue = Queue()
            stop_event = Event()

            with Timer("Transcription (Producer) and Translation (Consumer) process"):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.consumer_thread + 1, thread_name_prefix="OpenLRC"
                ) as executor:
                    consumer_futures = [
                        executor.submit(
                            self.translation_worker,
                            transcription_queue,
                            target_lang,
                            skip_trans,
                            bilingual_sub,
                            stop_event,
                        )
                        for _ in range(self.consumer_thread)
                    ]
                    producer_future = executor.submit(
                        self.produce_transcriptions,
                        transcription_queue,
                        audio_paths,
                        src_lang,
                        self.consumer_thread,
                        stop_event,
                    )
                    futures = [producer_future, *consumer_futures]
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
                    failure = next((future.exception() for future in done if future.exception() is not None), None)
                    if failure is not None:
                        stop_event.set()
                    concurrent.futures.wait(futures)
                    if failure is None:
                        failure = next(
                            (future.exception() for future in futures if future.exception() is not None), None
                        )
                    if failure is not None:
                        raise failure

                if self.exception:
                    traceback.print_exception(type(self.exception), self.exception, self.exception.__traceback__)
                    raise self.exception

        logger.info(f"Total API fee used: {self.api_fee:.4f} USD")

        incomplete = [name for name, status in self.review_statuses.items() if status.get("incomplete")]
        if incomplete:
            logger.warning("Keeping temporary files so incomplete Hy-MT2 review can resume: " + ", ".join(incomplete))
        else:
            if clear_checkpoint is True and not clear_temp:
                self._clear_completed_run_checkpoints(audio_paths)
            if clear_temp and not skip_preprocess:
                with self._workflow_stage("cleanup"):
                    logger.info("Clearing temporary folder...")
                    self.clear_temp_files(audio_paths, keep_checkpoints=clear_checkpoint is False)

        input_order = {(path.parent.resolve(), path.stem): index for index, path in enumerate(input_paths)}
        self.transcribed_paths.sort(
            key=lambda output: input_order.get((Path(output).parent.resolve(), Path(output).stem), len(input_order))
        )
        return self.transcribed_paths

    def _clear_completed_run_checkpoints(self, audio_paths: list[Path]) -> None:
        for audio_path in audio_paths:
            transcribed_path = extend_filename(audio_path, TRANSCRIBED_SUFFIX).with_suffix(".json")
            base_name = self._get_base_name(transcribed_path)
            (transcribed_path.parent / f"{base_name}{COMPARE_SUFFIX}.json").unlink(missing_ok=True)

    def clear_temp_files(self, paths: list[Path], *, keep_checkpoints: bool = False) -> None:
        """
        Clear the temporary files generated during the transcription and translation process.

        Args:
            paths (List[Path]): List of paths to the processed audio files.

        This method removes temporary folders and generated wave files from video processing.
        """
        temp_folders: set[Path] = set()
        for path in paths:
            folder = path.parent
            if folder.name != PREPROCESSED_DIR:
                raise ValueError(f"Not a temporary folder: {folder}")
            temp_folders.add(folder)
            base_name = path.stem.removesuffix(PREPROCESSED_SUFFIX)
            exact_names = {
                f"{base_name}.json",
                f"{base_name}.lrc",
                f"{base_name}.srt",
                f"{base_name}{BILINGUAL_SUFFIX}.json",
            }
            if not keep_checkpoints:
                exact_names.add(f"{base_name}{COMPARE_SUFFIX}.json")
            for candidate in folder.iterdir():
                belongs_to_input = candidate.name == path.name or candidate.name.startswith(f"{path.stem}_")
                if (
                    candidate.is_file()
                    and (belongs_to_input or candidate.name in exact_names)
                    and self._owns_path(candidate)
                ):
                    candidate.unlink()
                    logger.debug(f"Removed {candidate}")

        for folder in temp_folders:
            try:
                folder.rmdir()
                logger.debug(f"Removed empty temporary folder {folder}")
            except OSError:
                logger.debug(f"Kept non-empty temporary folder {folder}")

        for input_video_path in self.from_video:
            generated_wave = input_video_path.with_suffix(".wav")
            if generated_wave.exists() and self._owns_path(generated_wave):
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

        output_path = Path(name)
        temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temporary_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                f.flush()
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

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
        paths = list(dict.fromkeys(Path(p) for p in paths))

        for i, path in enumerate(paths):
            self._check_cancelled()
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")

            if get_file_type(path) == "video":
                audio_path = path.with_suffix(".wav")
                if audio_path.exists():
                    raise FileExistsError(
                        f"Cannot process {path}: sidecar {audio_path} already exists and is not owned by this run."
                    )
                extract_audio(path, execution_context=self._execution_context)
                self.from_video.add(path.with_suffix(""))
                self._register_owned_path(audio_path)
                paths[i] = audio_path

        from openlrc.preprocess import Preprocessor

        expected_paths = [get_preprocessed_path(path) for path in paths]
        missing_before = {path for path in expected_paths if not path.exists()}
        processed = Preprocessor(paths, options=self.preprocess_options, execution_context=self._execution_context).run(
            noise_suppress
        )
        for path in processed:
            if path in missing_before and path.exists():
                self._register_owned_path(path)
        return processed

    @staticmethod
    def post_process(
        transcribed_sub: Path | Subtitle | BilingualSubtitle,
        output_name: str | Path | None = None,
        remove_files: list[Path] | None = None,
        update_name: bool = False,
        extend_time: bool = False,
        mode: SubtitleOptimizationMode | str = SubtitleOptimizationMode.AGGRESSIVE,
        stage: Literal["source", "target"] = "source",
    ):
        """
        Post-process the transcribed subtitles.

        Args:
            transcribed_sub: Path or Subtitle object to post-process.
            output_name: Path for the output file.
            remove_files: List of files to remove after processing.
            update_name: Whether to update the subtitle name.
            extend_time: Whether to extend the time of subtitles.
            mode: Aggressive or alignment-safe relaxed optimization.
            stage: Source cleanup before translation or target cleanup afterward.

        Returns:
            Subtitle: The post-processed subtitle object.

        This method applies various optimizations to the transcribed subtitles and saves the result.
        """
        optimizer = SubtitleOptimizer(transcribed_sub)
        optimizer.perform_all(extend_time=extend_time, mode=mode, stage=stage)
        optimizer.save(output_name, update_name=update_name)

        # Remove intermediate files
        if remove_files:
            _ = [file.unlink() for file in remove_files if file.is_file()]

        return optimizer.subtitle
