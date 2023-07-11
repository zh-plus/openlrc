#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import concurrent.futures
import json
from pathlib import Path
from pprint import pformat
from queue import Queue
from threading import Lock
from typing import List

from faster_whisper.transcribe import Segment

from openlrc.context import Context
from openlrc.defaults import default_asr_options, default_vad_options
from openlrc.logger import logger
from openlrc.opt import SubtitleOptimizer
from openlrc.subtitle import Subtitle
from openlrc.transcribe import Transcriber
from openlrc.translate import GPTTranslator
from openlrc.utils import Timer, extend_filename, get_audio_duration, format_timestamp, extract_audio, \
    get_file_type


class LRCer:
    """
     :param model_name: Name of whisper model (tiny, tiny.en, base, base.en, small, small.en, medium,
                    medium.en, large-v1, or large-v2) When a size is configured, the converted model is downloaded
                    from the Hugging Face Hub.
                    Default: ``large-v2``
    :param compute_type: The type of computation to use. Can be ``int8``, ``int8_float16``, ``int16``,
                    ``float16`` or ``float32``.
                    Default: ``float16``
    :param fee_limit: The maximum fee you are willing to pay for one translation call. Default: ``0.1``
    :param consumer_thread: To prevent exceeding the RPM and TPM limits set by OpenAI, the default is TPM/MAX_TOKEN.
    :param asr_options: parameters for whisper model.
    :param vad_options: parameters for VAD model.
    """

    def __init__(self, model_name='large-v2', compute_type='float16', fee_limit=0.1, consumer_thread=11,
                 asr_options=None, vad_options=None):
        self.fee_limit = fee_limit
        self.api_fee = 0  # Can be updated in different thread, operation should be thread-safe
        self.from_video = set()
        self.context: Context = Context()

        self._lock = Lock()
        self.exception = None
        self.consumer_thread = consumer_thread

        # Default Automatic Speech Recognition (ASR) options
        self.asr_options = default_asr_options

        # Parameters for VAD (see faster_whisper.vad.VadOptions), tune them if speech is not being detected
        self.vad_options = default_vad_options

        if asr_options:
            self.asr_options.update(asr_options)

        if vad_options:
            self.vad_options.update(vad_options)

        self.transcriber = Transcriber(model_name=model_name, compute_type=compute_type,
                                       asr_options=self.asr_options, vad_options=self.vad_options)

    def transcription_producer(self, transcription_queue, audio_paths, src_lang):
        """
        Sequential Producer.
        """
        for audio_path in audio_paths:
            transcribed_path = extend_filename(audio_path, '_transcribed').with_suffix('.json')
            if not transcribed_path.exists():
                with Timer('Transcription process'):
                    logger.info(
                        f'Audio length: {audio_path}: {format_timestamp(get_audio_duration(audio_path), fmt="srt")}')
                    segments, info = self.transcriber.transcribe(audio_path, language=src_lang)
                    logger.info(f'Detected language: {info.language}')

                    # [Segment(start, end, text, words=[Word(start, end, word, probability)])]

                # Save the transcribed json
                self.to_json(segments, name=transcribed_path, lang=info.language)  # xxx_transcribed.json
            else:
                logger.info(f'Found transcribed json file: {transcribed_path}')
            transcription_queue.put(transcribed_path)
            # logger.info(f'Put transcription: {transcribed_path}')

        transcription_queue.put(None)
        logger.info('Transcription producer finished.')

    def transcription_consumer(self, transcription_queue, target_lang, prompter, skip_trans):
        """
        Parallel Consumer.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.consumer_worker, transcription_queue, target_lang, prompter, skip_trans)
                       for _ in range(self.consumer_thread)]
            concurrent.futures.wait(futures)
        logger.info('Transcription consumer finished.')

    def consumer_worker(self, transcription_queue, target_lang, prompter, skip_trans):
        """
        Parallel translation.
        """
        while True:
            logger.debug(f'Translation worker waiting transcription...')
            transcribed_path = transcription_queue.get()

            if transcribed_path is None:
                transcription_queue.put(None)
                logger.debug('Translation worker finished.')
                return

            logger.info(f'Got transcription: {transcribed_path}')
            transcribed_sub = Subtitle.from_json(transcribed_path)
            transcribed_opt_sub = self.post_process(transcribed_sub, update_name=True)
            audio_name = transcribed_path.stem.replace('_transcribed', '')

            # xxx_transcribed_optimized_translated.json
            translated_path = extend_filename(transcribed_opt_sub.filename, '_translated')

            final_subtitle = transcribed_opt_sub
            if not skip_trans:
                with Timer('Translation process'):
                    try:
                        final_subtitle = self._translate(audio_name, prompter, target_lang, transcribed_opt_sub,
                                                         translated_path)
                    except Exception as e:
                        self.exception = e

            final_subtitle.filename = Path(translated_path.parent / f'{audio_name}.json')

            final_subtitle.to_lrc()
            if audio_name in self.from_video:
                final_subtitle.to_srt()
            logger.info(f'Translation fee til now: {self.api_fee:.4f} USD')

    def _translate(self, audio_name, prompter, target_lang, transcribed_opt_sub, translated_path):
        json_filename = Path(translated_path.parent / (audio_name + '.json'))
        compare_path = Path(translated_path.parent, f'{audio_name}_compare.json')
        if not translated_path.exists():
            if not compare_path.exists():
                # Translate the transcribed json
                translator = GPTTranslator(prompter=prompter, fee_limit=self.fee_limit)
                context = self.context

                target_texts = translator.translate(
                    transcribed_opt_sub.texts,
                    src_lang=transcribed_opt_sub.lang,
                    target_lang=target_lang,
                    title=audio_name,
                    audio_type=context.audio_type,
                    background=context.background,
                    synopsis=context.get_synopsis(audio_name),
                    compare_path=compare_path
                )

                with self._lock:
                    self.api_fee += translator.api_fee  # Ensure thread-safe
            else:
                logger.info(f'Found compare json file: {compare_path}')
                with open(compare_path, 'r', encoding='utf-8') as f:
                    target_texts = [item['output'] for item in json.load(f)['compare']]
                if len(target_texts) != len(transcribed_opt_sub):
                    logger.error(f'Compare json file {compare_path} is not valid.')
                    raise ValueError(f'Compare json file {compare_path} is not valid.')
            transcribed_opt_sub.set_texts(target_texts, lang=target_lang)

            # xxx_transcribed_optimized_translated.json
            transcribed_opt_sub.save(translated_path, update_name=True)
        else:
            logger.info(f'Found translated json file: {translated_path}')
        translated_sub = Subtitle.from_json(translated_path)

        final_subtitle = self.post_process(translated_sub, output_name=json_filename, t2m=target_lang == 'zh-cn',
                                           update_name=True)  # xxx.json
        return final_subtitle

    def run(self, paths, src_lang=None, target_lang='zh-cn', prompter='base_trans', context_path=None,
            skip_trans=False):
        """
        Split the translation into 2 phases: transcription and translation. They're running in parallel.
        Firstly, transcribe the audios one-by-one. At the same time, translation threads are created and waiting for
        the transcription results. After all the transcriptions are done, the translation threads will start to
        translate the transcribed texts.

        :param paths: Audio/Video paths, can be a list or a single path.
        :param src_lang: Language of the audio, default to auto detect.
        :param target_lang: Target language, default to Mandarin Chinese.
        :param prompter: Currently, only `base_trans` is supported.
        :param context_path: path to context config file. (Default to use `context.yaml` in the first audio's directory)
        :param skip_trans: Skip the translation process if True.
        """
        if not paths:
            logger.warning('No audio/video file given. Skip LRCer.run()')
            return

        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]

        audio_paths = self.pre_process(paths)

        logger.info(f'Working on {len(audio_paths)} audio files: {pformat(audio_paths)}')

        if context_path:
            context_path = Path(context_path)
            self.context.load_config(context_path)
            logger.info(f'Found context config: {context_path}')
            logger.debug(f'Context: {self.context}')
        else:
            # Try to find the default `context.yaml` in the first audio's directory
            try:
                context_path = audio_paths[0].parent / 'context.yaml'
                self.context.load_config(context_path)
                logger.info(f'Found context config file: {context_path}')
            except FileNotFoundError:
                logger.info(f'Default context config not found: {self.context}, using default context.')

        transcription_queue = Queue()

        with Timer('Transcription (Producer) and Translation (Consumer) process'):
            consumer = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='Consumer') \
                .submit(self.transcription_consumer, transcription_queue, target_lang, prompter, skip_trans)
            producer = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='Producer') \
                .submit(self.transcription_producer, transcription_queue, audio_paths, src_lang)

            producer.result()
            consumer.result()

            if self.exception:
                # traceback.print_exception(type(self.exception), self.exception, self.exception.__traceback__)
                raise self.exception

        logger.info(f'Totally used API fee: {self.api_fee:.4f} USD')

    @staticmethod
    def to_json(segments: List[Segment], name, lang):
        result = {
            'generator': 'LRC generated by https://github.com/zh-plus/Open-Lyrics',
            'language': lang,
            'segments': []
        }

        for segment in segments:
            result['segments'].append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })

        with open(name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        logger.info(f'File saved to {name}')

        return result

    def pre_process(self, paths):
        paths = [Path(path) for path in paths]

        # Check if path is audio or video
        for i, path in enumerate(paths):
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f'File not found: {path}')

            paths[i] = extract_audio(path)

            if get_file_type(path) == 'video':
                self.from_video.add(path.name)

        return paths

    @staticmethod
    def post_process(transcribed_sub: Path, output_name: Path = None, remove_files: List[Path] = None, t2m=False,
                     update_name=False):
        optimizer = SubtitleOptimizer(transcribed_sub)
        optimizer.perform_all(t2m=t2m)
        optimizer.save(output_name, update_name=update_name)

        # Remove intermediate files
        if remove_files:
            _ = [file.unlink() for file in remove_files if file.is_file()]

        return optimizer.subtitle
