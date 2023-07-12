#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

from pathlib import Path
from typing import NamedTuple, Union, List

import pysbd
import spacy
from faster_whisper.transcribe import WhisperModel, Segment
from pysbd.languages import LANGUAGE_CODES
from tqdm import tqdm

from openlrc.defaults import default_asr_options, default_vad_options
from openlrc.exceptions import DependencyException
from openlrc.logger import logger
from openlrc.utils import Timer, get_audio_duration, get_spacy_lib


class TranscriptionInfo(NamedTuple):
    language: str
    duration: float


class Transcriber:
    def __init__(self, model_name='large-v2', compute_type='float16', device='cuda',
                 asr_options=default_asr_options, vad_options=default_vad_options):
        self.model_name = model_name
        self.compute_type = compute_type
        self.device = device
        # self.no_need_align = ['en', 'ja', 'zh']  # Languages that is accurate enough without sentence alignment
        self.non_word_boundary = ['ja', 'zh']
        self.asr_options = asr_options
        self.vad_options = vad_options

        self.whisper_model = WhisperModel(model_name, device, compute_type=compute_type, num_workers=2)

    def transcribe(self, audio_path: Union[str, Path], language=None, vad_filter=True):
        seg_gen, info = self.whisper_model.transcribe(str(audio_path), language=language,
                                                      vad_filter=vad_filter, vad_parameters=self.vad_options,
                                                      **self.asr_options)

        segments = []  # [Segment(start, end, text, words=[Word(start, end, word, probability)])]
        timestamps = 0
        with tqdm(total=int(info.duration), unit=' seconds') as pbar:
            for seg in seg_gen:
                segments.append(seg)
                pbar.update(int(seg.end - timestamps))
                timestamps = seg.end
            if timestamps < info.duration:  # silence at the end of the audio
                pbar.update(info.duration - timestamps)

        assert segments, 'No voice found!'

        with Timer('Sentence Segmentation'):
            result = self.sentence_split(segments, info.language)

        info = TranscriptionInfo(language=info.language, duration=get_audio_duration(audio_path))

        return result, info

    def sentence_split(self, segments: List[Segment], lang):
        if lang not in LANGUAGE_CODES.keys():
            logger.warning(f'Language {lang} not supported. Skip sentence split.')
            return segments

        lib_name = get_spacy_lib(lang)
        try:
            nlp = spacy.load(lib_name)
        except (IOError, ImportError, OSError):
            raise DependencyException(
                f'Try `spacy download {lib_name}` to fix.'
                f'Check https://spacy.io/usage for more instruction.')

        def seg_from_words(seg: Segment, seg_id, words, tokens):
            text = ''.join([word.word for word in words])
            return Segment(seg_id, seg.seek, words[0].start, words[-1].end, text, tokens,
                           seg.temperature, seg.avg_logprob, seg.compression_ratio, seg.no_speech_prob, words)

        def mid_split(seg_entry):
            """
            Todo: Split into multiple segments (>2)
            """
            text = seg_entry.text
            doc = nlp(text)

            def is_punct(char):
                return doc.vocab[char].is_punct

            half = len(text) // 2

            former_words, former_len = [], 0

            for j, word in enumerate(seg_entry.words):
                former_words.append(word)
                former_len += len(word.word)
                if former_len >= half and is_punct(word.word[-1]):
                    break

            latter_words = seg_entry.words[len(former_words):]

            if not latter_words:
                # Directly split using the hard-mid
                former_words = seg_entry.words[:len(seg_entry.words) // 2]
                latter_words = seg_entry.words[len(seg_entry.words) // 2:]

            former = seg_from_words(seg_entry, seg_entry.id, former_words, seg_entry.tokens[:len(former_words)])
            latter = seg_from_words(seg_entry, seg_entry.id + 1, latter_words, seg_entry.tokens[len(former_words):])

            return [former, latter]

        segmenter = pysbd.Segmenter(language=lang, clean=False)

        # len(segments['text']) and len(segments['words']) may not same, align here
        # aligned_segments = [Transcriber.word_align(segment) for segment in aligned_result['segments']]

        id_cnt = 0
        sentences = []  # [{'text': , 'start': , 'end': , 'words': [{word: , start: , end: , score: }, ...]}, ...]
        for segment in segments:
            splits = segmenter.segment(segment.text)
            splits = [s for s in filter(None, splits)]  # Filter empty split
            word_start = 0

            for split in splits:
                split_words = []
                split_words_len = 0
                for i in range(len(split)):
                    if word_start + i < len(segment.words):
                        split_words.append(segment.words[word_start + i])
                        split_words_len = len(''.join([word.word for word in split_words]).rstrip())
                    else:
                        logger.warning(f'word_start + i exceed len(segment.words): '
                                       f'{word_start + i} >= {len(segment.words)}, keep split_words: {"".join([word.word for word in split_words])}, discard: {split[split_words_len:]}, skip')
                        break
                    if split_words_len >= len(split.rstrip()):
                        break

                if split_words_len >= len(split.rstrip()) + 2:
                    logger.warning(
                        f'Extracted split words len mismatch: {split_words_len} >= {len(split)} + 2')

                word_start += len(split_words)

                entry = seg_from_words(segment, id_cnt, split_words,
                                       segment.tokens[word_start: word_start + len(split_words)])

                if len(split) < (50 if lang in self.non_word_boundary else 100):
                    sentences.append(entry)
                    id_cnt += 1
                else:
                    # Split them in the middle
                    origin_len = len(sentences)
                    sentences.extend(mid_split(entry))
                    id_cnt += (len(sentences) - origin_len)

        return sentences
