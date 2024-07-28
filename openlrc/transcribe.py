#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

from pathlib import Path
from typing import NamedTuple, Union, List

import pysbd
from faster_whisper.transcribe import WhisperModel, Segment, BatchedInferencePipeline
from pysbd.languages import LANGUAGE_CODES
from tqdm import tqdm

from openlrc.defaults import default_asr_options, default_vad_options
from openlrc.logger import logger
from openlrc.utils import Timer, get_audio_duration, spacy_load, format_timestamp


class TranscriptionInfo(NamedTuple):
    language: str
    duration: float
    duration_after_vad: float

    @property
    def vad_ratio(self):
        return 1 - self.duration_after_vad / self.duration


class Transcriber:
    def __init__(self, model_name='large-v3', compute_type='float16', device='cuda', vad_filter=True,
                 asr_options=default_asr_options, vad_options=default_vad_options):
        self.model_name = model_name
        self.compute_type = compute_type
        self.device = device
        # self.no_need_align = ['en', 'ja', 'zh']  # Languages that is accurate enough without sentence alignment
        self.continuous_scripted = ['ja', 'zh', 'zh-cn', 'th', 'vi', 'lo', 'km', 'my', 'bo']
        self.asr_options = asr_options
        self.vad_options = vad_options

        model = WhisperModel(model_name, device, compute_type=compute_type, num_workers=1)
        self.whisper_model = BatchedInferencePipeline(model, use_vad_model=vad_filter, **self.vad_options)

    def transcribe(self, audio_path: Union[str, Path], language=None):
        seg_gen, info = self.whisper_model.transcribe(str(audio_path), language=language, **self.asr_options)

        segments = []  # [Segment(start, end, text, words=[Word(start, end, word, probability)])]
        timestamps = 0
        with tqdm(total=int(info.duration), unit=' seconds') as pbar:
            for seg in seg_gen:
                segments.append(seg)
                pbar.update(int(seg.end - timestamps))
                timestamps = seg.end
            if timestamps < info.duration:  # silence at the end of the audio
                pbar.update(info.duration - timestamps)

        assert segments, f'No voice found for {audio_path}'

        with Timer('Sentence Segmentation'):
            result = self.sentence_split(segments, info.language)

        info = TranscriptionInfo(language=info.language, duration=get_audio_duration(audio_path),
                                 duration_after_vad=info.duration_after_vad)

        logger.info(
            f'VAD removed {format_timestamp(info.duration - info.duration_after_vad)}s of silence ({info.vad_ratio}%) ')
        if info.vad_ratio > 0.5:
            logger.warning(f'VAD ratio is too high, check your audio quality. '
                           f'VAD ratio: {info.vad_ratio}, duration: {format_timestamp(info.duration, fmt="srt")}, '
                           f'duration_after_vad: {format_timestamp(info.duration_after_vad, fmt="srt")}. '
                           f'Try to decrease the threshold in vad_options.')

        return result, info

    def sentence_split(self, segments: List[Segment], lang):
        if lang not in LANGUAGE_CODES.keys():
            logger.warning(f'Language {lang} not supported. Skip sentence split.')
            return segments

        nlp = spacy_load(lang)

        def seg_from_words(seg: Segment, seg_id, words, tokens):
            text = ''.join([word.word for word in words])
            return Segment(seg_id, seg.seek, words[0].start, words[-1].end, text, tokens, seg.avg_logprob,
                           seg.compression_ratio, seg.no_speech_prob, words, seg.temperature)

        def mid_split(seg_entry):
            text = seg_entry.text
            doc = nlp(text)

            def is_punct(char):
                return doc.vocab[char].is_punct

            splittable = int(len(text) / 3)

            former_words, former_len = [], 0

            for j, word in enumerate(seg_entry.words):
                former_words.append(word)
                former_len += len(word.word)

                # For non continuous_scripted languages, Whisper may use ' '(space) to split 2 sentence
                if lang in self.continuous_scripted and former_len >= splittable:
                    if word.word.startswith(' '):
                        break
                    elif word.word.endswith(' '):
                        former_words.append(word)
                        former_len += len(word.word)
                        break

                if former_len >= splittable and is_punct(word.word[-1]):
                    break

            latter_words = seg_entry.words[len(former_words):]

            if not latter_words:
                # Directly split using the largest word-word gap
                gaps = [-1]
                for k in range(len(seg_entry.words) - 1):
                    gaps.append(seg_entry.words[k + 1].start - seg_entry.words[k].end)
                max_gap = max(gaps)
                split_idx = gaps.index(max_gap)

                if max_gap >= 2:  # Split using the max gap
                    former_words = seg_entry.words[:split_idx]
                    latter_words = seg_entry.words[split_idx:]
                else:  # Split using hard-mid
                    former_words = seg_entry.words[:len(seg_entry.words) // 2]
                    latter_words = seg_entry.words[len(seg_entry.words) // 2:]

            if not former_words or not latter_words:
                logger.warning(f'Empty former_words or latter_words: {former_words} or {latter_words}, skip')
                return [seg_entry]

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

                if split_words_len >= len(split.rstrip()) + 3:
                    logger.warning(
                        f'Extracted split words len mismatch: {split_words_len} >= {len(split)} + 3')
                if split_words_len == 0:
                    logger.warning(
                        f'Extracted zero split words: {split_words_len} == 0, for split: {split}, skip'
                    )
                    continue

                word_start += len(split_words)

                entry = seg_from_words(segment, id_cnt, split_words,
                                       segment.tokens[word_start: word_start + len(split_words)])

                def recursive_segment(entry):
                    if len(entry.text) < (45 if lang in self.continuous_scripted else 90) or len(entry.words) == 1:
                        if entry.end - entry.start > 10:
                            # split if duration > 10s
                            segmented_entries = mid_split(entry)
                            if len(segmented_entries) == 1:  # if cant be further segmented
                                return [entry]

                            further_segmented = []
                            for segment in segmented_entries:
                                further_segmented.extend(recursive_segment(segment))
                        else:
                            return [entry]
                    else:
                        # Split them in the middle
                        segmented_entries = mid_split(entry)
                        further_segmented = []
                        for segment in segmented_entries:
                            further_segmented.extend(recursive_segment(segment))

                    return further_segmented

                # Check if the sentence is too long in words
                segmented_entries = recursive_segment(entry)

                sentences.extend(segmented_entries)
                id_cnt += len(segmented_entries)

        return sentences
