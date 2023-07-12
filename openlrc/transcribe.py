#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import re
from pathlib import Path
from typing import NamedTuple, Dict, Union

import numpy as np
import pysbd
import spacy
import whisperx
from punctuators.models import PunctCapSegModelONNX
from pysbd.languages import LANGUAGE_CODES

from openlrc.exceptions import DependencyException
from openlrc.logger import logger
from openlrc.utils import Timer, release_memory, get_audio_duration, normalize, get_spacy_lib


class TranscriptionInfo(NamedTuple):
    language: str
    duration: float


class Transcriber:
    def __init__(self, model_name='large-v2', compute_type='float16', device='cuda'):
        self.model_name = model_name
        self.compute_type = compute_type
        self.device = device
        self.no_need_align = ['en', 'ja', 'zh']  # Languages that is accurate enough without sentence alignment
        self.non_word_boundary = ['ja', 'zh']

    def transcribe(self, audio_path: Union[str, Path], batch_size=8, language=None, asr_options=None, vad_options=None):
        whisper_model = whisperx.load_model(self.model_name, language=language, compute_type=self.compute_type,
                                            device=self.device,
                                            asr_options=asr_options, vad_options=vad_options)
        audio = whisperx.load_audio(str(audio_path))

        with Timer('Base Whisper Transcription'):
            result = whisper_model.transcribe(audio, batch_size=batch_size)

        release_memory(whisper_model)

        with Timer('Phoneme Alignment'):
            align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, self.device,
                                            return_char_alignments=False)
            # {
            #   'segments': [{start: 0.0, end: 0.5, text: 'hello', words: [{word: , start: , end: }]}, ...],
            #   'word_segments': [{start: , end: , word:}]
            # }
        release_memory(align_model)

        if self.need_sentence_align(aligned_result, result["language"]):
            with Timer('Sentence Alignment'):
                pcs_result = self.sentence_align(aligned_result)
                # TODO: Unify the pcs_result and aligned_result
                # {'sentences': [{text: , start: , end:}, ...]}
        else:
            with Timer('Sentence Segmentation'):
                if language in self.non_word_boundary:
                    aligned_result = self.sentence_split(aligned_result, result["language"])
                # TODO: add sentence_split for word_boundary languages, e.g. en.
                pcs_result = {'sentences': aligned_result['segments']}

        info = TranscriptionInfo(language=result['language'], duration=get_audio_duration(audio_path))

        return pcs_result, info

    @staticmethod
    def sentence_split(aligned_result, lang):
        if lang not in LANGUAGE_CODES.keys():
            logger.warning(f'Language {lang} not supported. Skip sentence split.')
            return aligned_result

        lib_name = get_spacy_lib(lang)
        try:
            nlp = spacy.load(lib_name)
        except (IOError, ImportError, OSError):
            raise DependencyException(
                f'Try `spacy download {lib_name}` to fix.'
                f'Check https://spacy.io/usage for more instruction.')

        def mid_split(seg_entry):
            text = seg_entry['text']
            doc = nlp(text)
            # Use w.is_punct for punctuation

            half = len(text) // 2
            punct_idxes = [i for i, word in enumerate(text) if doc.vocab[word].is_punct]

            if len(punct_idxes) <= 1:
                # Only the last punct or no punct, return directly from the mid
                split_idx = half
                while split_idx > 0 and 'end' not in seg_entry['words'][split_idx].keys():
                    split_idx -= 1

                if split_idx == 0:
                    logger.warning(
                        f'No valid word to the left of {seg_entry["text"][half]}, skip: {seg_entry["text"]}')
                    return [seg_entry]

                former = {
                    'text': seg_entry['text'][:half],
                    'start': seg_entry['start'],
                    'end': seg_entry['words'][split_idx]['end'],
                    'words': seg_entry['words'][:half]
                }

                split_idx = half
                while split_idx < len(text) - 1 and 'start' not in seg_entry['words'][split_idx].keys():
                    split_idx += 1

                if split_idx == len(seg_entry['words']) or 'start' not in seg_entry['words'][split_idx].keys():
                    logger.warning(
                        f'No valid word to the right of {seg_entry["text"][half]}, skip: {seg_entry["text"]}')
                    return [seg_entry]

                latter = {
                    'text': seg_entry['text'][half:],
                    'start': seg_entry['words'][split_idx]['start'],
                    'end': seg_entry['end'],
                    'words': seg_entry['words'][half:]
                }

            else:
                # Multiple punct, find the nearest one
                distance = np.abs(np.array(punct_idxes) - half)
                nearest_idx = punct_idxes[np.argmin(distance)]

                # Split using the first valid seg_entry['words'] in the right part
                split_idx = nearest_idx
                while split_idx < len(seg_entry['words']) and 'end' not in seg_entry['words'][split_idx].keys():
                    split_idx += 1

                former_end_idx = nearest_idx
                while former_end_idx > 0 and 'end' not in seg_entry['words'][former_end_idx].keys():
                    former_end_idx -= 1

                if split_idx == len(seg_entry['words']):
                    # No valid word to the right, skip
                    logger.warning(
                        f'No valid word to the right of {seg_entry["text"][nearest_idx]}, skip: {seg_entry["text"]}')
                    return [seg_entry]
                elif former_end_idx == 0:
                    # No valid word to the left, skip
                    logger.warning(
                        f'No valid word to the left of {seg_entry["text"][nearest_idx]}, skip: {seg_entry["text"]}')
                    return [seg_entry]

                former = {
                    'text': seg_entry['text'][:split_idx],
                    'start': seg_entry['start'],
                    'end': seg_entry['words'][former_end_idx]['end'],
                    'words': seg_entry['words'][:split_idx]
                }

                latter = {
                    'text': seg_entry['text'][split_idx:],
                    'start': seg_entry['words'][split_idx]['start'],
                    'end': seg_entry['end'],
                    'words': seg_entry['words'][split_idx:]
                }

            return [former, latter]

        segmenter = pysbd.Segmenter(language=lang, clean=False)

        # len(segments['text']) and len(segments['words']) may not same, align here
        aligned_segments = [Transcriber.word_align(segment) for segment in aligned_result['segments']]

        sentences = []  # [{'text': , 'start': , 'end': , 'words': [{word: , start: , end: , score: }, ...]}, ...]
        for segment in aligned_segments:
            splits = segmenter.segment(segment['text'])
            splits = list(filter(None, splits))  # Filter empty split
            split_start = 0

            if not segment['words']:
                continue

            for split in splits:
                start_idx = split_start
                while start_idx < split_start + len(split) and 'start' not in segment['words'][start_idx].keys():
                    start_idx += 1

                end_idx = split_start + len(split) - 1
                while end_idx >= start_idx and 'end' not in segment['words'][end_idx].keys():
                    end_idx -= 1

                if end_idx < start_idx:
                    logger.warning(f'End idx < Start idx for sentence split: {split}, among {segment["text"]}')
                end_idx = max(start_idx, end_idx)  # Ensure end >= start

                entry = {
                    'text': split,
                    'start': segment['words'][start_idx]['start'],
                    'end': segment['words'][end_idx]['end'],
                    'words': segment['words'][split_start: split_start + len(split)]
                }

                if len(split) < 50:
                    sentences.append(entry)
                else:
                    # Split them in the middle
                    sentences.extend(mid_split(entry))

                split_start += len(split)

        aligned_result['segments'] = sentences

        return aligned_result

    def need_sentence_align(self, aligned_result: Dict[str, list], language: str):
        if language in self.no_need_align:
            return False

        word_segments = aligned_result['word_segments']
        if len(word_segments) == 0:
            return False

        avg_word_len = sum([len(word['word']) for word in word_segments]) / len(word_segments)
        return avg_word_len <= 1.5

    @staticmethod
    def sentence_align(aligned_result):
        """
        Align the word-level whisper transcribe result to sentence-level.
        Should be only used by those languages that whisper cannot produce punctuations.

        TODO: Simplify (or remove) this function.

        :return A dict with key 'sentences' and value a list of dict with key 'text', 'start_word', 'end_word'.
        """
        pcs_model: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
            "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
        )
        punctuations = '.,?？，。、・।؟;።፣፧،'

        sentences_list = pcs_model.infer([segment['text'] for segment in aligned_result['segments']], apply_sbd=True)

        # len(segments['text']) and len(segments['words']) may not same, align here
        aligned_segments = [Transcriber.word_align(segment) for segment in aligned_result['segments']]

        pcs_result = {'sentences': []}
        for segment, sentences in zip(aligned_segments, sentences_list):
            last_end_idx = 0
            for sentence in sentences:
                if '<unk>' in sentence.lower():
                    logger.error(f'Unknown token in sentence: {sentence}')
                    if len(sentence.lower().replace('<unk>', '').strip()) <= 3:
                        logger.warning(f'Unknown token in sentence: {sentence} is too short, skip')
                        continue
                    else:
                        logger.warning(f'Unknown token in sentence: {sentence} is still long, keep')

                stc_split = re.split(f'[{punctuations} ]', sentence)

                # Remove empty string
                stc_split = [split for split in stc_split if split]

                if not stc_split:
                    continue

                # Normalize text, head and tail
                text = normalize(segment['text'])
                head = normalize(stc_split[0])
                tail = normalize(stc_split[-1])
                # check if first and last is substring of sentence
                if head not in text:
                    logger.error(f'First split: {head} not in {text}, skip')
                    continue

                if tail not in text:
                    logger.error(f'Last split: {head} not in {text}, skip')
                    continue

                # Locate the start and end split in transcribed sentences
                start_idx = text.find(head, last_end_idx)
                if len(stc_split) == 1:
                    end_idx = start_idx + len(head) - 1
                else:
                    start_find = last_end_idx + len(''.join(stc_split[:-1]))
                    end_idx = text.find(head, start_find) + len(tail) - 1

                start_idx = max(start_idx, 0)  # ensure start_idx is not out of range
                end_idx = min(end_idx, len(segment['words']) - 1)  # ensure end_idx is not out of range

                if not segment['words']:
                    start = segment['start']
                    end = segment['end']
                    pcs_result['sentences'].append({'text': sentence, 'start': start, 'end': end})
                    continue

                # start word and end word should not be punctuation
                while start_idx < len(segment['words']) and 'start' not in segment['words'][start_idx]:
                    start_idx += 1

                if start_idx == len(segment['words']):
                    # Cant find valid start word
                    continue
                else:
                    start = segment['words'][start_idx]['start']

                while end_idx >= last_end_idx and 'start' not in segment['words'][end_idx]:
                    end_idx -= 1

                if end_idx < last_end_idx:
                    # Cant find valid end word
                    end = start
                else:
                    end = segment['words'][end_idx]['end']

                last_end_idx = end_idx

                pcs_result['sentences'].append({'text': sentence.lstrip(punctuations), 'start': start, 'end': end})

        return pcs_result

    @staticmethod
    def word_align(segment):
        """
        Align segment['text'] and segment['words']
        """
        # Remove the space in segment['text']
        # Note this shouldn't be performed for space separated language (check by need_sentence_align already)
        segment['text'] = segment['text'].replace(' ', '')

        if len(segment['text']) == len(segment['words']):
            return segment

        if not segment['words']:
            logger.error(f'Empty segment["words"], but segment["text"]: {segment["text"]}, skip word alignment')
            return segment

        new_words = []
        words_idx = 0
        if len(segment['text']) < len(segment['words']):
            # Align them by kicking out some word in segment['words']
            logger.warning(f'Segment length mismatch: '
                           f'segment["text"] {len(segment["text"])} < segment["words"] {len(segment["words"])}')
            for word in segment['text']:
                while words_idx < len(segment['words']) and word != segment['words'][words_idx]['word']:
                    logger.warning(f'Word mismatch: {word} != {segment["words"][words_idx]["word"]}')
                    words_idx += 1

                if words_idx >= len(segment['words']):
                    break
                new_words.append(segment['words'][words_idx])
                words_idx += 1
        else:
            # Align them by inserting empty element to segment['words']
            logger.warning(f'Segment length mismatch: '
                           f'segment["text"] {len(segment["text"])} > segment["words"] {len(segment["words"])}')
            for word in segment['text']:
                if word == segment['words'][words_idx]['word']:
                    new_words.append(segment['words'][words_idx])
                    words_idx += 1
                else:
                    new_words.append({'word': '<Added By Transcriber.word_align>'})

        assert len(segment['text']) == len(new_words)

        segment['words'] = new_words
        return segment
