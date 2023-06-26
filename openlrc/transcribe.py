#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import json
import re
from typing import NamedTuple

import whisperx
from punctuators.models import PunctCapSegModelONNX

from openlrc.logger import logger
from openlrc.utils import Timer, release_memory, get_audio_duration


class TranscriptionInfo(NamedTuple):
    language: str
    duration: float


class Transcriber:
    def __init__(self, model_name='large-v2', compute_type='float16', device='cuda'):
        self.model_name = model_name
        self.compute_type = compute_type
        self.device = device

    def transcribe(self, audio_path, batch_size=8):
        whisper_model = whisperx.load_model(self.model_name, compute_type=self.compute_type, device=self.device)
        audio = whisperx.load_audio(audio_path)

        with Timer('Base Whisper Transcription'):
            result = whisper_model.transcribe(audio, batch_size=batch_size)

        release_memory(whisper_model)

        with Timer('Phoneme Alignment'):
            align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, self.device,
                                            return_char_alignments=False)

        release_memory(align_model)

        with open('test_aligned_result.json', 'w', encoding='utf-8') as f:
            json.dump(aligned_result, f, ensure_ascii=False, indent=4)

        with Timer('Sentence Alignment'):
            pcs_result = self.sentence_align(aligned_result)

        info = TranscriptionInfo(language=result['language'], duration=get_audio_duration(audio_path))

        return pcs_result, info

    @staticmethod
    def sentence_align(transcribe_result):
        """
        Align the word-level whisper transcribe result to sentence-level.

        :return A dict with key 'sentences' and value a list of dict with key 'text', 'start_word', 'end_word'.
        """
        pcs_model: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
            "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
        )
        punctuations = '.,?？，。、・।؟;።፣፧،'

        sentences_list = pcs_model.infer([segment['text'] for segment in transcribe_result['segments']], apply_sbd=True)

        pcs_result = {'sentences': []}
        for segment, sentences in zip(transcribe_result['segments'], sentences_list):
            last_end_idx = 0
            for sentence in sentences:
                sentence = sentence.lower()
                stc_split = re.split(f'[{punctuations}]', sentence)

                # Remove empty string
                stc_split = [split for split in stc_split if split]

                if not stc_split:
                    continue

                # check if first and last is substring of sentence
                assert stc_split[0] in segment['text'], f'First split: {stc_split[0]} not in {segment["text"]}'
                assert stc_split[-1] in segment['text'], f'Last split: {stc_split[-1]} not in {segment["text"]}'

                # Locate the start and end split in transcribed sentences
                start_idx = segment['text'].find(stc_split[0], last_end_idx)
                if len(stc_split) == 1:
                    end_idx = start_idx + len(stc_split[0]) - 1
                else:
                    start_find = last_end_idx + len(''.join(stc_split[:-1]))
                    end_idx = segment['text'].find(stc_split[-1], start_find) + len(stc_split[-1]) - 1

                start_idx = max(start_idx, 0)  # ensure start_idx is not out of range
                end_idx = min(end_idx, len(segment['words']) - 1)  # ensure end_idx is not out of range
                last_end_idx = end_idx

                if not segment['words']:
                    start_word = {
                        'start': segment['start'],
                        'end': segment['start'],
                        'word': segment['text'][0]
                    }
                    end_word = {
                        'start': segment['end'],
                        'end': segment['end'],
                        'word': segment['text'][-1]
                    }
                    pcs_result['sentences'].append({'text': sentence, 'start_word': start_word, 'end_word': end_word})
                    continue

                # start word and end word should not be punctuation
                while start_idx < len(segment['words']) and 'start' not in segment['words'][start_idx]:
                    start_idx += 1

                if start_idx == len(segment['words']):
                    # Cant find valid start word
                    continue
                else:
                    start = segment['words'][start_idx]

                while end_idx >= last_end_idx and 'start' not in segment['words'][end_idx]:
                    end_idx -= 1

                if end_idx < last_end_idx:
                    # Cant find valid end word
                    end = start
                else:
                    end = segment['words'][end_idx]

                sentence = sentence.lstrip(punctuations)

                # TODO: Should remove this case in next release.
                if start['word'] not in sentence and end['word'] not in sentence:
                    logger.warning(
                        f'Cannot find start word {start["word"]} or end word {end["word"]} in sentence {sentence}'
                    )

                pcs_result['sentences'].append({'text': sentence, 'start_word': start, 'end_word': end})

        return pcs_result
