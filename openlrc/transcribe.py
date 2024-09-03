#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

from pathlib import Path
from typing import NamedTuple, Union, List, Optional

import pysbd
from faster_whisper.transcribe import WhisperModel, Segment, BatchedInferencePipeline
from pysbd.languages import LANGUAGE_CODES
from tqdm import tqdm

from openlrc.defaults import default_asr_options, default_vad_options
from openlrc.logger import logger
from openlrc.utils import Timer, get_audio_duration, spacy_load, format_timestamp


class TranscriptionInfo(NamedTuple):
    """
    Stores information about a transcription.

    Attributes:
        language (str): The detected language of the audio.
        duration (float): The total duration of the audio in seconds.
        duration_after_vad (float): The duration of the audio after Voice Activity Detection (VAD).
    """
    language: str
    duration: float
    duration_after_vad: float

    @property
    def vad_ratio(self):
        """
        Calculate the ratio of audio removed by VAD.

        Returns:
            float: The proportion of audio removed by VAD.
        """
        return 1 - self.duration_after_vad / self.duration


class Transcriber:
    """
    A class for transcribing audio files using the Whisper model.

    Attributes:
        model_name (str): The name of the Whisper model to use.
        compute_type (str): The compute type for the model (e.g., 'float16').
        device (str): The device to run the model on (e.g., 'cuda').
        continuous_scripted (list): List of languages that are continuously scripted.
        asr_options (dict): Options for the ASR model.
        vad_options (dict): Options for Voice Activity Detection.
        whisper_model (BatchedInferencePipeline): The Whisper model pipeline.
    """

    def __init__(self, model_name: str = 'large-v3', compute_type: str = 'float16', device: str = 'cuda',
                 vad_filter: bool = True, asr_options: Optional[dict] = None, vad_options: Optional[dict] = None):
        self.model_name = model_name
        self.compute_type = compute_type
        self.device = device
        # self.no_need_align = ['en', 'ja', 'zh']  # Languages that is accurate enough without sentence alignment
        self.continuous_scripted = ['ja', 'zh', 'zh-cn', 'th', 'vi', 'lo', 'km', 'my', 'bo']
        self.asr_options = asr_options or default_asr_options
        self.vad_options = vad_options or default_vad_options

        model = WhisperModel(model_name, device, compute_type=compute_type, num_workers=1)
        self.whisper_model = BatchedInferencePipeline(model, use_vad_model=vad_filter, **self.vad_options)

    def transcribe(self, audio_path: Union[str, Path], language: Optional[str] = None):
        """
        Transcribe an audio file.

        Args:
            audio_path (Union[str, Path]): Path to the audio file.
            language (Optional[str]): Language of the audio. If None, it will be auto-detected.

        Returns:
            tuple: A tuple containing:
                - list: List of transcribed segments.
                - TranscriptionInfo: Information about the transcription.
        """
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

    def sentence_split(self, segments: List[Segment], lang: str):
        """
        Split transcribed segments into sentences.

        This function takes the raw transcribed segments and splits them into more
        natural sentence-like units. It handles different languages and uses
        language-specific segmentation rules.

        Args:
            segments (List[Segment]): List of transcribed segments from the ASR model.
            lang (str): Language code of the transcription.

        Returns:
            list: List of sentence-split segments.
        """
        if lang not in LANGUAGE_CODES.keys():
            logger.warning(f'Language {lang} not supported. Skipping sentence split.')
            return segments

        # Load language-specific NLP model
        nlp = spacy_load(lang)

        def seg_from_words(seg: Segment, seg_id: int, words: List, tokens: List):
            """
            Create a new segment from a subset of words.

            This helper function constructs a new Segment object from a given
            list of words, preserving the necessary metadata from the original segment.

            Args:
                seg (Segment): Original segment containing the words.
                seg_id (int): New ID for the created segment.
                words (List): List of Word objects to include in the new segment.
                tokens (List): List of tokens corresponding to the words.

            Returns:
                Segment: A new Segment object created from the given words.
            """
            text = ''.join([word.word for word in words])
            return Segment(seg_id, seg.seek, words[0].start, words[-1].end, text, tokens, seg.avg_logprob,
                           seg.compression_ratio, seg.no_speech_prob, words, seg.temperature)

        def mid_split(seg_entry: Segment):
            """
            Split a segment roughly in the middle.

            This function attempts to split a segment into two parts, preferably
            at a natural break point like punctuation or space. If no suitable
            break point is found, it falls back to splitting based on word gaps
            or exactly in the middle.

            Args:
                seg_entry (Segment): The segment to split.

            Returns:
                list: List of split segments.
            """
            text = seg_entry.text
            doc = nlp(text)

            def is_punct(char):
                return doc.vocab[char].is_punct

            splittable = int(len(text) / 3)

            # Attempt to find a natural split point
            former_words, former_len = [], 0
            for j, word in enumerate(seg_entry.words):
                former_words.append(word)
                former_len += len(word.word)

                # Special handling for languages without spaces between words
                if lang in self.continuous_scripted and former_len >= splittable:
                    if word.word.startswith(' '):
                        break
                    elif word.word.endswith(' '):
                        former_words.append(word)
                        former_len += len(word.word)
                        break

                # Split at punctuation if possible
                if former_len >= splittable and is_punct(word.word[-1]):
                    break

            latter_words = seg_entry.words[len(former_words):]

            # If no natural split point found, use alternative methods
            if not latter_words:
                # Find the largest gap between words
                gaps = [-1] + [seg_entry.words[k + 1].start - seg_entry.words[k].end
                               for k in range(len(seg_entry.words) - 1)]
                max_gap = max(gaps)
                split_idx = gaps.index(max_gap)

                if max_gap >= 2:  # Split at the largest gap if it's significant
                    former_words = seg_entry.words[:split_idx]
                    latter_words = seg_entry.words[split_idx:]
                else:  # Otherwise, split exactly in the middle
                    mid_point = len(seg_entry.words) // 2
                    former_words = seg_entry.words[:mid_point]
                    latter_words = seg_entry.words[mid_point:]

            # Safeguard against empty splits
            if not former_words or not latter_words:
                logger.warning(f'Empty split detected: {former_words} or {latter_words}, skipping split')
                return [seg_entry]

            # Create new segments from the split
            former = seg_from_words(seg_entry, seg_entry.id, former_words, seg_entry.tokens[:len(former_words)])
            latter = seg_from_words(seg_entry, seg_entry.id + 1, latter_words, seg_entry.tokens[len(former_words):])

            return [former, latter]

        # Initialize sentence segmenter for the given language
        segmenter = pysbd.Segmenter(language=lang, clean=False)

        id_cnt = 0
        sentences = []  # [{'text': , 'start': , 'end': , 'words': [{word: , start: , end: , score: }, ...]}, ...]
        for segment in segments:
            # Use pysbd to split the segment text into potential sentences
            splits = [s for s in segmenter.segment(segment.text) if s]  # Also filter out empty splits
            word_start = 0

            for split in splits:
                # Align words with the split text
                split_words = []
                split_words_len = 0
                for i in range(len(split)):
                    if word_start + i < len(segment.words):
                        split_words.append(segment.words[word_start + i])
                        split_words_len = len(''.join([word.word for word in split_words]).rstrip())
                    else:
                        logger.warning(f'Word alignment issue: {word_start + i} >= {len(segment.words)}. '
                                       f'Keeping: {"".join([word.word for word in split_words])}, '
                                       f'Discarding: {split[split_words_len:]}')
                        break
                    if split_words_len >= len(split.rstrip()):
                        break

                # Sanity checks for split quality
                if split_words_len >= len(split.rstrip()) + 3:
                    logger.warning(f'Split words length mismatch: {split_words_len} >= {len(split)} + 3')
                if split_words_len == 0:
                    logger.warning(f'Zero-length split detected for: {split}, skipping')
                    continue

                word_start += len(split_words)

                # Create a new segment for this split
                entry = seg_from_words(segment, id_cnt, split_words,
                                       segment.tokens[word_start: word_start + len(split_words)])

                def recursive_segment(entry: Segment):
                    """
                    Recursively segment an entry if it's too long.

                    This function checks if a segment is too long (based on character count
                    or duration) and splits it if necessary. It applies different thresholds
                    for different language types.

                    Args:
                        entry (Segment): The segment to potentially split.

                    Returns:
                        list: List of segments after recursive splitting.
                    """
                    # Check if the segment needs splitting
                    char_limit = 45 if lang in self.continuous_scripted else 90
                    if len(entry.text) < char_limit or len(entry.words) == 1:
                        if entry.end - entry.start > 10:  # Split if duration > 10s
                            segmented_entries = mid_split(entry)
                            if len(segmented_entries) == 1:  # Can't be further segmented
                                return [entry]

                            further_segmented = []
                            for segment in segmented_entries:
                                further_segmented.extend(recursive_segment(segment))
                            return further_segmented
                        else:
                            return [entry]
                    else:
                        # Split in the middle and recursively process the results
                        segmented_entries = mid_split(entry)
                        further_segmented = []
                        for segment in segmented_entries:
                            further_segmented.extend(recursive_segment(segment))
                        return further_segmented

                # Apply recursive segmentation to handle long sentences
                segmented_entries = recursive_segment(entry)

                sentences.extend(segmented_entries)
                id_cnt += len(segmented_entries)

        return sentences
