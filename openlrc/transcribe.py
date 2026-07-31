#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from openlrc.workflow import ExecutionContext

import pysbd
from pysbd.languages import LANGUAGE_CODES
from tqdm import tqdm

from openlrc.defaults import default_whisper_cpp_options
from openlrc.exceptions import TranscribeException
from openlrc.logger import logger
from openlrc.media_utils import get_audio_duration
from openlrc.utils import Timer, format_timestamp
from openlrc.whisper_backend import WhisperCLIBackend
from openlrc.whisper_resources import DEFAULT_MODEL_NAME, DEFAULT_VAD_MODEL_NAME
from openlrc.whisper_types import Segment, Word


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


def _parse_timestamp_str(ts: str) -> float:
    """解析 whisper.cpp 时间戳字符串为秒数。

    whisper.cpp JSON 的 timestamps.from/to 格式为 "HH:MM:SS.mmm" 或 "HH:MM:SS,mmm"
    (源码 cli.cpp L687: to_timestamp(t0, true) 输出 SRT 格式)

    Args:
        ts: 时间戳字符串，例如 "00:00:01.500" 或 "00:00:01,500"。

    Returns:
        浮点秒数。
    """
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(ts)


def _time_range(item: Mapping[str, object], *, label: str, allow_missing: bool = False) -> tuple[float, float] | None:
    """Read a whisper.cpp time range, preferring integer-millisecond offsets."""
    offsets = item.get("offsets")
    if isinstance(offsets, Mapping):
        start_value = offsets.get("from")
        end_value = offsets.get("to")
        if (
            isinstance(start_value, (int, float))
            and not isinstance(start_value, bool)
            and isinstance(end_value, (int, float))
            and not isinstance(end_value, bool)
        ):
            start = float(start_value) / 1000.0
            end = float(end_value) / 1000.0
            if math.isfinite(start) and math.isfinite(end) and end >= start:
                return start, end

    timestamps = item.get("timestamps")
    if isinstance(timestamps, Mapping):
        start_value = timestamps.get("from")
        end_value = timestamps.get("to")
        if isinstance(start_value, str) and isinstance(end_value, str):
            try:
                start = _parse_timestamp_str(start_value)
                end = _parse_timestamp_str(end_value)
            except (TypeError, ValueError):
                pass
            else:
                if math.isfinite(start) and math.isfinite(end) and end >= start:
                    return start, end

    if allow_missing:
        return None
    raise TranscribeException(f"whisper-cli JSON {label} has no valid offsets or timestamps.")


def map_cli_json_to_segments(cli_json: dict) -> list[Segment]:
    """将 whisper-cli -ojf 输出的 JSON 转换为 Segment 列表。

    JSON 结构（-ojf 模式，源码验证于 cli.cpp L725-L765）::

        {
          "result": {"language": "en"},
          "transcription": [
            {
              "timestamps": {"from": "00:00:00,000", "to": "00:00:03,000"},
              "offsets": {"from": 0, "to": 3000},
              "text": "hello world",
              "tokens": [
                {
                  "text": "hello",
                  "timestamps": {"from": "00:00:00,000", "to": "00:00:01,500"},
                  "offsets": {"from": 0, "to": 1500},
                  "id": 123, "p": 0.95, "t_dtw": -1.0
                }, ...
              ]
            }, ...
          ]
        }

    时间单位说明:
      offsets 值为毫秒。源码 cli.cpp L691: ``value_i("from", t0 * 10, ...)``
      其中 t0 是 centiseconds（10ms 精度），乘以 10 得到 milliseconds。
      因此转换公式: ``seconds = offsets["from"] / 1000.0``

    Args:
        cli_json: whisper-cli 的 JSON 输出 dict。

    Returns:
        Segment 列表，每个 Segment 包含带 Word 级时间戳的 words 列表。
    """
    transcription = cli_json.get("transcription", [])
    if not isinstance(transcription, list):
        raise TranscribeException("whisper-cli JSON field 'transcription' must be an array.")

    segments: list[Segment] = []
    for i, seg in enumerate(transcription):
        if not isinstance(seg, Mapping):
            raise TranscribeException(f"whisper-cli JSON segment {i} must be an object.")

        # --- Segment 级时间 ---
        segment_range = _time_range(seg, label=f"segment {i}")
        assert segment_range is not None
        seg_start, seg_end = segment_range
        seg_text_value = seg.get("text", "")
        if not isinstance(seg_text_value, str):
            raise TranscribeException(f"whisper-cli JSON segment {i} field 'text' must be a string.")
        seg_text = seg_text_value.strip()

        # --- Token/Word 级时间 ---
        tokens = seg.get("tokens", [])
        if not isinstance(tokens, list):
            raise TranscribeException(f"whisper-cli JSON segment {i} field 'tokens' must be an array.")
        words: list[Word] = []
        for token_index, tok in enumerate(tokens):
            if not isinstance(tok, Mapping):
                raise TranscribeException(f"whisper-cli JSON segment {i} token {token_index} must be an object.")
            tok_text = tok.get("text", "")
            if not isinstance(tok_text, str):
                raise TranscribeException(
                    f"whisper-cli JSON segment {i} token {token_index} field 'text' must be a string."
                )

            # 跳过特殊 token（如 [SOT], [EOT], [_BEG_] 等）
            if tok_text.startswith("[") and tok_text.endswith("]"):
                continue
            # 跳过纯空白 token
            if not tok_text.strip():
                continue

            # 优先用 offsets（精确整数毫秒），再回退 timestamps 和 segment 时间。
            token_range = _time_range(tok, label=f"segment {i} token {token_index}", allow_missing=True)
            if token_range is None:
                # 无时间戳的 token，使用 segment 级时间作为 fallback
                w_start = seg_start
                w_end = seg_end
            else:
                w_start, w_end = token_range

            probability = tok.get("p", 0.0)
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise TranscribeException(
                    f"whisper-cli JSON segment {i} token {token_index} field 'p' must be numeric."
                )
            words.append(Word(start=w_start, end=w_end, word=tok_text, probability=float(probability)))

        # 过滤空 words 的 segment
        # sentence_split() 中有 assert segment.words is not None
        if not words:
            logger.warning(f"Segment {i} has no valid words, skipping: '{seg_text}'")
            continue

        segments.append(
            Segment(
                id=i,
                seek=0,
                start=seg_start,
                end=seg_end,
                text=seg_text,
                tokens=[],  # OpenLRC 下游不使用 raw token IDs
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
                words=words,
                temperature=0.0,
            )
        )

    return segments


class Transcriber:
    """
    A class for transcribing audio files using whisper-cli.

    This uses a subprocess-based whisper.cpp CLI backend that supports Metal acceleration.

    Attributes:
        model_name (str): Path to the Whisper GGML model file.
        cli_backend (WhisperCLIBackend): The CLI backend instance.
        continuous_scripted (list): List of languages that are continuously scripted.
        asr_options (dict): Options for the ASR model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cli_path: str = "",
        vad_model: str = DEFAULT_VAD_MODEL_NAME,
        asr_options: dict | None = None,
        vad_filter: bool = True,
        execution_context: ExecutionContext | None = None,
    ):
        self.model_name = model_name
        self.continuous_scripted = ["ja", "zh", "zh-cn", "th", "vi", "lo", "km", "my", "bo"]
        self.asr_options = {**default_whisper_cpp_options, **(asr_options or {})}
        self.execution_context = execution_context

        self.cli_backend = WhisperCLIBackend(
            cli_path=cli_path, model_path=model_name, vad_model_path=vad_model if vad_filter else ""
        )

    def transcribe(self, audio_path: str | Path, language: str | None = None):
        """
        Transcribe an audio file using whisper-cli.

        Args:
            audio_path (Union[str, Path]): Path to the audio file.
            language (Optional[str]): Language of the audio. If None, it will be auto-detected.

        Returns:
            tuple: A tuple containing:
                - list: List of transcribed segments (after sentence splitting).
                - TranscriptionInfo: Information about the transcription.
        """
        total_duration = get_audio_duration(audio_path)

        # Direct API calls retain tqdm; Workflow clients receive typed progress events.
        pbar = None if self.execution_context is not None else tqdm(total=100, unit="%", desc="Transcribing")
        model_ready = False
        if self.execution_context is not None:
            self.execution_context.model_event(self.model_name, "starting", owned=True, role="transcription")
            from openlrc.workflow import WorkflowStage

            self.execution_context.stage_progress(WorkflowStage.TRANSCRIBE, 0, 100, item=audio_path, message="Whisper")

        def _progress_cb(pct: int) -> None:
            nonlocal model_ready
            if self.execution_context is not None and not model_ready:
                self.execution_context.model_event(self.model_name, "ready", owned=True, role="transcription")
                model_ready = True
            if pbar is not None:
                delta = pct - pbar.n
                if delta > 0:
                    pbar.update(delta)
            if self.execution_context is not None:
                from openlrc.workflow import WorkflowStage

                self.execution_context.stage_progress(
                    WorkflowStage.TRANSCRIBE, pct, 100, item=audio_path, message="Whisper"
                )

        # 调用 whisper-cli
        try:
            cli_json = self.cli_backend.transcribe(
                audio_path=str(audio_path),
                lang=language,
                progress_cb=_progress_cb,
                extra_args=self._build_extra_args(),
                cancellation_token=(self.execution_context.cancellation_token if self.execution_context else None),
                process_registry=(self.execution_context.processes if self.execution_context else None),
            )
            if self.execution_context is not None:
                _progress_cb(100)
        finally:
            if pbar is not None:
                pbar.close()
            if self.execution_context is not None:
                self.execution_context.model_event(self.model_name, "stopped", owned=True, role="transcription")

        # JSON -> Segment 列表
        segments = map_cli_json_to_segments(cli_json)

        # 语言检测结果
        # whisper-cli JSON: result.language 为短代码如 "en"
        # (cli.cpp L723: whisper_lang_str(whisper_full_lang_id(ctx)))
        result_info = cli_json.get("result", {})
        if not isinstance(result_info, Mapping):
            raise TranscribeException("whisper-cli JSON field 'result' must be an object.")
        detected_lang = language or result_info.get("language", "en")
        if not isinstance(detected_lang, str) or not detected_lang:
            raise TranscribeException("whisper-cli JSON detected language must be a non-empty string.")

        # 估算 VAD 后时长（累加所有 segment 的有效时长）
        duration_after_vad = sum(s.end - s.start for s in segments)
        if duration_after_vad == 0:
            duration_after_vad = total_duration

        info = TranscriptionInfo(language=detected_lang, duration=total_duration, duration_after_vad=duration_after_vad)

        if not segments:
            logger.warning(f"No speech found for {audio_path}")
            result = []
        else:
            with Timer("Sentence Segmentation"):
                result = self.sentence_split(segments, info.language)

        logger.info(
            f"VAD removed {format_timestamp(info.duration - info.duration_after_vad)}s "
            f"of silence ({info.vad_ratio * 100:.1f}%) "
        )
        if info.vad_ratio > 0.5:
            logger.warning(
                f"VAD ratio is too high, check your audio quality. "
                f"VAD ratio: {info.vad_ratio}, duration: {format_timestamp(info.duration, fmt='srt')}, "
                f"duration_after_vad: {format_timestamp(info.duration_after_vad, fmt='srt')}. "
            )

        return result, info

    def _build_extra_args(self) -> list[str]:
        """从 asr_options 构建额外的 whisper-cli 参数。

        Returns:
            额外 CLI 参数列表。
        """
        args: list[str] = []
        opts = self.asr_options

        if opts.get("beam_size", 5) > 1:
            args.extend(["-bs", str(opts["beam_size"])])
        if opts.get("best_of", 5) > 1:
            args.extend(["-bo", str(opts["best_of"])])
        if opts.get("initial_prompt"):
            args.extend(["--prompt", str(opts["initial_prompt"])])
        if opts.get("temperature", 0.0) != 0.0:
            args.extend(["-t", str(opts["temperature"])])
        if opts.get("suppress_nst", False):
            args.append("-sns")
        if opts.get("use_gpu", True) is False or opts.get("no_gpu", False):
            args.append("-ng")
        if opts.get("flash_attn", True) is False:
            args.append("-nfa")

        return args

    def sentence_split(self, segments: list[Segment], lang: str):
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
        if lang not in LANGUAGE_CODES:
            logger.warning(f"Language {lang} not supported. Skipping sentence split.")
            return segments

        def seg_from_words(seg: Segment, seg_id: int, words: list, tokens: list):
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
            text = "".join([word.word for word in words])
            return Segment(
                seg_id,
                seg.seek,
                words[0].start,
                words[-1].end,
                text,
                tokens,
                seg.avg_logprob,
                seg.compression_ratio,
                seg.no_speech_prob,
                words,
                seg.temperature,
            )

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
            if seg_entry.words is None:
                raise ValueError("Segment must have word-level timestamps for splitting")
            text = seg_entry.text

            def is_punct(char):
                return bool(char) and unicodedata.category(char[-1]).startswith("P")

            splittable = int(len(text) / 3)

            # Attempt to find a natural split point
            former_words, former_len = [], 0
            for j, word in enumerate(seg_entry.words):
                former_words.append(word)
                former_len += len(word.word)

                # Special handling for languages without spaces between words:
                # break at the first space boundary once we've accumulated enough text.
                if lang in self.continuous_scripted and former_len >= splittable and " " in word.word:
                    break

                # Split at punctuation if possible
                if former_len >= splittable and is_punct(word.word[-1]):
                    break

            latter_words = seg_entry.words[len(former_words) :]

            # If no natural split point found, use alternative methods
            if not latter_words:
                # Find the largest gap between words
                gaps = [-1] + [
                    seg_entry.words[k + 1].start - seg_entry.words[k].end for k in range(len(seg_entry.words) - 1)
                ]
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
                logger.warning(f"Empty split detected: {former_words} or {latter_words}, skipping split")
                return [seg_entry]

            # Create new segments from the split
            former = seg_from_words(seg_entry, seg_entry.id, former_words, seg_entry.tokens[: len(former_words)])
            latter = seg_from_words(seg_entry, seg_entry.id + 1, latter_words, seg_entry.tokens[len(former_words) :])

            return [former, latter]

        # Initialize sentence segmenter for the given language
        segmenter = pysbd.Segmenter(language=lang, clean=False)

        id_cnt = 0
        sentences = []  # [{'text': , 'start': , 'end': , 'words': [{word: , start: , end: , score: }, ...]}, ...]
        for segment in segments:
            if segment.words is None:
                raise ValueError("Segment must have word-level timestamps")
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
                        split_words_len = len("".join([word.word for word in split_words]).rstrip())
                    else:
                        logger.warning(
                            f"Word alignment issue: {word_start + i} >= {len(segment.words)}. "
                            f"Keeping: {''.join([word.word for word in split_words])}, "
                            f"Discarding: {split[split_words_len:]}"
                        )
                        break
                    if split_words_len >= len(split.rstrip()):
                        break

                # Sanity checks for split quality
                if split_words_len >= len(split.rstrip()) + 3:
                    logger.warning(f"Split words length mismatch: {split_words_len} >= {len(split)} + 3")
                if split_words_len == 0:
                    logger.warning(f"Zero-length split detected for: {split}, skipping")
                    continue

                word_start += len(split_words)

                # Create a new segment for this split
                entry = seg_from_words(
                    segment, id_cnt, split_words, segment.tokens[word_start : word_start + len(split_words)]
                )

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
                    if entry.words is None:
                        raise ValueError("Segment must have word-level timestamps")
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
