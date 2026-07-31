#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlrc.workflow import ExecutionContext

from ffmpeg_normalize import FFmpegNormalize

from openlrc.defaults import LOUDNORM_SUFFIX, PREPROCESSED_DIR
from openlrc.logger import logger
from openlrc.utils import get_preprocessed_path


def loudness_norm_single(audio_path: Path, ln_path: Path):
    """
    Normalize the loudness of a single audio file using FFmpegNormalize.

    Args:
        audio_path (Path): The path to the input audio file.
        ln_path (Path): The path to save the normalized audio file.
    """
    normalizer = FFmpegNormalize(
        output_format="wav",
        sample_rate=48000,
        progress=logger.level <= logging.DEBUG,
        keep_lra_above_loudness_range_target=True,
    )

    if not ln_path.exists():
        normalizer.add_media_file(str(audio_path), str(ln_path))
        normalizer.run_normalization()


class Preprocessor:
    """
    Preprocess audio to make it clear and normalized.
    """

    def __init__(
        self,
        audio_paths: str | Path | Sequence[str | Path],
        *,
        output_folder: str = PREPROCESSED_DIR,
        execution_context: ExecutionContext | None = None,
    ):
        paths_list = [audio_paths] if isinstance(audio_paths, (str, Path)) else list(audio_paths)
        self.audio_paths: list[Path] = [Path(p) for p in paths_list]
        self.output_paths = [p.parent / output_folder for p in self.audio_paths]
        self.execution_context = execution_context

        for path in self.output_paths:
            if not path.exists():
                path.mkdir()

    def loudness_normalization(self, audio_paths: list[Path]):
        """
        Normalize loudness of audio.
        """
        logger.info("Loudness normalizing...")

        args = []
        ln_audio_paths = []
        for audio_path, output_path in zip(audio_paths, self.output_paths):
            ln_path = output_path / f"{audio_path.stem}{LOUDNORM_SUFFIX}.wav"
            args.append((audio_path, ln_path))
            ln_audio_paths.append(ln_path)

        if self.execution_context is not None:
            with ThreadPoolExecutor(thread_name_prefix="Loudness") as executor:
                futures = [
                    executor.submit(self._managed_loudness_normalization, audio_path, ln_path)
                    for audio_path, ln_path in args
                ]
                for future in futures:
                    future.result()
            return ln_audio_paths

        # ffmpeg-normalize performs the CPU-heavy work in child ffmpeg processes;
        # threads avoid nested multiprocessing semaphore failures on macOS sandboxes.
        with ThreadPoolExecutor(thread_name_prefix="Loudness") as executor:
            results = [executor.submit(loudness_norm_single, *arg) for arg in args]

            exceptions = [res.exception() for res in results]
            if any(exceptions):
                # Get the first not None exception
                exception = next(filter(None, exceptions))

                logger.error(f"Loudness normalization failed, exception: {exception}")
                raise exception

        return ln_audio_paths

    def _managed_loudness_normalization(self, audio_path: Path, ln_path: Path) -> None:
        assert self.execution_context is not None
        if ln_path.exists():
            return
        partial_path = ln_path.with_name(f".{ln_path.name}.{uuid.uuid4().hex}.partial.wav")
        command = [
            sys.executable,
            "-m",
            "ffmpeg_normalize",
            str(audio_path),
            "-o",
            str(partial_path),
            "--output-format",
            "wav",
            "--sample-rate",
            "48000",
            "--keep-lra-above-loudness-range-target",
            "--quiet",
            "--force",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True
        )
        self.execution_context.processes.register(process, process_group=True)
        try:
            while process.poll() is None:
                self.execution_context.cancellation_token.wait_or_raise(0.1)
            self.execution_context.check_cancelled()
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg-normalize exited with code {process.returncode}: {stderr}")
            partial_path.replace(ln_path)
        finally:
            if process.poll() is None:
                self.execution_context.processes.terminate(process)
            else:
                self.execution_context.processes.unregister(process)
            partial_path.unlink(missing_ok=True)

    def run(self):
        """
        Returns:
            list of Path: A list of Path objects representing the final processed audio paths.
        """
        # Check if the preprocessed audio already exists.
        need_process = []
        final_processed_audios = []
        for audio_path, output_path in zip(self.audio_paths, self.output_paths):
            if self.execution_context is not None:
                self.execution_context.check_cancelled()
            preprocessed_path = get_preprocessed_path(audio_path)
            final_processed_audios.append(preprocessed_path)
            if preprocessed_path.exists():
                logger.info(f"Preprocessed audio already exists in {preprocessed_path}")
                if self.execution_context is not None:
                    from openlrc.workflow import StageOutcome, WorkflowStage

                    self.execution_context.stage_completed(
                        WorkflowStage.PREPROCESS,
                        outcome=StageOutcome.SKIPPED,
                        item=audio_path,
                        message="Using cached preprocessed audio.",
                    )
                continue
            else:
                need_process.append(audio_path)

        ln_paths: list[Path] = self.loudness_normalization(need_process)

        for path, audio_path in zip(ln_paths, need_process):
            if self.execution_context is not None:
                self.execution_context.check_cancelled()
            final_path = get_preprocessed_path(audio_path)
            path.rename(final_path)
            logger.info(f"Preprocessed audio saved to {final_path}")

        return final_processed_audios
