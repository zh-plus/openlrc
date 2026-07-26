#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlrc.workflow import ExecutionContext

from ffmpeg_normalize import FFmpegNormalize
from tqdm import tqdm

from openlrc.defaults import LOUDNORM_SUFFIX, NOISE_SUPPRESSED_SUFFIX, PREPROCESSED_DIR, default_preprocess_options
from openlrc.logger import logger
from openlrc.media_utils import release_memory
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
        audio_paths: str | Path | list[str] | list[Path],
        output_folder: str = PREPROCESSED_DIR,
        options: dict | None = None,
        execution_context: ExecutionContext | None = None,
    ):
        if options is None:
            options = dict(default_preprocess_options)
        paths_list = audio_paths if isinstance(audio_paths, list) else [audio_paths]
        self.audio_paths: list[Path] = [Path(p) for p in paths_list]
        self.output_paths = [p.parent / output_folder for p in self.audio_paths]
        self.options = options
        self.execution_context = execution_context

        for path in self.output_paths:
            if not path.exists():
                path.mkdir()

    def noise_suppression(self, audio_paths: list[Path], atten_lim_db: int = 15):
        """
        Suppress noise in audio.
        """
        if not audio_paths:
            return []

        try:
            import torch  # pyright: ignore[reportMissingImports]
            from df.enhance import enhance, init_df, load_audio, save_audio  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "Noise suppression requires torch and deepfilternet. Install them with: pip install 'openlrc-mac[full]'"
            )

        if "atten_lim_db" in self.options:
            atten_lim_db = self.options["atten_lim_db"]

        model, df_state, _ = init_df()
        chunk_size = 180  # 3 min

        try:
            ns_audio_paths = []
            for audio_path, output_path in zip(audio_paths, self.output_paths):
                audio_name = audio_path.stem
                ns_path = output_path / f"{audio_name}{NOISE_SUPPRESSED_SUFFIX}.wav"

                if not ns_path.exists():
                    audio, info = load_audio(str(audio_path), sr=df_state.sr())

                    # Split audio into 3 min chunks
                    audio_chunks = [
                        audio[:, i : i + chunk_size * info.sample_rate]
                        for i in range(0, audio.shape[1], chunk_size * info.sample_rate)
                    ]

                    enhanced_chunks = []
                    chunks = (
                        audio_chunks
                        if self.execution_context is not None
                        else tqdm(audio_chunks, desc=f"Noise suppressing for {audio_name}")
                    )
                    for index, ac in enumerate(chunks, start=1):
                        if self.execution_context is not None:
                            self.execution_context.check_cancelled()
                        enhanced_chunks.append(enhance(model, df_state, ac, atten_lim_db=atten_lim_db))
                        if self.execution_context is not None:
                            from openlrc.workflow import WorkflowStage

                            self.execution_context.stage_progress(
                                WorkflowStage.PREPROCESS,
                                index,
                                len(audio_chunks),
                                item=audio_path,
                                message="Noise suppression",
                            )

                    enhanced = torch.cat(enhanced_chunks, dim=1)

                    if enhanced.shape != audio.shape:
                        raise ValueError(
                            f"Enhanced audio shape does not match original audio shape: {enhanced.shape} != {audio.shape}"
                        )

                    save_audio(str(ns_path), enhanced, sr=df_state.sr())

                ns_audio_paths.append(ns_path)

            return ns_audio_paths
        finally:
            release_memory(model)

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

    def run(self, noise_suppress=False):
        """
        Args:
            noise_suppress (bool, optional): A boolean flag indicating whether to perform noise suppression.
                Defaults to False.

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

        ns_paths = need_process
        if noise_suppress:
            ns_paths = self.noise_suppression(need_process)
        ln_paths: list[Path] = self.loudness_normalization(ns_paths)

        for path, audio_path in zip(ln_paths, need_process):
            if self.execution_context is not None:
                self.execution_context.check_cancelled()
            final_path = get_preprocessed_path(audio_path)
            path.rename(final_path)
            logger.info(f"Preprocessed audio saved to {final_path}")

        return final_processed_audios
