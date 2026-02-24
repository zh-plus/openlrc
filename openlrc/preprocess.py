#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Union

import torch
from df.enhance import enhance, init_df, load_audio, save_audio
from ffmpeg_normalize import FFmpegNormalize
from tqdm import tqdm

from openlrc.defaults import default_preprocess_options
from openlrc.logger import logger
from openlrc.utils import release_memory, get_preprocessed_path


def loudness_norm_single(audio_path: Path, ln_path: Path):
    """
    Normalize the loudness of a single audio file using FFmpegNormalize.

    Args:
        audio_path (Path): The path to the input audio file.
        ln_path (Path): The path to save the normalized audio file.
    """
    normalizer = FFmpegNormalize(output_format='wav', sample_rate=48000, progress=logger.level <= logging.DEBUG,
                                 keep_lra_above_loudness_range_target=True)

    if not ln_path.exists():
        normalizer.add_media_file(str(audio_path), str(ln_path))
        normalizer.run_normalization()


class Preprocessor:
    """
    Preprocess audio to make it clear and normalized.
    """

    def __init__(self, audio_paths: Union[str, Path, List[str], List[Path]], output_folder='preprocessed',
                 options: dict = default_preprocess_options):
        if not isinstance(audio_paths, list):
            audio_paths = [audio_paths]
        self.audio_paths = [Path(p) for p in audio_paths]
        self.output_paths = [p.parent / output_folder for p in self.audio_paths]
        self.options = options

        for path in self.output_paths:
            if not path.exists():
                path.mkdir()

    def noise_suppression(self, audio_paths: Union[str, Path, List[str], List[Path]], atten_lim_db=15):
        """
        Supress noise in audio.
        """
        if not audio_paths:
            return []

        if 'atten_lim_db' in self.options.keys():
            atten_lim_db = self.options['atten_lim_db']

        model, df_state, _ = init_df()
        chunk_size = 180  # 3 min

        ns_audio_paths = []
        for audio_path, output_path in zip(audio_paths, self.output_paths):
            audio_name = audio_path.stem
            ns_path = output_path / f'{audio_name}_ns.wav'

            if not ns_path.exists():
                audio, info = load_audio(audio_path, sr=df_state.sr())

                # Split audio into 3 min chunks
                audio_chunks = [audio[:, i:i + chunk_size * info.sample_rate]
                                for i in range(0, audio.shape[1], chunk_size * info.sample_rate)]

                enhanced_chunks = []
                for ac in tqdm(audio_chunks, desc=f'Noise suppressing for {audio_name}'):
                    enhanced_chunks.append(enhance(model, df_state, ac, atten_lim_db=atten_lim_db))

                enhanced = torch.cat(enhanced_chunks, dim=1)

                assert enhanced.shape == audio.shape, f'Enhanced audio shape does not match original audio shape: {enhanced.shape} != {audio.shape}'

                save_audio(ns_path, enhanced, sr=df_state.sr())

            ns_audio_paths.append(ns_path)

        release_memory(model)

        return ns_audio_paths

    def loudness_normalization(self, audio_paths: Union[str, Path, List[str], List[Path]]):
        """
        Normalize loudness of audio.
        """
        logger.info('Loudness normalizing...')

        args = []
        ln_audio_paths = []
        for audio_path, output_path in zip(audio_paths, self.output_paths):
            ln_path = output_path / f'{audio_path.stem}_ln.wav'
            args.append((audio_path, ln_path))
            ln_audio_paths.append(ln_path)

        # Multi-processing
        with ProcessPoolExecutor() as executor:
            results = [executor.submit(loudness_norm_single, *arg) for arg in args]

            exceptions = [res.exception() for res in results]
            if any(exceptions):
                # Get the first not None exception
                exception = next(filter(None, exceptions))

                logger.error(f'Loudness normalization failed, exception: {exception}')
                raise exception

        return ln_audio_paths

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
            preprocessed_path = get_preprocessed_path(audio_path)
            final_processed_audios.append(preprocessed_path)
            if preprocessed_path.exists():
                logger.info(f'Preprocessed audio already exists in {preprocessed_path}')
                continue
            else:
                need_process.append(audio_path)

        ns_paths = need_process
        if noise_suppress:
            ns_paths = self.noise_suppression(need_process)
        ln_paths: list[Path] = self.loudness_normalization(ns_paths)

        for path, audio_path in zip(ln_paths, need_process):
            final_path = get_preprocessed_path(audio_path)
            path.rename(final_path)
            logger.info(f'Preprocessed audio saved to {final_path}')

        return final_processed_audios
