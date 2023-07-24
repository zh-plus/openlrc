#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.
import logging
from pathlib import Path
from typing import List, Union

import torch
from df.enhance import enhance, init_df, load_audio, save_audio
from ffmpeg_normalize import FFmpegNormalize
from tqdm import tqdm

from openlrc.logger import logger
from openlrc.utils import release_memory


class Preprocessor:
    """
    Preprocess audio to make it clear and normalized.
    """

    def __init__(self, audio_paths: Union[str, Path, List[str], List[Path]], output_folder='preprocessed'):
        if not isinstance(audio_paths, list):
            audio_paths = [audio_paths]
        self.audio_paths = [Path(p) for p in audio_paths]
        self.output_paths = [p.parent / output_folder for p in self.audio_paths]

        for path in self.output_paths:
            if not path.exists():
                path.mkdir()

    def noise_suppression(self, audio_paths: Union[str, Path, List[str], List[Path]], atten_lim_db=15):
        """
        Supress noise in audio.
        """
        if not audio_paths:
            return []

        model, df_state, _ = init_df()

        ns_audio_paths = []
        for audio_path, output_path in zip(audio_paths, self.output_paths):
            audio_name = audio_path.stem
            ns_path = output_path / f'{audio_name}_ns.wav'

            if not ns_path.exists():
                audio, info = load_audio(audio_path, sr=df_state.sr())

                # Split audio into 10min chunks
                audio_chunks = [audio[:, i:i + 600 * info.sample_rate]
                                for i in range(0, audio.shape[1], 600 * info.sample_rate)]

                enhanced_chunks = []
                for ac in tqdm(audio_chunks, desc='Noise suppressing...'):
                    enhanced_chunks.append(enhance(model, df_state, ac, atten_lim_db=atten_lim_db))

                enhanced = torch.cat(enhanced_chunks, dim=1)

                assert enhanced.shape == audio.shape, 'Enhanced audio shape does not match original audio shape.'

                save_audio(ns_path, enhanced, sr=df_state.sr())

            ns_audio_paths.append(ns_path)
            logger.debug(f'Noise suppressed audio saved to {ns_path}')

        release_memory(model)

        return ns_audio_paths

    def loudness_normalization(self, audio_paths: Union[str, Path, List[str], List[Path]]):
        """
        Normalize loudness of audio.
        """
        logger.info('Loudness normalizing...')

        normalizer = FFmpegNormalize(output_format='wav', sample_rate=48000, progress=logger.level <= logging.DEBUG,
                                     keep_lra_above_loudness_range_target=True)
        ln_audio_paths = []
        for audio_path, output_path in zip(audio_paths, self.output_paths):
            audio_name = audio_path.stem

            ln_path = output_path / f'{audio_name}_ln.wav'

            if not ln_path.exists():
                normalizer.add_media_file(audio_path, ln_path)

            ln_audio_paths.append(ln_path)

        normalizer.run_normalization()

        return ln_audio_paths

    def run(self, noise_suppress=False):
        """
        :param noise_suppress: a boolean flag indicating whether to perform noise suppression. Default is False.
        :return: a list of Path objects representing the final processed audio paths.
        """
        # Check if the preprocessed audio already exists.
        need_process = []
        final_processed = []
        for audio_path, output_path in zip(self.audio_paths, self.output_paths):
            audio_name = audio_path.stem
            preprocessed_path = output_path / f'{audio_name}_preprocessed.wav'
            final_processed.append(preprocessed_path)
            if preprocessed_path.exists():
                logger.info(f'Preprocessed audio already exists in {preprocessed_path}')
                continue
            else:
                need_process.append(audio_path)

        ns_paths = need_process
        if noise_suppress:
            ns_paths = self.noise_suppression(need_process)
        ln_paths: list[Path] = self.loudness_normalization(ns_paths)

        for ln_path, audio_path in zip(ln_paths, need_process):
            audio_name = audio_path.stem

            ln_path = ln_path.rename(ln_path.parent / f'{audio_name}_preprocessed.wav')
            logger.info(f'Preprocessed audio saved to {ln_path}')

        return final_processed
