#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

"""Media-related utility functions that depend on heavy external libraries.

Functions in this module import packages such as ``ffmpeg``, ``filetype``,
``audioread``, and ``spacy`` inside their bodies.  Keeping them
separate from :mod:`openlrc.utils` ensures that the lightweight translation
path never triggers those imports — critical for Nuitka ``--nofollow-import-to``
builds where the heavy packages are intentionally excluded.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spacy.language import Language as SpacyLanguage

    from openlrc.workflow import ExecutionContext

from openlrc.logger import logger
from openlrc.utils import detect_lang


def extract_audio(path: Path, *, execution_context: ExecutionContext | None = None) -> Path:
    """
    Extract audio from video.
    :return: Audio path
    """
    import ffmpeg

    file_type = get_file_type(path)
    if file_type == "audio":
        return path

    probe = _probe_media(path, execution_context=execution_context)
    audio_streams = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None)
    if audio_streams is None:
        raise RuntimeError(f"No audio stream found in {path}")
    sample_rate = audio_streams["sample_rate"]
    logger.info(f"File {path}: Audio sample rate: {sample_rate}")

    audio_path = path.with_suffix(".wav")
    if audio_path.exists():
        raise FileExistsError(
            f"Cannot extract audio from {path}: {audio_path} already exists and is not owned by this workflow. "
            "Move or rename the sidecar WAV before retrying."
        )
    if execution_context is None:
        audio, err = (
            ffmpeg.input(path)
            .output("pipe:", format="wav", acodec="pcm_s16le", ar=sample_rate, loglevel="quiet")
            .run(capture_stdout=True)
        )

        if err:
            raise RuntimeError(f"ffmpeg error: {err}")

        with open(audio_path, "wb") as f:
            f.write(audio)
        return audio_path

    partial_path = audio_path.with_name(f".{audio_path.name}.{uuid.uuid4().hex}.partial")
    command = (
        ffmpeg.input(path)
        .output(str(partial_path), format="wav", acodec="pcm_s16le", ar=sample_rate, loglevel="quiet")
        .overwrite_output()
        .compile()
    )
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    execution_context.processes.register(process)
    try:
        while process.poll() is None:
            execution_context.cancellation_token.wait_or_raise(0.1)
        execution_context.check_cancelled()
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}: {stderr.decode(errors='replace')}")
        partial_path.replace(audio_path)
        execution_context.register_owned_path(audio_path)
    finally:
        if process.poll() is None:
            execution_context.processes.terminate(process)
        else:
            execution_context.processes.unregister(process)
        partial_path.unlink(missing_ok=True)

    return audio_path


def _probe_media(path: Path, *, execution_context: ExecutionContext | None = None) -> dict[str, Any]:
    """Probe media directly or through the workflow-owned process registry."""
    import ffmpeg

    if execution_context is None:
        return ffmpeg.probe(path)

    execution_context.check_cancelled()
    process = subprocess.Popen(
        ["ffprobe", "-show_format", "-show_streams", "-of", "json", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    execution_context.processes.register(process)
    try:
        stdout, stderr = process.communicate()
        execution_context.check_cancelled()
        if process.returncode != 0:
            raise ffmpeg.Error("ffprobe", stdout, stderr)
        return json.loads(stdout.decode("utf-8"))
    finally:
        if process.poll() is None:
            execution_context.processes.terminate(process)
        else:
            execution_context.processes.unregister(process)


def get_file_type(path: Path) -> str:
    import filetype

    if path.suffix == ".ts":
        return "video"

    try:
        guess = filetype.guess(path)
        if guess is None:
            raise RuntimeError(f"File {path} is not a valid file.")
        file_type = guess.mime.split("/")[0]
    except (TypeError, AttributeError) as e:
        raise RuntimeError(f"File {path} is not a valid file.") from e

    if file_type not in ["audio", "video"]:
        raise RuntimeError(f"File {path} is not a valid file. Should be audio or video file.")

    return file_type


def get_audio_duration(path: str | Path) -> float:
    import audioread

    with audioread.audio_open(str(path)) as audio:
        return audio.duration


def get_spacy_lib(lang):
    special_case = {"core_web": ["zh", "en"], "ent_wiki": ["xx"]}

    mid_str = "core_news"
    for k, v in special_case.items():
        if lang in v:
            mid_str = k

    return f"{lang}_{mid_str}_sm"


def spacy_load(lang) -> SpacyLanguage:
    import spacy
    import spacy.cli

    lib_name = get_spacy_lib(lang)
    try:
        nlp = spacy.load(lib_name)
    except (ImportError, OSError):
        logger.warning(f"Spacy model {lib_name} missed, downloading")
        spacy.cli.download(lib_name)  # pyright: ignore[reportPrivateImportUsage]
        nlp = spacy.load(lib_name)

    return nlp


def get_similarity(text1, text2):
    lang1 = detect_lang(text1)
    lang2 = detect_lang(text2)

    if lang1 != lang2:
        raise ValueError(f'language of "{text1}" ({lang1}) is not the same as "{text2}" ({lang2})')

    nlp = spacy_load(lang1)

    doc1 = nlp(text1)
    doc2 = nlp(text2)

    return doc1.similarity(doc2)


def merge_subtitle(video_path, subtitle_path, output_path):
    # check ffmpeg
    try:
        subprocess.check_output(["ffmpeg", "-version"])
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed. Please install ffmpeg first.") from None

    subtitle_abs = str(Path(subtitle_path).absolute())
    style = "FontSize=24,Bold=1"
    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={subtitle_abs}:force_style='{style}'",
            str(output_path),
        ]
    )

    logger.info(f"Subtitled video saved to {output_path}")
