"""Read-only path planning that mirrors the core pipeline naming rules."""

from __future__ import annotations

from pathlib import Path

from openlrc.defaults import (
    BILINGUAL_SUFFIX,
    COMPARE_SUFFIX,
    EDIT_REPORT_SUFFIX,
    EDIT_SESSION_SUFFIX,
    NONTRANS_SUFFIX,
    OPTIMIZED_SUFFIX,
    PREPROCESSED_DIR,
    PREPROCESSED_SUFFIX,
    RELAXED_OPTIMIZED_SUFFIX,
    TRANSCRIBED_SUFFIX,
    TRANSLATED_SUFFIX,
)
from openlrc.media_utils import get_file_type
from openlrc.workflow.types import RunRequest, TranscribeRequest, TranslateRequest, WorkflowRequest


def plan_request_paths(request: WorkflowRequest) -> dict[str, dict[str, tuple[Path, ...]]]:
    """Return temporary, recovery, output, and sidecar paths for each input."""
    plans: dict[str, dict[str, tuple[Path, ...]]] = {}
    if isinstance(request, TranslateRequest):
        for raw_path in request.transcribed_paths:
            path = Path(raw_path).expanduser().resolve(strict=False)
            base_name = path.stem.replace(f"{PREPROCESSED_SUFFIX}{TRANSCRIBED_SUFFIX}", "")
            output_dir = path.parent.parent
            subtitle_suffix = _translation_subtitle_suffix(path, base_name)
            outputs = [output_dir / f"{base_name}{subtitle_suffix}"]
            if request.bilingual_sub:
                outputs.extend(
                    (
                        output_dir / f"{base_name}{BILINGUAL_SUFFIX}{subtitle_suffix}",
                        output_dir / f"{base_name}{NONTRANS_SUFFIX}{subtitle_suffix}",
                    )
                )
            plans[str(path)] = {
                "sidecars": (),
                "temporary": _translation_temporary_paths(path.parent, path.stem, base_name),
                "recovery": _recovery_paths(path.parent, output_dir, base_name),
                "outputs": tuple(outputs),
            }
        return plans

    assert isinstance(request, (RunRequest, TranscribeRequest))
    translated = isinstance(request, RunRequest) and request.translation is not None
    bilingual = isinstance(request, RunRequest) and request.bilingual_sub and translated
    for raw_path in request.paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        base_name = path.stem
        temporary_dir = path.parent / PREPROCESSED_DIR
        preprocessed = temporary_dir / f"{base_name}{PREPROCESSED_SUFFIX}.wav"
        transcription = temporary_dir / f"{base_name}{PREPROCESSED_SUFFIX}{TRANSCRIBED_SUFFIX}.json"
        subtitle_suffix = ".srt" if get_file_type(path) == "video" else ".lrc"
        temporary = [preprocessed, transcription]
        if translated:
            temporary.extend(_translation_temporary_paths(temporary_dir, transcription.stem, base_name))
        outputs = [path.parent / f"{base_name}{subtitle_suffix}"]
        if bilingual:
            outputs.extend(
                (
                    path.parent / f"{base_name}{BILINGUAL_SUFFIX}{subtitle_suffix}",
                    path.parent / f"{base_name}{NONTRANS_SUFFIX}{subtitle_suffix}",
                )
            )
        plans[str(path)] = {
            "sidecars": (
                (path.with_suffix(".wav"),) if subtitle_suffix == ".srt" and not request.skip_preprocess else ()
            ),
            "temporary": tuple(dict.fromkeys(temporary)),
            "recovery": _recovery_paths(temporary_dir, path.parent, base_name) if translated else (),
            "outputs": tuple(outputs),
        }
    return plans


def _translation_temporary_paths(temporary_dir: Path, stem: str, base_name: str) -> tuple[Path, ...]:
    return (
        temporary_dir / f"{stem}{OPTIMIZED_SUFFIX}.json",
        temporary_dir / f"{stem}{RELAXED_OPTIMIZED_SUFFIX}.json",
        temporary_dir / f"{stem}{OPTIMIZED_SUFFIX}{TRANSLATED_SUFFIX}.json",
        temporary_dir / f"{stem}{RELAXED_OPTIMIZED_SUFFIX}{TRANSLATED_SUFFIX}.json",
        temporary_dir / f"{base_name}{BILINGUAL_SUFFIX}.json",
    )


def _recovery_paths(temporary_dir: Path, output_dir: Path, base_name: str) -> tuple[Path, ...]:
    return (
        temporary_dir / f"{base_name}{COMPARE_SUFFIX}.json",
        output_dir / f"{base_name}{EDIT_REPORT_SUFFIX}.json",
        output_dir / f"{base_name}{EDIT_SESSION_SUFFIX}.json",
    )


def _translation_subtitle_suffix(path: Path, base_name: str) -> str:
    source_stem = path.parent.parent / base_name
    video_suffixes = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m4v", ".flv", ".wmv")
    return ".srt" if any(source_stem.with_suffix(suffix).exists() for suffix in video_suffixes) else ".lrc"
