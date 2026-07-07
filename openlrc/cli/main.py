#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Typer-based command line interface for OpenLRC Mac."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from openlrc import __app_name__, __dist_name__, __upstream_version__, __version__
from openlrc.config import TranscriptionConfig
from openlrc.llama_resources import (
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_MODEL_REPO,
    DEFAULT_LLAMA_PORT,
    resolve_llama_cli,
    resolve_llama_model_path,
    resolve_llama_server,
    user_llm_model_dir,
)
from openlrc.llama_resources import vendor_dir as llama_vendor_dir
from openlrc.setup.llama_cpp import LlamaSetupResult, setup_llama_cpp
from openlrc.setup.whisper_cpp import DEFAULT_MODEL, DEFAULT_VAD_MODEL, WhisperSetupResult, setup_whisper_cpp
from openlrc.whisper_resources import (
    DEFAULT_MODEL_NAME,
    DEFAULT_VAD_MODEL_NAME,
    resolve_vad_model_path,
    resolve_whisper_cli,
    resolve_whisper_model_path,
    user_model_dir,
)
from openlrc.whisper_resources import vendor_dir as whisper_vendor_dir

if TYPE_CHECKING:
    from openlrc.openlrc import LRCer

console = Console()

app = typer.Typer(
    help="OpenLRC Mac command line tools.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
setup_app = typer.Typer(help="Build local toolchains and download models.", no_args_is_help=True, add_completion=False)
models_app = typer.Typer(help="Inspect local model state.", no_args_is_help=True, add_completion=False)
app.add_typer(setup_app, name="setup")
app.add_typer(models_app, name="models")


class TranslationBackend(str, Enum):
    none = "none"
    local = "local"
    online = "online"


class TranslationOnlyBackend(str, Enum):
    local = "local"
    online = "online"


class TranslateMode(str, Enum):
    lean = "lean"
    standard = "standard"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _check_path(name: str, path: Path, hint: str) -> CheckResult:
    if path.exists():
        return CheckResult(name, True, str(path))
    return CheckResult(name, False, f"Missing: {path}", hint)


def _check_executable(name: str, executable: str, hint: str) -> CheckResult:
    resolved = shutil.which(executable)
    if resolved:
        return CheckResult(name, True, resolved)
    return CheckResult(name, False, f"{executable} not found on PATH", hint)


def _check_resolver(name: str, resolver, hint: str) -> CheckResult:
    try:
        resolved = resolver()
    except Exception as exc:
        return CheckResult(name, False, str(exc), hint)
    return CheckResult(name, True, str(resolved))


def _render_checks(title: str, checks: list[CheckResult]) -> None:
    table = Table(title=title)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_column("Hint")

    for check in checks:
        status = "[green]OK[/green]" if check.ok else "[red]Missing[/red]"
        table.add_row(check.name, status, check.detail, check.hint)

    console.print(table)


def _doctor_checks() -> list[CheckResult]:
    return [
        _check_executable("ffmpeg", "ffmpeg", "Install ffmpeg and make sure it is on PATH."),
        _check_executable("cmake", "cmake", "Install CMake and Xcode Command Line Tools."),
        _check_path(
            "whisper.cpp submodule",
            whisper_vendor_dir() / "CMakeLists.txt",
            "Run `git submodule update --init --recursive`.",
        ),
        _check_resolver("whisper-cli", lambda: resolve_whisper_cli(""), "Run `openlrc setup whisper`."),
        _check_resolver(
            "Whisper model",
            lambda: resolve_whisper_model_path(DEFAULT_MODEL_NAME),
            "Run `openlrc setup whisper --model base`.",
        ),
        _check_resolver(
            "Whisper VAD model", lambda: resolve_vad_model_path(DEFAULT_VAD_MODEL_NAME), "Run `openlrc setup whisper`."
        ),
        _check_path(
            "llama.cpp submodule",
            llama_vendor_dir() / "CMakeLists.txt",
            "Run `git submodule update --init --recursive`.",
        ),
        _check_resolver("llama-server", lambda: resolve_llama_server(""), "Run `openlrc setup llama`."),
        _check_resolver("llama-cli", lambda: resolve_llama_cli(""), "Run `openlrc setup llama`."),
        _check_resolver(
            "Qwen GGUF model", lambda: resolve_llama_model_path(DEFAULT_LLAMA_MODEL_FILE), "Run `openlrc setup llama`."
        ),
    ]


def _model_checks() -> list[CheckResult]:
    return [
        _check_resolver(
            "Whisper default model",
            lambda: resolve_whisper_model_path(DEFAULT_MODEL_NAME),
            f"Default directory: {user_model_dir()}",
        ),
        _check_resolver(
            "Whisper VAD model",
            lambda: resolve_vad_model_path(DEFAULT_VAD_MODEL_NAME),
            f"Default directory: {user_model_dir()}",
        ),
        _check_resolver(
            "Local Qwen GGUF",
            lambda: resolve_llama_model_path(DEFAULT_LLAMA_MODEL_FILE),
            f"Default directory: {user_llm_model_dir()}",
        ),
    ]


def _transcription_config(whisper_model: str, vad_model: str) -> TranscriptionConfig:
    return TranscriptionConfig(whisper_model=whisper_model, vad_model=vad_model)


def _lrcer_cls():
    from openlrc.openlrc import LRCer

    return LRCer


def _lrcer_for_run(
    *,
    translation: TranslationBackend,
    whisper_model: str,
    vad_model: str,
    llama_model: str,
    llama_port: int,
    idle_timeout: int,
    translate_mode: TranslateMode,
) -> LRCer:
    lrcer_cls = _lrcer_cls()
    transcription = _transcription_config(whisper_model, vad_model)
    if translation == TranslationBackend.local:
        return lrcer_cls.local(
            model=llama_model,
            idle_timeout=idle_timeout,
            port=llama_port,
            translate_mode=translate_mode.value,
            transcription=transcription,
        )
    return lrcer_cls(transcription=transcription)


def _lrcer_for_translation(
    *,
    translation: TranslationOnlyBackend,
    llama_model: str,
    llama_port: int,
    idle_timeout: int,
    translate_mode: TranslateMode,
) -> LRCer:
    lrcer_cls = _lrcer_cls()
    if translation == TranslationOnlyBackend.local:
        return lrcer_cls.local(
            model=llama_model, idle_timeout=idle_timeout, port=llama_port, translate_mode=translate_mode.value
        )
    return lrcer_cls()


def _print_outputs(outputs: list[Path] | list[str]) -> None:
    if not outputs:
        console.print("[yellow]No output files generated.[/yellow]")
        return

    table = Table(title="Generated files")
    table.add_column("#", justify="right")
    table.add_column("Path")
    for index, output in enumerate(outputs, start=1):
        table.add_row(str(index), str(output))
    console.print(table)


def _print_whisper_setup_result(result: WhisperSetupResult) -> None:
    table = Table(title="whisper.cpp setup complete")
    table.add_column("Resource")
    table.add_column("Path")
    if result.cli_path:
        table.add_row("whisper-cli", str(result.cli_path))
    if result.whisper_model:
        table.add_row("Whisper model", str(result.whisper_model))
    if result.vad_model:
        table.add_row("VAD model", str(result.vad_model))
    console.print(table)


def _print_llama_setup_result(result: LlamaSetupResult) -> None:
    table = Table(title="llama.cpp setup complete")
    table.add_column("Resource")
    table.add_column("Path")
    if result.server_path:
        table.add_row("llama-server", str(result.server_path))
    if result.cli_path:
        table.add_row("llama-cli", str(result.cli_path))
    if result.model_path:
        table.add_row("LLM model", str(result.model_path))
    console.print(table)


@app.callback()
def _callback(
    version: Annotated[bool, typer.Option("--version", help="Show OpenLRC Mac version and exit.")] = False,
) -> None:
    if version:
        console.print(
            f"{__app_name__} {__version__} "
            f"(distribution: {__dist_name__}; upstream base: OpenLRC {__upstream_version__})"
        )
        raise typer.Exit()


@app.command()
def doctor(
    strict: Annotated[bool, typer.Option("--strict", help="Exit with a non-zero status if any check fails.")] = False,
) -> None:
    """Check local tools, submodules, binaries, and model files."""
    checks = _doctor_checks()
    _render_checks("OpenLRC doctor", checks)
    if strict and any(not check.ok for check in checks):
        raise typer.Exit(code=1)


@models_app.command("status")
def models_status() -> None:
    """Show installed status for the default local models."""
    _render_checks("OpenLRC model status", _model_checks())


@setup_app.command("whisper")
def setup_whisper_command(
    model: Annotated[str, typer.Option("--model", help="whisper.cpp model name.")] = DEFAULT_MODEL,
    vad_model: Annotated[str, typer.Option("--vad-model", help="whisper.cpp VAD model name.")] = DEFAULT_VAD_MODEL,
    model_dir: Annotated[Path | None, typer.Option("--model-dir", help="Directory for downloaded models.")] = None,
    skip_build: Annotated[
        bool, typer.Option("--skip-build", help="Initialize submodule but skip CMake build.")
    ] = False,
    skip_models: Annotated[bool, typer.Option("--skip-models", help="Skip model downloads.")] = False,
) -> None:
    """Build whisper.cpp and download Whisper/VAD models."""
    result = setup_whisper_cpp(
        model=model, vad_model=vad_model, model_dir=model_dir, skip_build=skip_build, skip_models=skip_models
    )
    _print_whisper_setup_result(result)


@setup_app.command("llama")
def setup_llama_command(
    model_repo: Annotated[
        str, typer.Option("--model-repo", help="Hugging Face GGUF model repo.")
    ] = DEFAULT_LLAMA_MODEL_REPO,
    model_file: Annotated[str, typer.Option("--model-file", help="GGUF model filename.")] = DEFAULT_LLAMA_MODEL_FILE,
    revision: Annotated[str, typer.Option("--revision", help="Hugging Face revision to download.")] = "main",
    model_url: Annotated[str | None, typer.Option("--model-url", help="Direct GGUF download URL.")] = None,
    model_dir: Annotated[
        Path | None, typer.Option("--model-dir", help="Directory for downloaded GGUF model files.")
    ] = None,
    skip_build: Annotated[
        bool, typer.Option("--skip-build", help="Initialize submodule but skip CMake build.")
    ] = False,
    skip_models: Annotated[bool, typer.Option("--skip-models", help="Skip model downloads.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-download the model even if it exists.")] = False,
) -> None:
    """Build llama.cpp and download the default local LLM model."""
    result = setup_llama_cpp(
        model_repo=model_repo,
        model_file=model_file,
        revision=revision,
        model_url=model_url,
        model_dir=model_dir,
        skip_build=skip_build,
        skip_models=skip_models,
        force=force,
    )
    _print_llama_setup_result(result)


@setup_app.command("all")
def setup_all_command(
    skip_build: Annotated[
        bool, typer.Option("--skip-build", help="Initialize submodules but skip CMake builds.")
    ] = False,
    skip_models: Annotated[bool, typer.Option("--skip-models", help="Skip all model downloads.")] = False,
) -> None:
    """Build whisper.cpp and llama.cpp, then download default models."""
    whisper_result = setup_whisper_cpp(skip_build=skip_build, skip_models=skip_models)
    llama_result = setup_llama_cpp(skip_build=skip_build, skip_models=skip_models)
    _print_whisper_setup_result(whisper_result)
    _print_llama_setup_result(llama_result)


@app.command()
def transcribe(
    paths: Annotated[list[Path], typer.Argument(help="Input audio/video paths.")],
    src_lang: Annotated[str | None, typer.Option("--src-lang", help="Source language code.")] = None,
    whisper_model: Annotated[
        str, typer.Option("--whisper-model", help="Whisper model name, filename, or path.")
    ] = DEFAULT_MODEL_NAME,
    vad_model: Annotated[
        str, typer.Option("--vad-model", help="VAD model name, filename, or path. Empty disables VAD.")
    ] = DEFAULT_VAD_MODEL_NAME,
    noise_suppress: Annotated[
        bool, typer.Option("--noise-suppress", help="Apply noise suppression before transcription.")
    ] = False,
    skip_preprocess: Annotated[
        bool, typer.Option("--skip-preprocess", help="Use existing preprocessed audio files.")
    ] = False,
) -> None:
    """Transcribe audio/video files and write transcription JSON."""
    lrcer = _lrcer_cls()(transcription=_transcription_config(whisper_model, vad_model))
    try:
        outputs = lrcer.transcribe(
            cast(list[str | Path], paths),
            src_lang=src_lang,
            noise_suppress=noise_suppress,
            skip_preprocess=skip_preprocess,
        )
        _print_outputs(outputs)
    finally:
        lrcer.close()


@app.command()
def translate(
    json_paths: Annotated[list[Path], typer.Argument(help="Transcribed JSON files.")],
    translation: Annotated[TranslationOnlyBackend, typer.Option("--translation", help="Translation backend to use.")],
    target_lang: Annotated[str, typer.Option("--target-lang", help="Target language code.")] = "zh-cn",
    bilingual_sub: Annotated[bool, typer.Option("--bilingual-sub", help="Generate bilingual subtitle files.")] = False,
    llama_model: Annotated[
        str, typer.Option("--llama-model", help="Local GGUF model alias, filename, or path.")
    ] = "qwen3.5-9b",
    llama_port: Annotated[int, typer.Option("--llama-port", help="Local llama-server port.")] = DEFAULT_LLAMA_PORT,
    idle_timeout: Annotated[
        int, typer.Option("--idle-timeout", help="Seconds before an owned local server shuts down.")
    ] = DEFAULT_LLAMA_IDLE_TIMEOUT,
    translate_mode: Annotated[
        TranslateMode, typer.Option("--translate-mode", help="OpenLRC translation strategy.")
    ] = TranslateMode.lean,
) -> None:
    """Translate existing transcription JSON files."""
    lrcer = _lrcer_for_translation(
        translation=translation,
        llama_model=llama_model,
        llama_port=llama_port,
        idle_timeout=idle_timeout,
        translate_mode=translate_mode,
    )
    try:
        outputs = lrcer.translate(json_paths, target_lang=target_lang, bilingual_sub=bilingual_sub)
        _print_outputs(outputs)
    finally:
        lrcer.close()


@app.command()
def run(
    paths: Annotated[list[Path], typer.Argument(help="Input audio/video paths.")],
    translation: Annotated[
        TranslationBackend, typer.Option("--translation", help="Translation backend to use.")
    ] = TranslationBackend.none,
    src_lang: Annotated[str | None, typer.Option("--src-lang", help="Source language code.")] = None,
    target_lang: Annotated[str, typer.Option("--target-lang", help="Target language code.")] = "zh-cn",
    whisper_model: Annotated[
        str, typer.Option("--whisper-model", help="Whisper model name, filename, or path.")
    ] = DEFAULT_MODEL_NAME,
    vad_model: Annotated[
        str, typer.Option("--vad-model", help="VAD model name, filename, or path. Empty disables VAD.")
    ] = DEFAULT_VAD_MODEL_NAME,
    noise_suppress: Annotated[
        bool, typer.Option("--noise-suppress", help="Apply noise suppression before transcription.")
    ] = False,
    bilingual_sub: Annotated[bool, typer.Option("--bilingual-sub", help="Generate bilingual subtitle files.")] = False,
    clear_temp: Annotated[
        bool, typer.Option("--clear-temp", help="Clear preprocessed temporary files after success.")
    ] = False,
    skip_preprocess: Annotated[
        bool, typer.Option("--skip-preprocess", help="Use existing preprocessed audio files.")
    ] = False,
    llama_model: Annotated[
        str, typer.Option("--llama-model", help="Local GGUF model alias, filename, or path.")
    ] = "qwen3.5-9b",
    llama_port: Annotated[int, typer.Option("--llama-port", help="Local llama-server port.")] = DEFAULT_LLAMA_PORT,
    idle_timeout: Annotated[
        int, typer.Option("--idle-timeout", help="Seconds before an owned local server shuts down.")
    ] = DEFAULT_LLAMA_IDLE_TIMEOUT,
    translate_mode: Annotated[
        TranslateMode, typer.Option("--translate-mode", help="OpenLRC translation strategy.")
    ] = TranslateMode.lean,
) -> None:
    """Run the transcription pipeline and optionally translate subtitles."""
    lrcer = _lrcer_for_run(
        translation=translation,
        whisper_model=whisper_model,
        vad_model=vad_model,
        llama_model=llama_model,
        llama_port=llama_port,
        idle_timeout=idle_timeout,
        translate_mode=translate_mode,
    )
    try:
        outputs = lrcer.run(
            cast(list[str | Path], paths),
            src_lang=src_lang,
            target_lang=target_lang,
            skip_trans=translation == TranslationBackend.none,
            noise_suppress=noise_suppress,
            bilingual_sub=bilingual_sub,
            clear_temp=clear_temp,
            skip_preprocess=skip_preprocess,
        )
        _print_outputs(outputs)
    finally:
        lrcer.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
