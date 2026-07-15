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
from openlrc.config import ContextLLMConfig, HyMT2Mode, TranscriptionConfig
from openlrc.llama_resources import (
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_MODEL_REPO,
    DEFAULT_LLAMA_PORT,
    HY_MT2_7B_MODEL_FILE,
    HY_MT2_7B_PROFILE,
    HY_MT2_30B_A3B_PROFILE,
    QWEN35_9B_PROFILE,
    get_local_llm_profile,
    infer_local_llm_profile,
    is_hy_mt2_30b_profile_alias,
    resolve_llama_cli,
    resolve_llama_model_path,
    resolve_llama_server,
    user_llm_model_dir,
)
from openlrc.llama_resources import vendor_dir as llama_vendor_dir
from openlrc.models import ModelProvider
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


class LocalModelProfile(str, Enum):
    qwen35_9b = QWEN35_9B_PROFILE
    hy_mt2_7b = HY_MT2_7B_PROFILE
    hy_mt2_30b_a3b = HY_MT2_30B_A3B_PROFILE


class ContextProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    google = "google"
    litellm = "litellm"
    third_party = "third-party"
    local = "local"


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
        _check_resolver(
            "Local Hy-MT2 7B Q6_K GGUF",
            lambda: resolve_llama_model_path(HY_MT2_7B_MODEL_FILE),
            f"Optional. Run `openlrc setup llama --local-model-profile {HY_MT2_7B_PROFILE}`.",
        ),
    ]


def _transcription_config(whisper_model: str, vad_model: str) -> TranscriptionConfig:
    return TranscriptionConfig(whisper_model=whisper_model, vad_model=vad_model)


def _lrcer_cls():
    from openlrc.openlrc import LRCer

    return LRCer


def _selected_profile_and_model(
    *, local_model_profile: LocalModelProfile | None, llama_model: str
) -> tuple[str, str | None]:
    if local_model_profile is None and is_hy_mt2_30b_profile_alias(llama_model):
        raise typer.BadParameter(
            f"{HY_MT2_30B_A3B_PROFILE} must be selected with --local-model-profile and a local .gguf path."
        )

    inferred_profile = infer_local_llm_profile(llama_model)
    if local_model_profile is not None:
        selected_profile = local_model_profile.value
        if is_hy_mt2_30b_profile_alias(llama_model) and selected_profile != HY_MT2_30B_A3B_PROFILE:
            raise typer.BadParameter(
                f"--llama-model {llama_model!r} conflicts with --local-model-profile {selected_profile!r}."
            )
        if inferred_profile and inferred_profile != selected_profile and llama_model != QWEN35_9B_PROFILE:
            raise typer.BadParameter(
                f"--llama-model {llama_model!r} conflicts with --local-model-profile {selected_profile!r}."
            )

    profile_name = local_model_profile.value if local_model_profile else inferred_profile
    profile_name = profile_name or QWEN35_9B_PROFILE

    if profile_name == HY_MT2_30B_A3B_PROFILE:
        if llama_model == QWEN35_9B_PROFILE or is_hy_mt2_30b_profile_alias(llama_model):
            raise typer.BadParameter(
                f"--local-model-profile {HY_MT2_30B_A3B_PROFILE} requires --llama-model /path/to/model.gguf."
            )
        if not (llama_model.endswith(".gguf") or "/" in llama_model or Path(llama_model).is_absolute()):
            raise typer.BadParameter(f"--llama-model for {HY_MT2_30B_A3B_PROFILE} must be a GGUF filename or path.")
        return profile_name, llama_model

    if profile_name == HY_MT2_7B_PROFILE:
        if local_model_profile is not None and llama_model == QWEN35_9B_PROFILE:
            return profile_name, None
        return profile_name, llama_model

    return profile_name, llama_model


def _context_llm_config(
    *,
    mode: HyMT2Mode,
    provider: ContextProvider | None,
    model: str | None,
    base_url: str | None,
    fee_limit: float,
    port: int,
) -> ContextLLMConfig | None:
    if mode is HyMT2Mode.FAST:
        return None
    if provider is None or not model:
        raise typer.BadParameter(f"--hy-mt2-mode {mode.value} requires both --context-provider and --context-model.")
    if provider is ContextProvider.local:
        if base_url:
            raise typer.BadParameter("--context-base-url cannot be used with --context-provider local.")
        return ContextLLMConfig.local_qwen35_9b(model=model, port=port)

    provider_map = {
        ContextProvider.openai: ModelProvider.OPENAI,
        ContextProvider.anthropic: ModelProvider.ANTHROPIC,
        ContextProvider.google: ModelProvider.GOOGLE,
        ContextProvider.litellm: ModelProvider.LITELLM,
        ContextProvider.third_party: ModelProvider.THIRD_PARTY,
    }
    if provider is ContextProvider.third_party and not base_url:
        raise typer.BadParameter("--context-provider third-party requires --context-base-url.")
    return ContextLLMConfig.online(provider=provider_map[provider], model=model, base_url=base_url, fee_limit=fee_limit)


def _lrcer_for_run(
    *,
    translation: TranslationBackend,
    whisper_model: str,
    vad_model: str,
    llama_model: str,
    local_model_profile: LocalModelProfile | None,
    llama_port: int,
    idle_timeout: int,
    hy_mt2_mode: HyMT2Mode,
    context_provider: ContextProvider | None,
    context_model: str | None,
    context_base_url: str | None,
    context_fee_limit: float,
) -> LRCer:
    lrcer_cls = _lrcer_cls()
    transcription = _transcription_config(whisper_model, vad_model)
    if translation == TranslationBackend.local:
        profile_name, selected_model = _selected_profile_and_model(
            local_model_profile=local_model_profile, llama_model=llama_model
        )
        if profile_name != QWEN35_9B_PROFILE:
            context_llm = _context_llm_config(
                mode=hy_mt2_mode,
                provider=context_provider,
                model=context_model,
                base_url=context_base_url,
                fee_limit=context_fee_limit,
                port=llama_port,
            )
            return lrcer_cls.local_hy_mt2(
                size=profile_name,
                model=selected_model,
                idle_timeout=idle_timeout,
                port=llama_port,
                transcription=transcription,
                mode=hy_mt2_mode,
                context_llm=context_llm,
            )
        if hy_mt2_mode is not HyMT2Mode.FAST or context_provider is not None or context_model is not None:
            raise typer.BadParameter("Hy-MT2 context options require a Hy-MT2 local model profile.")
        return lrcer_cls.local(
            model=selected_model or llama_model, idle_timeout=idle_timeout, port=llama_port, transcription=transcription
        )
    return lrcer_cls(transcription=transcription)


def _lrcer_for_translation(
    *,
    translation: TranslationOnlyBackend,
    llama_model: str,
    local_model_profile: LocalModelProfile | None,
    llama_port: int,
    idle_timeout: int,
    hy_mt2_mode: HyMT2Mode,
    context_provider: ContextProvider | None,
    context_model: str | None,
    context_base_url: str | None,
    context_fee_limit: float,
) -> LRCer:
    lrcer_cls = _lrcer_cls()
    if translation == TranslationOnlyBackend.local:
        profile_name, selected_model = _selected_profile_and_model(
            local_model_profile=local_model_profile, llama_model=llama_model
        )
        if profile_name != QWEN35_9B_PROFILE:
            context_llm = _context_llm_config(
                mode=hy_mt2_mode,
                provider=context_provider,
                model=context_model,
                base_url=context_base_url,
                fee_limit=context_fee_limit,
                port=llama_port,
            )
            return lrcer_cls.local_hy_mt2(
                size=profile_name,
                model=selected_model,
                idle_timeout=idle_timeout,
                port=llama_port,
                mode=hy_mt2_mode,
                context_llm=context_llm,
            )
        if hy_mt2_mode is not HyMT2Mode.FAST or context_provider is not None or context_model is not None:
            raise typer.BadParameter("Hy-MT2 context options require a Hy-MT2 local model profile.")
        return lrcer_cls.local(model=selected_model or llama_model, idle_timeout=idle_timeout, port=llama_port)
    return lrcer_cls()


def _print_outputs(outputs: list[Path] | list[str], review_statuses: dict[str, dict] | None = None) -> None:
    if not outputs:
        console.print("[yellow]No output files generated.[/yellow]")
        return

    table = Table(title="Generated files")
    table.add_column("#", justify="right")
    table.add_column("Path")
    review_statuses = review_statuses or {}
    if review_statuses:
        table.add_column("Review status")
    for index, output in enumerate(outputs, start=1):
        status = review_statuses.get(Path(output).stem)
        status_text = ""
        if status:
            if status.get("incomplete"):
                failed = status.get("failed_chunks", [])
                status_text = f"[yellow]incomplete ({len(failed)} chunk(s) kept as Hy-MT2 draft)[/yellow]"
            else:
                status_text = "[green]complete[/green]"
        row = [str(index), str(output)]
        if review_statuses:
            row.append(status_text)
        table.add_row(*row)
    console.print(table)
    for name, status in review_statuses.items():
        if status.get("incomplete"):
            console.print(
                f"[yellow]Review incomplete for {name}: chunks {status.get('failed_chunks', [])} kept their "
                "Hy-MT2 draft. Temporary checkpoint retained for the next run.[/yellow]"
            )


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
    local_model_profile: Annotated[
        LocalModelProfile | None,
        typer.Option("--local-model-profile", help="Registered local LLM profile to download."),
    ] = None,
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
    if local_model_profile is not None and not skip_models:
        profile = get_local_llm_profile(local_model_profile.value)
        if not profile.model_repo or not profile.model_file:
            raise typer.BadParameter(
                f"{profile.name} has no downloadable default GGUF. Place a converted model locally instead."
            )
        if model_repo == DEFAULT_LLAMA_MODEL_REPO:
            model_repo = profile.model_repo
        if model_file == DEFAULT_LLAMA_MODEL_FILE:
            model_file = profile.model_file

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
    keep_checkpoint: Annotated[
        bool, typer.Option("--keep-checkpoint", help="Keep a completed translation checkpoint for debugging.")
    ] = False,
    llama_model: Annotated[
        str, typer.Option("--llama-model", help="Local GGUF model alias, filename, or path.")
    ] = QWEN35_9B_PROFILE,
    local_model_profile: Annotated[
        LocalModelProfile | None,
        typer.Option("--local-model-profile", help="Local LLM profile for sampling and prompt settings."),
    ] = None,
    llama_port: Annotated[int, typer.Option("--llama-port", help="Local llama-server port.")] = DEFAULT_LLAMA_PORT,
    idle_timeout: Annotated[
        int, typer.Option("--idle-timeout", help="Seconds before an owned local server shuts down.")
    ] = DEFAULT_LLAMA_IDLE_TIMEOUT,
    hy_mt2_mode: Annotated[
        HyMT2Mode, typer.Option("--hy-mt2-mode", help="Hy-MT2 quality/context mode.")
    ] = HyMT2Mode.FAST,
    context_provider: Annotated[
        ContextProvider | None, typer.Option("--context-provider", help="General model provider for Hy-MT2 context.")
    ] = None,
    context_model: Annotated[
        str | None, typer.Option("--context-model", help="General model name, local alias, GGUF filename, or path.")
    ] = None,
    context_base_url: Annotated[
        str | None, typer.Option("--context-base-url", help="Custom OpenAI-compatible context-model endpoint.")
    ] = None,
    context_fee_limit: Annotated[
        float, typer.Option("--context-fee-limit", help="Maximum estimated fee per context-model call.")
    ] = 0.8,
) -> None:
    """Translate existing transcription JSON files."""
    lrcer = _lrcer_for_translation(
        translation=translation,
        llama_model=llama_model,
        local_model_profile=local_model_profile,
        llama_port=llama_port,
        idle_timeout=idle_timeout,
        hy_mt2_mode=hy_mt2_mode,
        context_provider=context_provider,
        context_model=context_model,
        context_base_url=context_base_url,
        context_fee_limit=context_fee_limit,
    )
    try:
        outputs = lrcer.translate(
            json_paths, target_lang=target_lang, bilingual_sub=bilingual_sub, clear_checkpoint=not keep_checkpoint
        )
        _print_outputs(outputs, lrcer.review_statuses)
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
        bool,
        typer.Option(
            "--clear-temp/--keep-temp",
            help="Clear temporary files after complete success; incomplete review is always retained.",
        ),
    ] = True,
    skip_preprocess: Annotated[
        bool, typer.Option("--skip-preprocess", help="Use existing preprocessed audio files.")
    ] = False,
    llama_model: Annotated[
        str, typer.Option("--llama-model", help="Local GGUF model alias, filename, or path.")
    ] = QWEN35_9B_PROFILE,
    local_model_profile: Annotated[
        LocalModelProfile | None,
        typer.Option("--local-model-profile", help="Local LLM profile for sampling and prompt settings."),
    ] = None,
    llama_port: Annotated[int, typer.Option("--llama-port", help="Local llama-server port.")] = DEFAULT_LLAMA_PORT,
    idle_timeout: Annotated[
        int, typer.Option("--idle-timeout", help="Seconds before an owned local server shuts down.")
    ] = DEFAULT_LLAMA_IDLE_TIMEOUT,
    hy_mt2_mode: Annotated[
        HyMT2Mode, typer.Option("--hy-mt2-mode", help="Hy-MT2 quality/context mode.")
    ] = HyMT2Mode.FAST,
    context_provider: Annotated[
        ContextProvider | None, typer.Option("--context-provider", help="General model provider for Hy-MT2 context.")
    ] = None,
    context_model: Annotated[
        str | None, typer.Option("--context-model", help="General model name, local alias, GGUF filename, or path.")
    ] = None,
    context_base_url: Annotated[
        str | None, typer.Option("--context-base-url", help="Custom OpenAI-compatible context-model endpoint.")
    ] = None,
    context_fee_limit: Annotated[
        float, typer.Option("--context-fee-limit", help="Maximum estimated fee per context-model call.")
    ] = 0.8,
) -> None:
    """Run the transcription pipeline and optionally translate subtitles."""
    lrcer = _lrcer_for_run(
        translation=translation,
        whisper_model=whisper_model,
        vad_model=vad_model,
        llama_model=llama_model,
        local_model_profile=local_model_profile,
        llama_port=llama_port,
        idle_timeout=idle_timeout,
        hy_mt2_mode=hy_mt2_mode,
        context_provider=context_provider,
        context_model=context_model,
        context_base_url=context_base_url,
        context_fee_limit=context_fee_limit,
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
        _print_outputs(outputs, lrcer.review_statuses)
    finally:
        lrcer.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
