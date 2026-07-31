#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Typer-based command line interface for OpenLRC Mac."""

from __future__ import annotations

import json
import shutil
import signal
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from openlrc import __app_name__, __dist_name__, __upstream_version__, __version__
from openlrc.application.resources import ResourceStatusService
from openlrc.config import (
    ContextAssistance,
    ContextLLMConfig,
    EditConfig,
    GlossaryOptions,
    HyMT2Mode,
    SubtitleOptimizationMode,
    TranscriptionConfig,
    TranslationConfig,
    context_model_required,
    normalize_hymt2_mode,
)
from openlrc.context import TranslationBriefInput
from openlrc.editing import EditAction, EditSeverity, parse_segment_ids
from openlrc.llama_resources import (
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_MODEL_REPO,
    DEFAULT_LLAMA_PORT,
    HY_MT2_7B_PROFILE,
    HY_MT2_30B_A3B_PROFILE,
    QWEN35_9B_PROFILE,
    get_local_llm_profile,
    infer_local_llm_profile,
    is_hy_mt2_30b_profile_alias,
)
from openlrc.models import ModelProvider
from openlrc.setup.llama_cpp import LlamaSetupResult, setup_llama_cpp
from openlrc.setup.whisper_cpp import DEFAULT_MODEL, DEFAULT_VAD_MODEL, WhisperSetupResult, setup_whisper_cpp
from openlrc.whisper_resources import DEFAULT_MODEL_NAME, DEFAULT_VAD_MODEL_NAME
from openlrc.workflow import (
    CancellationToken,
    ExecutionContext,
    ModelLifecycleEvent,
    RunRequest,
    StageProgressEvent,
    TranscribeRequest,
    TranslateRequest,
    TranslationMode,
    WorkflowExecutor,
    WorkflowKind,
    WorkflowResult,
    WorkflowStatus,
    WorkflowTranslationConfig,
    WorkflowTranslationFactory,
)

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
glossary_app = typer.Typer(help="Validate and inspect task glossaries.", no_args_is_help=True, add_completion=False)
app.add_typer(setup_app, name="setup")
app.add_typer(models_app, name="models")
app.add_typer(glossary_app, name="glossary")


@app.command("tui")
def tui_command() -> None:
    """Open the keyboard-first Textual interface."""
    from openlrc.tui import run

    run()


class TranslationBackend(StrEnum):
    none = "none"
    local = "local"
    online = "online"


class TranslationOnlyBackend(StrEnum):
    local = "local"
    online = "online"


class LocalModelProfile(StrEnum):
    qwen35_9b = QWEN35_9B_PROFILE
    hy_mt2_7b = HY_MT2_7B_PROFILE
    hy_mt2_30b_a3b = HY_MT2_30B_A3B_PROFILE


class ContextProvider(StrEnum):
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


class RichEventSink:
    """Concise terminal progress adapter for typed workflow events."""

    def __init__(self, output: Console = console) -> None:
        self.console = output
        self._reported: dict[tuple[str | None, str], int] = {}

    def __call__(self, event) -> None:
        # Captured/non-interactive output remains free of dynamic control codes.
        if not self.console.is_terminal:
            return
        if isinstance(event, StageProgressEvent):
            bucket = min(100, int(event.percent // 10) * 10)
            key = (event.header.item, event.stage.value)
            if bucket <= self._reported.get(key, -1):
                return
            self._reported[key] = bucket
            item = f" {Path(event.header.item).name}" if event.header.item else ""
            self.console.print(f"[cyan]{event.stage.value}[/cyan]{item}: {event.percent:5.1f}%")
        elif isinstance(event, ModelLifecycleEvent):
            self.console.print(f"[cyan]{event.model}[/cyan]: {event.state}")


def _review_status_mapping(result: WorkflowResult) -> dict[str, dict]:
    statuses: dict[str, dict] = {}
    for review in result.reviews:
        status = dict(review.details)
        status["incomplete"] = review.incomplete
        if review.checkpoint is not None:
            status["checkpoint"] = str(review.checkpoint)
        if review.report is not None:
            status["report"] = str(review.report)
        if review.session is not None:
            status["session"] = str(review.session)
        statuses[review.item] = status
    return statuses


def _execute_workflow(request, workflow: WorkflowKind) -> WorkflowResult:
    token = CancellationToken()
    context = ExecutionContext(workflow, event_sink=RichEventSink(), cancellation_token=token)
    previous_sigint = None
    if threading.current_thread() is threading.main_thread():
        previous_sigint = signal.getsignal(signal.SIGINT)

        def cancel_then_interrupt(_signum, _frame) -> None:
            token.cancel()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, cancel_then_interrupt)
    try:
        result = WorkflowExecutor().execute(request, context)
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
    if result.status is WorkflowStatus.FAILED:
        assert result.error is not None
        detail = f" ({result.error.hint})" if result.error.hint else ""
        console.print(f"[red]{result.error.category.value}: {result.error.message}{detail}[/red]")
        raise typer.Exit(code=1)
    if result.status is WorkflowStatus.CANCELLED:
        console.print("[yellow]Cancelled after cleaning up owned resources.[/yellow]")
        raise typer.Exit(code=130)
    return result


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
    return [CheckResult(item.name, item.available, item.detail, item.hint) for item in ResourceStatusService().doctor()]


def _model_checks() -> list[CheckResult]:
    return [CheckResult(item.name, item.available, item.detail, item.hint) for item in ResourceStatusService().models()]


def _transcription_config(
    whisper_model: str, vad_model: str, *, whisper_gpu: bool = True, whisper_flash_attn: bool = True
) -> TranscriptionConfig:
    return TranscriptionConfig(
        whisper_model=whisper_model,
        vad_model=vad_model,
        asr_options={"use_gpu": whisper_gpu, "flash_attn": whisper_flash_attn},
    )


def _glossary_and_edit_options(
    *, glossary: Path | None, force_glossary: bool, glossary_strict: bool, edit_rounds: int, enable_restore: bool
) -> tuple[str | None, GlossaryOptions, EditConfig]:
    if not 0 <= edit_rounds <= 3:
        raise typer.BadParameter("--edit-rounds must be between 0 and 3.")
    return (
        str(glossary) if glossary is not None else None,
        GlossaryOptions(strict=glossary_strict, force=force_glossary),
        EditConfig(
            enabled=False, max_rounds=edit_rounds, semantic_review=edit_rounds > 0, restore_enabled=enable_restore
        ),
    )


def _translation_brief_input(
    *,
    summary: str | None,
    characters: str | None,
    tone_style: str | None,
    context_assistance: ContextAssistance = ContextAssistance.AUTO,
    mode: HyMT2Mode | None = None,
) -> TranslationBriefInput | None:
    """Parse strict CLI brief fields while preserving omitted versus explicitly empty values."""
    assistance = ContextAssistance(context_assistance)
    selected_mode = HyMT2Mode(mode).canonical if mode is not None else None
    contextual_off = assistance is ContextAssistance.OFF and selected_mode is not HyMT2Mode.FAST
    if contextual_off:
        if summary is None or not summary.strip():
            raise typer.BadParameter("Context assistance off requires --brief-summary.", param_hint="--brief-summary")
        summary = summary.strip()
        characters = characters if characters is not None else ""
        tone_style = tone_style if tone_style is not None else ""
    if summary is None and characters is None and tone_style is None:
        return None

    parsed_characters = None
    if characters is not None:
        raw = characters
        if contextual_off and not raw.strip():
            parsed_characters = []
        elif raw.startswith("@"):
            path_text = raw[1:]
            if not path_text:
                raise typer.BadParameter("Expected a UTF-8 JSON file after '@'.", param_hint="--brief-characters")
            path = Path(path_text)
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise typer.BadParameter(
                    f"Cannot read brief characters file {path}: {exc}", param_hint="--brief-characters"
                ) from exc
        if parsed_characters is None:
            try:
                parsed_characters = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(
                    f"Brief characters must be valid JSON: {exc.msg}", param_hint="--brief-characters"
                ) from exc
            if not isinstance(parsed_characters, list):
                raise typer.BadParameter("Brief characters JSON must be an array.", param_hint="--brief-characters")

    try:
        return TranslationBriefInput(summary=summary, characters=parsed_characters, tone_style=tone_style)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid Translation Brief: {exc}") from exc


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
    required: bool = True,
) -> ContextLLMConfig | None:
    if mode is HyMT2Mode.FAST:
        return None
    if provider is None and not model and not required:
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


def _workflow_translation_for(
    *,
    translation: TranslationBackend | TranslationOnlyBackend,
    llama_model: str,
    local_model_profile: LocalModelProfile | None,
    llama_port: int,
    idle_timeout: int,
    hy_mt2_mode: HyMT2Mode,
    context_provider: ContextProvider | None,
    context_model: str | None,
    context_base_url: str | None,
    context_fee_limit: float,
    context_assistance: ContextAssistance = ContextAssistance.AUTO,
    glossary: Path | None = None,
    force_glossary: bool = False,
    glossary_strict: bool = True,
    edit_rounds: int = 1,
    enable_restore: bool = False,
    translation_brief: TranslationBriefInput | None = None,
    semantic_editor: bool = True,
) -> WorkflowTranslationConfig:
    """Build the CLI's canonical translation request through the shared factory."""
    hy_mt2_mode = normalize_hymt2_mode(hy_mt2_mode)
    context_assistance = ContextAssistance(context_assistance)
    glossary_value, glossary_options, edit_config = _glossary_and_edit_options(
        glossary=glossary,
        force_glossary=force_glossary,
        glossary_strict=glossary_strict,
        edit_rounds=edit_rounds,
        enable_restore=enable_restore,
    )
    edit_config.semantic_review = bool(semantic_editor and edit_rounds > 0)
    if translation.value == TranslationBackend.local.value:
        profile_name, selected_model = _selected_profile_and_model(
            local_model_profile=local_model_profile, llama_model=llama_model
        )
        if profile_name != QWEN35_9B_PROFILE:
            if hy_mt2_mode is HyMT2Mode.FAST and translation_brief is not None:
                raise typer.BadParameter("Translation Brief is not supported in Hy-MT2 fast mode.")
            if context_assistance is ContextAssistance.OFF and any(
                (context_provider is not None, context_model, context_base_url)
            ):
                raise typer.BadParameter(
                    "Context assistance off cannot be combined with Context provider/model options.",
                    param_hint="--context-assistance",
                )
            try:
                requires_context = context_model_required(
                    mode=hy_mt2_mode,
                    translation_brief=translation_brief,
                    edit_config=edit_config,
                    context_assistance=context_assistance,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--context-assistance") from exc
            context_llm = _context_llm_config(
                mode=hy_mt2_mode,
                provider=context_provider,
                model=context_model,
                base_url=context_base_url,
                fee_limit=context_fee_limit,
                port=llama_port,
                required=requires_context,
            )
            return WorkflowTranslationFactory.hymt2(
                mode=TranslationMode(hy_mt2_mode.value),
                profile=profile_name,
                model=selected_model,
                idle_timeout=idle_timeout,
                port=llama_port,
                context_llm=context_llm,
                context_assistance=context_assistance,
                glossary=glossary_value,
                glossary_options=glossary_options,
                edit_config=edit_config,
                translation_brief=translation_brief,
            )
        if translation_brief is not None:
            raise typer.BadParameter("Translation Brief requires a Hy-MT2 local model profile.")
        if (
            hy_mt2_mode is not HyMT2Mode.FAST
            or context_provider is not None
            or context_model is not None
            or context_assistance is not ContextAssistance.AUTO
        ):
            raise typer.BadParameter("Hy-MT2 context options require a Hy-MT2 local model profile.")
        return WorkflowTranslationFactory.standard_local_qwen(
            model=selected_model or llama_model,
            idle_timeout=idle_timeout,
            port=llama_port,
            glossary=glossary_value,
            glossary_options=glossary_options,
            edit_config=edit_config,
        )
    if translation_brief is not None:
        raise typer.BadParameter("Translation Brief requires a Hy-MT2 local translation backend.")
    if context_assistance is not ContextAssistance.AUTO:
        raise typer.BadParameter("Context assistance requires a Hy-MT2 local translation backend.")
    return WorkflowTranslationFactory.standard_online(
        glossary=glossary_value, glossary_options=glossary_options, edit_config=edit_config
    )


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
    context_assistance: ContextAssistance = ContextAssistance.AUTO,
    subtitle_optimization: SubtitleOptimizationMode,
    glossary: Path | None = None,
    force_glossary: bool = False,
    glossary_strict: bool = True,
    edit_rounds: int = 1,
    enable_restore: bool = False,
    translation_brief: TranslationBriefInput | None = None,
    semantic_editor: bool = True,
) -> LRCer:
    lrcer_cls = _lrcer_cls()
    workflow_translation = _workflow_translation_for(
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
        context_assistance=context_assistance,
        glossary=glossary,
        force_glossary=force_glossary,
        glossary_strict=glossary_strict,
        edit_rounds=edit_rounds,
        enable_restore=enable_restore,
        translation_brief=translation_brief,
        semantic_editor=semantic_editor,
    )
    return lrcer_cls(translation=workflow_translation.config, subtitle_optimization=subtitle_optimization)


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
                unresolved = status.get("unresolved_issues", 0)
                detail = (
                    f"{len(failed)} chunk(s) kept as Hy-MT2 draft" if failed else f"{unresolved} unresolved issue(s)"
                )
                status_text = f"[yellow]incomplete ({detail})[/yellow]"
            else:
                status_text = "[green]complete[/green]"
        row = [str(index), str(output)]
        if review_statuses:
            row.append(status_text)
        table.add_row(*row)
    console.print(table)
    for name, status in review_statuses.items():
        if status.get("incomplete"):
            failed = status.get("failed_chunks", [])
            unresolved = status.get("unresolved_issues", 0)
            if failed:
                detail = f"chunks {failed} kept their Hy-MT2 draft"
            else:
                detail = f"{unresolved} unresolved issue(s) remain"
            console.print(
                f"[yellow]Editing incomplete for {name}: {detail}. "
                "Temporary checkpoint retained for the next run.[/yellow]"
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


@glossary_app.command("validate")
def glossary_validate(
    path: Annotated[Path, typer.Argument(help="Glossary JSON path.")],
    source_language: Annotated[
        str | None, typer.Option("--source-language", help="Expected source language code.")
    ] = None,
    target_language: Annotated[
        str | None, typer.Option("--target-language", help="Expected target language code.")
    ] = None,
    strict: Annotated[bool, typer.Option("--strict/--no-strict", help="Reject same-priority conflicts.")] = True,
) -> None:
    """Validate glossary JSON, schema version, languages, and conflicts."""
    from openlrc.glossary import GlossaryService

    service = GlossaryService(strict=strict)
    catalog, conflicts = service.load(path, source_language=source_language, target_language=target_language)
    console.print(
        f"[green]Valid glossary[/green]: {len(catalog.entries)} entries, {len(conflicts)} reported conflicts."
    )


@glossary_app.command("inspect")
def glossary_inspect(
    path: Annotated[Path, typer.Argument(help="Glossary JSON path.")],
    strict: Annotated[bool, typer.Option("--strict/--no-strict", help="Reject same-priority conflicts.")] = True,
    force: Annotated[bool, typer.Option("--force", help="Show enabled task entries as required.")] = False,
) -> None:
    """Show normalized glossary entries, priority, and conflicts."""
    from openlrc.glossary import GlossaryService

    service = GlossaryService(strict=strict, force=force)
    catalog, conflicts = service.load(path)
    state = service.merge(catalog, load_conflicts=conflicts)
    table = Table(title=f"Glossary: {catalog.name or path.name}")
    table.add_column("Source")
    table.add_column("Target")
    table.add_column("Required")
    table.add_column("Aliases")
    for entry in state.merged_entries:
        table.add_row(entry.source, entry.target, "yes" if entry.required else "no", ", ".join(entry.aliases))
    console.print(table)
    console.print(f"Fingerprint: {state.fingerprint}")
    if state.conflicts:
        console.print(f"[yellow]Conflicts: {len(state.conflicts)} (deterministic first-item precedence)[/yellow]")


@glossary_app.command("check")
def glossary_check(
    path: Annotated[Path, typer.Argument(help="Glossary JSON path.")],
    source_path: Annotated[Path, typer.Option("--source", help="Source subtitle JSON.")],
    target_path: Annotated[Path, typer.Option("--target", help="Translated subtitle JSON.")],
    output: Annotated[Path | None, typer.Option("--output", help="Compliance report JSON path.")] = None,
    strict: Annotated[bool, typer.Option("--strict/--no-strict", help="Reject same-priority conflicts.")] = True,
    force: Annotated[bool, typer.Option("--force", help="Treat enabled task entries as required.")] = False,
) -> None:
    """Check glossary compliance without loading a model."""
    from openlrc.checkpoint import save_json_checkpoint
    from openlrc.glossary import GlossaryService
    from openlrc.subtitle import Subtitle

    source = Subtitle.from_json(source_path)
    target = Subtitle.from_json(target_path)
    service = GlossaryService(strict=strict, force=force)
    catalog, conflicts = service.load(path, source_language=source.lang, target_language=target.lang)
    state = service.merge(catalog, load_conflicts=conflicts)
    checked = service.check(state, source.texts, target.texts)
    report_path = output or target_path.with_name(f"{target_path.stem}.glossary-report.json")
    save_json_checkpoint(report_path, checked.model_dump(mode="json"))
    console.print(
        f"Glossary compliance: {checked.metrics.required_compliant} required compliant, "
        f"{checked.metrics.required_noncompliant} required unresolved. Report: {report_path}"
    )
    if checked.metrics.required_noncompliant:
        raise typer.Exit(code=1)


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
    whisper_gpu: Annotated[
        bool,
        typer.Option(
            "--whisper-gpu/--no-whisper-gpu", help="Enable Whisper GPU acceleration; disable for an explicit CPU run."
        ),
    ] = True,
    whisper_flash_attn: Annotated[
        bool,
        typer.Option(
            "--whisper-flash-attn/--no-whisper-flash-attn",
            help="Enable Whisper flash attention; disable for compatibility diagnostics.",
        ),
    ] = True,
    skip_preprocess: Annotated[
        bool, typer.Option("--skip-preprocess", help="Use existing preprocessed audio files.")
    ] = False,
) -> None:
    """Transcribe audio/video files and write transcription JSON."""
    request = TranscribeRequest(
        paths=tuple(paths),
        transcription=_transcription_config(
            whisper_model, vad_model, whisper_gpu=whisper_gpu, whisper_flash_attn=whisper_flash_attn
        ),
        src_lang=src_lang,
        skip_preprocess=skip_preprocess,
    )
    result = _execute_workflow(request, WorkflowKind.TRANSCRIBE)
    _print_outputs(list(result.outputs))


@app.command()
def translate(
    json_paths: Annotated[list[Path], typer.Argument(help="Transcribed JSON files.")],
    translation: Annotated[TranslationOnlyBackend, typer.Option("--translation", help="Translation backend to use.")],
    target_lang: Annotated[str, typer.Option("--target-lang", help="Target language code.")] = "zh-cn",
    bilingual_sub: Annotated[bool, typer.Option("--bilingual-sub", help="Generate bilingual subtitle files.")] = False,
    subtitle_optimization: Annotated[
        SubtitleOptimizationMode, typer.Option("--subtitle-optimization", help="Subtitle cleanup profile.")
    ] = SubtitleOptimizationMode.AGGRESSIVE,
    keep_checkpoint: Annotated[
        bool, typer.Option("--keep-checkpoint", help="Keep a completed translation checkpoint for debugging.")
    ] = False,
    glossary: Annotated[Path | None, typer.Option("--glossary", help="Task glossary JSON path.")] = None,
    force_glossary: Annotated[
        bool, typer.Option("--force-glossary", help="Treat enabled task glossary entries as required.")
    ] = False,
    glossary_strict: Annotated[
        bool,
        typer.Option(
            "--glossary-strict/--no-glossary-strict",
            help="Reject same-priority glossary conflicts instead of reporting first-item precedence.",
        ),
    ] = True,
    edit_rounds: Annotated[
        int, typer.Option("--edit-rounds", min=0, max=3, help="Semantic edit rounds; 0 runs deterministic checks only.")
    ] = 1,
    enable_restore: Annotated[
        bool, typer.Option("--enable-restore", help="Keep a compact edit session after successful translation.")
    ] = False,
    brief_summary: Annotated[
        str | None, typer.Option("--brief-summary", help="Human-supplied global Translation Brief summary.")
    ] = None,
    brief_characters: Annotated[
        str | None,
        typer.Option("--brief-characters", help="Strict character JSON array, or @PATH to a UTF-8 JSON file."),
    ] = None,
    brief_tone_style: Annotated[
        str | None, typer.Option("--brief-tone-style", help="Human-supplied global tone and style guidance.")
    ] = None,
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
    context_assistance: Annotated[
        ContextAssistance,
        typer.Option("--context-assistance", help="Use a Context model automatically, or require a manual Brief."),
    ] = ContextAssistance.AUTO,
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
    translation_brief = _translation_brief_input(
        summary=brief_summary,
        characters=brief_characters,
        tone_style=brief_tone_style,
        context_assistance=context_assistance,
        mode=hy_mt2_mode,
    )
    workflow_translation = _workflow_translation_for(
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
        context_assistance=context_assistance,
        glossary=glossary,
        force_glossary=force_glossary,
        glossary_strict=glossary_strict,
        edit_rounds=edit_rounds,
        enable_restore=enable_restore,
        translation_brief=translation_brief,
    )
    request = TranslateRequest(
        transcribed_paths=tuple(json_paths),
        translation=workflow_translation,
        target_lang=target_lang,
        bilingual_sub=bilingual_sub,
        subtitle_optimization=subtitle_optimization,
        clear_checkpoint=not keep_checkpoint,
    )
    result = _execute_workflow(request, WorkflowKind.TRANSLATE)
    _print_outputs(list(result.outputs), _review_status_mapping(result))


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
    whisper_gpu: Annotated[
        bool,
        typer.Option(
            "--whisper-gpu/--no-whisper-gpu", help="Enable Whisper GPU acceleration; disable for an explicit CPU run."
        ),
    ] = True,
    whisper_flash_attn: Annotated[
        bool,
        typer.Option(
            "--whisper-flash-attn/--no-whisper-flash-attn",
            help="Enable Whisper flash attention; disable for compatibility diagnostics.",
        ),
    ] = True,
    bilingual_sub: Annotated[bool, typer.Option("--bilingual-sub", help="Generate bilingual subtitle files.")] = False,
    subtitle_optimization: Annotated[
        SubtitleOptimizationMode, typer.Option("--subtitle-optimization", help="Subtitle cleanup profile.")
    ] = SubtitleOptimizationMode.AGGRESSIVE,
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
    glossary: Annotated[Path | None, typer.Option("--glossary", help="Task glossary JSON path.")] = None,
    force_glossary: Annotated[
        bool, typer.Option("--force-glossary", help="Treat enabled task glossary entries as required.")
    ] = False,
    glossary_strict: Annotated[
        bool,
        typer.Option(
            "--glossary-strict/--no-glossary-strict",
            help="Reject same-priority glossary conflicts instead of reporting first-item precedence.",
        ),
    ] = True,
    edit_rounds: Annotated[
        int, typer.Option("--edit-rounds", min=0, max=3, help="Semantic edit rounds; 0 runs deterministic checks only.")
    ] = 1,
    enable_restore: Annotated[
        bool, typer.Option("--enable-restore", help="Keep a compact edit session after successful translation.")
    ] = False,
    brief_summary: Annotated[
        str | None, typer.Option("--brief-summary", help="Human-supplied global Translation Brief summary.")
    ] = None,
    brief_characters: Annotated[
        str | None,
        typer.Option("--brief-characters", help="Strict character JSON array, or @PATH to a UTF-8 JSON file."),
    ] = None,
    brief_tone_style: Annotated[
        str | None, typer.Option("--brief-tone-style", help="Human-supplied global tone and style guidance.")
    ] = None,
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
    context_assistance: Annotated[
        ContextAssistance,
        typer.Option("--context-assistance", help="Use a Context model automatically, or require a manual Brief."),
    ] = ContextAssistance.AUTO,
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
    translation_brief = _translation_brief_input(
        summary=brief_summary,
        characters=brief_characters,
        tone_style=brief_tone_style,
        context_assistance=context_assistance,
        mode=hy_mt2_mode,
    )
    transcription_config = _transcription_config(
        whisper_model, vad_model, whisper_gpu=whisper_gpu, whisper_flash_attn=whisper_flash_attn
    )
    workflow_translation = None
    if translation is not TranslationBackend.none:
        workflow_translation = _workflow_translation_for(
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
            context_assistance=context_assistance,
            glossary=glossary,
            force_glossary=force_glossary,
            glossary_strict=glossary_strict,
            edit_rounds=edit_rounds,
            enable_restore=enable_restore,
            translation_brief=translation_brief,
        )
    request = RunRequest(
        paths=tuple(paths),
        transcription=transcription_config,
        translation=workflow_translation,
        src_lang=src_lang,
        target_lang=target_lang,
        bilingual_sub=bilingual_sub,
        subtitle_optimization=subtitle_optimization,
        clear_temp=clear_temp,
        skip_preprocess=skip_preprocess,
    )
    result = _execute_workflow(request, WorkflowKind.RUN)
    _print_outputs(list(result.outputs), _review_status_mapping(result))


@app.command("edit")
def edit_command(
    source_path: Annotated[Path, typer.Option("--source", help="Source subtitle JSON.")],
    target_path: Annotated[Path, typer.Option("--target", help="Current translated subtitle JSON.")],
    action: Annotated[EditAction, typer.Option("--action", help="Independent edit action.")],
    ids: Annotated[str | None, typer.Option("--ids", help="Subtitle IDs/ranges, for example 12,18-24.")] = None,
    restore_round: Annotated[int | None, typer.Option("--round", min=0, help="Historical round to restore.")] = None,
    session: Annotated[Path | None, typer.Option("--session", help="Edit session JSON path.")] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Output translation JSON path.")] = None,
    markdown_report: Annotated[
        bool, typer.Option("--markdown-report", help="Write the edit report as Markdown.")
    ] = False,
    glossary: Annotated[Path | None, typer.Option("--glossary", help="Task glossary JSON path.")] = None,
    force_glossary: Annotated[
        bool, typer.Option("--force-glossary", help="Treat enabled task glossary entries as required.")
    ] = False,
    glossary_strict: Annotated[
        bool, typer.Option("--glossary-strict/--no-glossary-strict", help="Glossary conflict policy.")
    ] = True,
    brief_summary: Annotated[
        str | None, typer.Option("--brief-summary", help="Human-supplied global Translation Brief summary.")
    ] = None,
    brief_characters: Annotated[
        str | None,
        typer.Option("--brief-characters", help="Strict character JSON array, or @PATH to a UTF-8 JSON file."),
    ] = None,
    brief_tone_style: Annotated[
        str | None, typer.Option("--brief-tone-style", help="Human-supplied global tone and style guidance.")
    ] = None,
    llama_model: Annotated[
        str | None, typer.Option("--llama-model", help="Explicit Hy-MT2 GGUF alias, filename, or path.")
    ] = None,
    local_model_profile: Annotated[
        LocalModelProfile | None, typer.Option("--local-model-profile", help="Explicit Hy-MT2 model profile.")
    ] = None,
    llama_port: Annotated[int, typer.Option("--llama-port", help="Local llama-server port.")] = DEFAULT_LLAMA_PORT,
    hy_mt2_mode: Annotated[HyMT2Mode, typer.Option("--hy-mt2-mode", help="Hy-MT2 edit mode.")] = HyMT2Mode.FAST,
    context_assistance: Annotated[
        ContextAssistance,
        typer.Option("--context-assistance", help="Use a Context model automatically, or require a manual Brief."),
    ] = ContextAssistance.AUTO,
    context_provider: Annotated[
        ContextProvider | None, typer.Option("--context-provider", help="Explicit context-model provider.")
    ] = None,
    context_model: Annotated[
        str | None, typer.Option("--context-model", help="Explicit context-model name or local model path.")
    ] = None,
    context_base_url: Annotated[
        str | None, typer.Option("--context-base-url", help="Custom context-model endpoint.")
    ] = None,
    context_fee_limit: Annotated[
        float, typer.Option("--context-fee-limit", help="Maximum context-model fee per call.")
    ] = 0.8,
) -> None:
    """Verify, review, retranslate, or restore existing subtitles without ASR."""
    if action is not EditAction.RETRANSLATE and context_assistance is ContextAssistance.OFF:
        raise typer.BadParameter(
            "--context-assistance off is only supported by edit retranslate.", param_hint="--context-assistance"
        )
    translation_brief = _translation_brief_input(
        summary=brief_summary,
        characters=brief_characters,
        tone_style=brief_tone_style,
        context_assistance=context_assistance,
        mode=hy_mt2_mode,
    )
    if action in {EditAction.VERIFY, EditAction.RESTORE} and translation_brief is not None:
        raise typer.BadParameter(f"Translation Brief is not supported by edit {action.value}.")
    selected_ids = None
    if action in {EditAction.REVIEW, EditAction.RETRANSLATE}:
        if ids is None:
            raise typer.BadParameter("review and retranslate require --ids.")
        selected_ids = parse_segment_ids(ids)
    elif ids is not None:
        raise typer.BadParameter("--ids is only valid for review and retranslate.")

    glossary_value, glossary_options, edit_config = _glossary_and_edit_options(
        glossary=glossary,
        force_glossary=force_glossary,
        glossary_strict=glossary_strict,
        edit_rounds=1,
        enable_restore=True,
    )
    lrcer_cls = _lrcer_cls()
    if action in {EditAction.VERIFY, EditAction.RESTORE}:
        lrcer = lrcer_cls(
            translation=TranslationConfig(
                glossary=glossary_value,
                glossary_options=glossary_options,
                edit_config=edit_config,
                translation_brief=translation_brief,
            )
        )
    elif action is EditAction.REVIEW:
        context_llm = _context_llm_config(
            mode=HyMT2Mode.NORMAL,
            provider=context_provider,
            model=context_model,
            base_url=context_base_url,
            fee_limit=context_fee_limit,
            port=llama_port,
        )
        lrcer = lrcer_cls(
            translation=TranslationConfig(
                context_llm=context_llm,
                glossary=glossary_value,
                glossary_options=glossary_options,
                edit_config=edit_config,
                translation_brief=translation_brief,
            )
        )
    else:
        if llama_model is None and local_model_profile is None:
            raise typer.BadParameter("retranslate requires --llama-model or --local-model-profile for Hy-MT2.")
        selected_profile = local_model_profile
        inferred = infer_local_llm_profile(llama_model or "")
        if selected_profile is None and inferred == QWEN35_9B_PROFILE:
            raise typer.BadParameter("retranslate requires a Hy-MT2 model, not bare Qwen.")
        lrcer = _lrcer_for_translation(
            translation=TranslationOnlyBackend.local,
            llama_model=llama_model or QWEN35_9B_PROFILE,
            local_model_profile=selected_profile,
            llama_port=llama_port,
            idle_timeout=0,
            hy_mt2_mode=hy_mt2_mode,
            context_provider=context_provider,
            context_model=context_model,
            context_base_url=context_base_url,
            context_fee_limit=context_fee_limit,
            context_assistance=context_assistance,
            subtitle_optimization=SubtitleOptimizationMode.RELAXED,
            glossary=glossary,
            force_glossary=force_glossary,
            glossary_strict=glossary_strict,
            edit_rounds=0,
            enable_restore=True,
            translation_brief=translation_brief,
            semantic_editor=False,
        )
    try:
        result = lrcer.edit(
            source_path,
            target_path,
            action=action.value,
            segment_ids=selected_ids,
            restore_round=restore_round,
            session_path=session,
            output_path=output,
            markdown_report=markdown_report,
        )
        console.print(
            f"Edit {action.value}: {len(result.changed_ids)} changed IDs, "
            f"{len(result.unresolved_issues)} unresolved issues. Report: {result.report_path}"
        )
        if action is EditAction.VERIFY and any(
            issue.severity is EditSeverity.ERROR for issue in result.unresolved_issues
        ):
            raise typer.Exit(code=1)
    finally:
        lrcer.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
