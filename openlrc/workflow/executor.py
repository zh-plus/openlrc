"""Synchronous in-process execution of OpenLRC workflow requests."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from openlrc.config import ContextAssistance, HyMT2Mode, context_model_required, normalize_hymt2_mode
from openlrc.defaults import (
    COMPARE_SUFFIX,
    EDIT_REPORT_SUFFIX,
    EDIT_SESSION_SUFFIX,
    PREPROCESSED_DIR,
    PREPROCESSED_SUFFIX,
    TRANSCRIBED_SUFFIX,
)
from openlrc.exceptions import ChatBotException, DependencyException, FfmpegException, TranscribeException
from openlrc.llama_resources import HY_MT2_PROMPT_PROFILE
from openlrc.models import ModelProvider
from openlrc.openlrc import LRCer
from openlrc.workflow.configuration import resolve_run_execution_strategy
from openlrc.workflow.runtime import ExecutionContext, WorkflowCancelled
from openlrc.workflow.security import redact_sensitive_text
from openlrc.workflow.types import (
    ArtifactKind,
    ErrorCategory,
    ReviewStatus,
    RunExecutionStrategy,
    RunRequest,
    TranscribeRequest,
    TranslateRequest,
    TranslationMode,
    WorkflowArtifact,
    WorkflowError,
    WorkflowKind,
    WorkflowRequest,
    WorkflowResult,
    WorkflowStage,
    WorkflowStatus,
    WorkflowTranslationConfig,
)


class WorkflowExecutor:
    """Run one workflow at a time using a freshly owned :class:`LRCer`."""

    def __init__(self) -> None:
        self._active_lock = threading.Lock()

    def execute(self, request: WorkflowRequest, context: ExecutionContext | None = None) -> WorkflowResult:
        workflow = _workflow_kind(request)
        if not self._active_lock.acquire(blocking=False):
            raise RuntimeError("This WorkflowExecutor is already executing a workflow.")

        if context is not None and context.workflow is not workflow:
            self._active_lock.release()
            raise ValueError(
                f"ExecutionContext workflow {context.workflow.value!r} does not match request {workflow.value!r}."
            )
        context = context or ExecutionContext(workflow)
        translation_mode: TranslationMode | None = None
        started_at = time.perf_counter()
        lrcer: LRCer | None = None
        outputs: tuple[Path, ...] = ()
        reviews: tuple[ReviewStatus, ...] = ()
        api_fee = 0.0
        cancellation_message = "Workflow cancelled."
        result: WorkflowResult | None = None
        try:
            context.workflow_started()
        except BaseException:
            self._active_lock.release()
            raise

        try:
            with context.stage(WorkflowStage.VALIDATE):
                translation_mode = _translation_mode(request)
                context.translation_mode = translation_mode
                self._validate_request(request)
            context.check_cancelled()
            lrcer = self._build_lrcer(request, context)
            outputs = tuple(Path(path) for path in self._dispatch(lrcer, request))
            api_fee = float(lrcer.api_fee)
            _discover_recovery_artifacts(request, context)
            reviews = _review_statuses(lrcer.review_statuses)
            for review in reviews:
                for path, kind in (
                    (review.checkpoint, ArtifactKind.CHECKPOINT),
                    (review.report, ArtifactKind.REVIEW_REPORT),
                    (review.session, ArtifactKind.EDIT_SESSION),
                ):
                    if path is not None:
                        context.artifact_created(WorkflowArtifact(path=path, kind=kind, item=review.item))
            for output in outputs:
                if not any(artifact.path == output and artifact.primary for artifact in context.artifacts):
                    context.artifact_created(
                        WorkflowArtifact(path=output, kind=_primary_artifact_kind(workflow, output), primary=True)
                    )
            incomplete = any(review.incomplete for review in reviews)
            status = WorkflowStatus.SUCCEEDED_WITH_WARNINGS if incomplete else WorkflowStatus.SUCCEEDED
            result = WorkflowResult(
                job_id=context.job_id,
                workflow=workflow,
                status=status,
                translation_mode=translation_mode,
                outputs=outputs,
                artifacts=context.artifacts,
                reviews=reviews,
                api_fee=api_fee,
            )
        except (WorkflowCancelled, KeyboardInterrupt) as exc:
            context.cancellation_token.cancel()
            cancellation_message = redact_sensitive_text(str(exc) or cancellation_message)
            _discover_recovery_artifacts(request, context)
            outputs = _partial_outputs(context)
            if lrcer is not None:
                reviews = _review_statuses(lrcer.review_statuses)
                api_fee = float(lrcer.api_fee)
            result = WorkflowResult(
                job_id=context.job_id,
                workflow=workflow,
                status=WorkflowStatus.CANCELLED,
                translation_mode=translation_mode,
                outputs=outputs,
                artifacts=context.artifacts,
                reviews=reviews,
                api_fee=api_fee,
            )
        except Exception as exc:
            _discover_recovery_artifacts(request, context)
            outputs = _partial_outputs(context)
            if lrcer is not None:
                reviews = _review_statuses(lrcer.review_statuses)
                api_fee = float(lrcer.api_fee)
            if context.cancellation_token.is_cancelled:
                result = WorkflowResult(
                    job_id=context.job_id,
                    workflow=workflow,
                    status=WorkflowStatus.CANCELLED,
                    translation_mode=translation_mode,
                    outputs=outputs,
                    artifacts=context.artifacts,
                    reviews=reviews,
                    api_fee=api_fee,
                )
            else:
                error = _map_error(exc, context)
                result = WorkflowResult(
                    job_id=context.job_id,
                    workflow=workflow,
                    status=WorkflowStatus.FAILED,
                    translation_mode=translation_mode,
                    outputs=outputs,
                    artifacts=context.artifacts,
                    reviews=reviews,
                    api_fee=api_fee,
                    error=error,
                )
        finally:
            if lrcer is not None:
                try:
                    lrcer.close()
                except Exception as exc:
                    detail = redact_sensitive_text(str(exc) or type(exc).__name__)
                    context.log("warning", f"Resource cleanup failed: {type(exc).__name__}: {detail}")
            try:
                context.close()
            except Exception as exc:
                detail = redact_sensitive_text(str(exc) or type(exc).__name__)
                context.log("warning", f"Process cleanup failed: {type(exc).__name__}: {detail}")
            finally:
                elapsed = time.perf_counter() - started_at
                if result is None:
                    error = WorkflowError(
                        category=ErrorCategory.INTERNAL,
                        message="Workflow ended without a result.",
                        exception_type="InternalWorkflowState",
                    )
                    result = WorkflowResult(
                        job_id=context.job_id,
                        workflow=workflow,
                        status=WorkflowStatus.FAILED,
                        translation_mode=translation_mode,
                        artifacts=_retained_artifacts(context),
                        elapsed_seconds=elapsed,
                        error=error,
                    )
                else:
                    result = replace(result, artifacts=_retained_artifacts(context), elapsed_seconds=elapsed)

                if result.status is WorkflowStatus.CANCELLED:
                    context.workflow_cancelled(cancellation_message)
                elif result.status is WorkflowStatus.FAILED:
                    assert result.error is not None
                    context.workflow_failed(result.error)
                else:
                    context.workflow_completed(result.status, elapsed)
                self._active_lock.release()

        return result

    @staticmethod
    def _validate_request(request: WorkflowRequest) -> None:
        paths = request.transcribed_paths if isinstance(request, TranslateRequest) else request.paths
        if not paths:
            raise ValueError("At least one input path is required.")
        resolved_paths = [Path(path) for path in paths]
        missing = [str(path) for path in resolved_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Input file not found: " + ", ".join(missing))
        if isinstance(request, TranslateRequest):
            _validate_translation_config(request.translation)
        elif isinstance(request, RunRequest) and request.translation is not None:
            _validate_translation_config(request.translation)

    @staticmethod
    def _build_lrcer(request: WorkflowRequest, context: ExecutionContext) -> LRCer:
        if isinstance(request, TranscribeRequest):
            return LRCer(
                transcription=deepcopy(request.transcription),
                subtitle_optimization=request.subtitle_optimization,
                execution_context=context,
            )
        if isinstance(request, TranslateRequest):
            return LRCer(
                translation=deepcopy(request.translation.config),
                subtitle_optimization=request.subtitle_optimization,
                execution_context=context,
            )
        translation = deepcopy(request.translation.config) if request.translation is not None else None
        return LRCer(
            transcription=deepcopy(request.transcription),
            translation=translation,
            subtitle_optimization=request.subtitle_optimization,
            execution_context=context,
        )

    @staticmethod
    def _dispatch(lrcer: LRCer, request: WorkflowRequest):
        if isinstance(request, TranscribeRequest):
            if request.subtitle_output:
                return lrcer.run(
                    list(request.paths),
                    src_lang=request.src_lang,
                    skip_trans=True,
                    noise_suppress=request.noise_suppress,
                    clear_temp=request.clear_temp,
                    skip_preprocess=request.skip_preprocess,
                    execution_strategy=RunExecutionStrategy.TRANSCRIBE_ONLY,
                )
            return lrcer.transcribe(
                list(request.paths),
                src_lang=request.src_lang,
                noise_suppress=request.noise_suppress,
                skip_preprocess=request.skip_preprocess,
            )
        if isinstance(request, TranslateRequest):
            return lrcer.translate(
                [Path(path) for path in request.transcribed_paths],
                target_lang=request.target_lang,
                bilingual_sub=request.bilingual_sub,
                clear_checkpoint=request.clear_checkpoint,
            )
        return lrcer.run(
            list(request.paths),
            src_lang=request.src_lang,
            target_lang=request.target_lang,
            skip_trans=request.translation is None,
            noise_suppress=request.noise_suppress,
            bilingual_sub=request.bilingual_sub,
            clear_temp=request.clear_temp,
            clear_checkpoint=request.clear_checkpoint,
            skip_preprocess=request.skip_preprocess,
            execution_strategy=resolve_run_execution_strategy(request),
        )


def _workflow_kind(request: WorkflowRequest) -> WorkflowKind:
    if isinstance(request, TranscribeRequest):
        return WorkflowKind.TRANSCRIBE
    if isinstance(request, TranslateRequest):
        return WorkflowKind.TRANSLATE
    if isinstance(request, RunRequest):
        return WorkflowKind.RUN
    raise TypeError(f"Unsupported workflow request: {type(request).__name__}")


def _translation_mode(request: WorkflowRequest) -> TranslationMode | None:
    if isinstance(request, TranslateRequest):
        return TranslationMode(request.translation.mode)
    if isinstance(request, RunRequest) and request.translation is not None:
        return TranslationMode(request.translation.mode)
    return None


def _validate_translation_config(workflow_config: WorkflowTranslationConfig) -> None:
    mode = TranslationMode(workflow_config.mode)
    config = workflow_config.config
    is_hymt2 = config.prompt_profile == HY_MT2_PROMPT_PROFILE or config._translator_engine == "lean"
    if config.consumer_thread < 1:
        raise ValueError("Translation consumer_thread must be at least 1.")
    if config.local_llm is not None and config.local_llm.enabled and config.consumer_thread != 1:
        raise ValueError("Managed local translation profiles require consumer_thread=1.")

    if mode is TranslationMode.STANDARD:
        if is_hymt2:
            raise ValueError("Standard mode requires the classic translation pipeline.")
        if config.chatbot is not None and config.chatbot.provider is ModelProvider.LOCAL_LLAMA:
            if config.local_llm is None or not config.local_llm.enabled:
                raise ValueError("Managed local Standard mode requires an enabled LocalLLMConfig.")
        return

    expected = {
        TranslationMode.FAST: HyMT2Mode.FAST,
        TranslationMode.NORMAL: HyMT2Mode.NORMAL,
        TranslationMode.NORMAL_PLUS: HyMT2Mode.NORMAL_PLUS,
        TranslationMode.PRO: HyMT2Mode.PRO,
    }[mode]
    actual = normalize_hymt2_mode(config.hy_mt2_mode)
    if not is_hymt2 or config._translator_engine != "lean":
        raise ValueError(f"{mode.value} mode requires the managed local Hy-MT2 pipeline.")
    if actual is not expected:
        raise ValueError(f"Workflow mode {mode.value!r} does not match TranslationConfig mode {actual.value!r}.")
    if config.local_llm is None or not config.local_llm.enabled:
        raise ValueError(f"{mode.value} mode requires an enabled managed local Hy-MT2 server.")
    if config.chatbot is None or config.chatbot.provider is not ModelProvider.LOCAL_LLAMA:
        raise ValueError("Hy-MT2 must be the local primary translation model.")

    brief = config.translation_brief
    if expected is HyMT2Mode.FAST and brief is not None:
        raise ValueError("Fast mode does not use a Translation Brief.")
    assistance = ContextAssistance(config.context_assistance)
    needs_context = context_model_required(
        mode=expected, translation_brief=brief, edit_config=config.edit_config, context_assistance=assistance
    )
    if assistance is ContextAssistance.OFF and config.context_llm is not None:
        raise ValueError("Context assistance off conflicts with an explicit context model.")
    if needs_context and config.context_llm is None:
        raise ValueError(f"{mode.value} mode requires a context model for this configuration.")


def _review_statuses(statuses: dict[str, dict]) -> tuple[ReviewStatus, ...]:
    result: list[ReviewStatus] = []
    for item, raw in statuses.items():
        workflow_item = str(raw.get("_workflow_item") or item)
        checkpoint = raw.get("checkpoint") or raw.get("checkpoint_path")
        report = raw.get("report") or raw.get("report_path")
        session = raw.get("session") or raw.get("session_path")
        checkpoint_path = Path(checkpoint) if checkpoint else None
        report_path = Path(report) if report else None
        session_path = Path(session) if session else None
        result.append(
            ReviewStatus(
                item=workflow_item,
                incomplete=bool(raw.get("incomplete")),
                checkpoint=checkpoint_path if checkpoint_path is not None and checkpoint_path.exists() else None,
                report=report_path if report_path is not None and report_path.exists() else None,
                session=session_path if session_path is not None and session_path.exists() else None,
                details={
                    key: value
                    for key, value in raw.items()
                    if key
                    not in {
                        "_workflow_item",
                        "checkpoint",
                        "checkpoint_path",
                        "report",
                        "report_path",
                        "session",
                        "session_path",
                    }
                    and _is_safe_detail(value)
                },
            )
        )
    return tuple(result)


def _is_safe_detail(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_safe_detail(item) for item in value)
    return False


def _primary_artifact_kind(workflow: WorkflowKind, output: Path) -> ArtifactKind:
    if workflow is WorkflowKind.TRANSCRIBE and output.suffix.lower() == ".json":
        return ArtifactKind.TRANSCRIPTION
    return ArtifactKind.SUBTITLE


def _partial_outputs(context: ExecutionContext) -> tuple[Path, ...]:
    return tuple(artifact.path for artifact in _retained_artifacts(context) if artifact.primary)


def _retained_artifacts(context: ExecutionContext) -> tuple[WorkflowArtifact, ...]:
    return tuple(artifact for artifact in context.artifacts if artifact.path.exists())


def _discover_recovery_artifacts(request: WorkflowRequest, context: ExecutionContext) -> None:
    """Record exact, known recovery files that currently exist on disk."""
    candidates: list[tuple[Path, ArtifactKind, str]] = []
    if isinstance(request, TranslateRequest):
        for raw_path in request.transcribed_paths:
            path = Path(raw_path)
            base_name = path.stem.replace(f"{PREPROCESSED_SUFFIX}{TRANSCRIBED_SUFFIX}", "")
            candidates.extend(
                _translation_recovery_candidates(path.parent, path.parent.parent, base_name, str(path.resolve()))
            )
    elif isinstance(request, RunRequest):
        for raw_path in request.paths:
            path = Path(raw_path)
            base_name = path.stem
            item = str(path.resolve())
            temporary_dir = path.parent / PREPROCESSED_DIR
            transcription = temporary_dir / f"{base_name}{PREPROCESSED_SUFFIX}{TRANSCRIBED_SUFFIX}.json"
            candidates.append((transcription, ArtifactKind.TRANSCRIPTION, item))
            candidates.extend(_translation_recovery_candidates(temporary_dir, path.parent, base_name, item))

    for path, kind, item in candidates:
        if path.exists():
            context.artifact_created(WorkflowArtifact(path=path, kind=kind, item=item))


def _translation_recovery_candidates(
    temporary_dir: Path, output_dir: Path, base_name: str, item: str
) -> list[tuple[Path, ArtifactKind, str]]:
    return [
        (temporary_dir / f"{base_name}{COMPARE_SUFFIX}.json", ArtifactKind.CHECKPOINT, item),
        (output_dir / f"{base_name}{EDIT_REPORT_SUFFIX}.json", ArtifactKind.REVIEW_REPORT, item),
        (output_dir / f"{base_name}{EDIT_SESSION_SUFFIX}.json", ArtifactKind.EDIT_SESSION, item),
    ]


def _map_error(exc: Exception, context: ExecutionContext) -> WorkflowError:
    category = ErrorCategory.INTERNAL
    retryable = False
    hint = None
    message = redact_sensitive_text(str(exc) or type(exc).__name__)
    lower = message.lower()

    if isinstance(exc, (ImportError, DependencyException)):
        category = ErrorCategory.DEPENDENCY
        hint = "Install the required OpenLRC optional dependency and retry."
    elif isinstance(exc, FileNotFoundError):
        category = ErrorCategory.INPUT if "input" in lower or "preprocessed" in lower else ErrorCategory.RESOURCE
        hint = "Check that the input, binary, and model paths exist."
    elif isinstance(exc, json.JSONDecodeError):
        category = ErrorCategory.VALIDATION
        hint = "Remove or repair the invalid JSON artifact before retrying."
    elif isinstance(exc, (subprocess.CalledProcessError, FfmpegException, TranscribeException)) or (
        "exited with code" in lower
    ):
        category = ErrorCategory.SUBPROCESS
        retryable = True
        hint = "Check the external process output and installed binary."
    elif isinstance(exc, OSError):
        category = ErrorCategory.RESOURCE
        hint = "Check file permissions, free disk space, and output paths."
    elif isinstance(exc, (ValueError, TypeError)):
        error_stage = context.current_stage or context.failed_stage
        category = ErrorCategory.CONFIGURATION if error_stage is WorkflowStage.VALIDATE else ErrorCategory.VALIDATION
        hint = "Review the workflow request and translation mode configuration."
    elif "checkpoint" in lower:
        category = ErrorCategory.CHECKPOINT
        hint = "Keep the checkpoint for inspection or remove it to restart the affected file."
    elif "llama-server" in lower or "model server" in lower or "port" in lower:
        category = ErrorCategory.MODEL_SERVER
        retryable = True
        hint = "Check the llama-server binary, model, and configured port."
    elif isinstance(exc, ChatBotException):
        category = ErrorCategory.PROVIDER
        retryable = not any(word in lower for word in ("authentication", "client error", "fee"))
        hint = "Check provider credentials, model availability, response format, and quota."
    elif any(word in lower for word in ("api key", "authentication", "provider", "rate limit")):
        category = ErrorCategory.PROVIDER
        retryable = "rate limit" in lower
        hint = "Check provider credentials, model availability, and quota."

    return WorkflowError(
        category=category,
        message=message,
        stage=context.current_stage or context.failed_stage,
        item=context.current_item or context.failed_item,
        retryable=retryable,
        hint=hint,
        exception_type=type(exc).__name__,
    )
