from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openlrc.config import (
    ContextAssistance,
    ContextLLMConfig,
    EditConfig,
    HyMT2Mode,
    TranscriptionConfig,
    TranslationConfig,
)
from openlrc.context import TranslationBriefInput
from openlrc.exceptions import ChatBotException, TranscribeException
from openlrc.models import ModelConfig, ModelProvider
from openlrc.openlrc import LRCer
from openlrc.workflow import (
    ArtifactKind,
    CancellationToken,
    ErrorCategory,
    ExecutionContext,
    LogMessageEvent,
    ModelLifecycleEvent,
    OwnedProcessRegistry,
    RunRequest,
    StageProgressEvent,
    StageStartedEvent,
    TranscribeRequest,
    TranslateRequest,
    TranslationMode,
    WorkflowArtifact,
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowExecutor,
    WorkflowFailedEvent,
    WorkflowKind,
    WorkflowStartedEvent,
    WorkflowStatus,
    WorkflowTranslationConfig,
)


def _fake_lrcer(outputs: list[Path]) -> MagicMock:
    lrcer = MagicMock()
    lrcer.transcribe.return_value = outputs
    lrcer.translate.return_value = outputs
    lrcer.run.return_value = outputs
    lrcer.review_statuses = {}
    lrcer.api_fee = 0.25
    return lrcer


def _execute_with_fake(request, outputs: list[Path]):
    lrcer = _fake_lrcer(outputs)
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(request)
    return result, lrcer


@pytest.mark.parametrize(
    "provider",
    [
        ModelProvider.OPENAI,
        ModelProvider.ANTHROPIC,
        ModelProvider.GOOGLE,
        ModelProvider.LITELLM,
        ModelProvider.THIRD_PARTY,
    ],
)
def test_standard_accepts_all_online_providers(tmp_path: Path, provider: ModelProvider) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    config = TranslationConfig(chatbot=ModelConfig(provider=provider, name="test-model"))
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.STANDARD, config))

    result, _ = _execute_with_fake(request, [tmp_path / "source.lrc"])

    assert result.status is WorkflowStatus.SUCCEEDED


def test_standard_accepts_managed_local_qwen(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    config = TranslationConfig.local_qwen35_9b(model="qwen.gguf")
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.STANDARD, config))

    result, _ = _execute_with_fake(request, [tmp_path / "source.lrc"])

    assert result.status is WorkflowStatus.SUCCEEDED


def test_canonical_hymt2_mode_matrix(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    context = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="context")
    complete = TranslationBriefInput(summary="Story", characters=[], tone_style="Natural")
    configs = [
        (TranslationMode.FAST, TranslationConfig.local_hy_mt2_7b(mode=HyMT2Mode.FAST)),
        (TranslationMode.NORMAL, TranslationConfig.local_hy_mt2_7b(mode=HyMT2Mode.NORMAL, translation_brief=complete)),
        (
            TranslationMode.NORMAL_PLUS,
            TranslationConfig.local_hy_mt2_7b(
                mode=HyMT2Mode.NORMAL_PLUS,
                translation_brief=complete,
                edit_config=EditConfig(enabled=True, max_rounds=0, semantic_review=False),
            ),
        ),
        (TranslationMode.PRO, TranslationConfig.local_hy_mt2_7b(mode=HyMT2Mode.PRO, context_llm=context)),
        (
            TranslationMode.FAST,
            TranslationConfig.local_hy_mt2(size="hy-mt2-30b-a3b", model="explicit-30b.gguf", mode=HyMT2Mode.FAST),
        ),
    ]

    for mode, config in configs:
        request = TranslateRequest((source,), WorkflowTranslationConfig(mode, config))
        result, _ = _execute_with_fake(request, [tmp_path / f"{mode.value}.lrc"])
        assert result.status is WorkflowStatus.SUCCEEDED


def test_deprecated_config_alias_is_reported_canonically(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    complete = TranslationBriefInput(summary="Story", characters=[], tone_style="Natural")
    with pytest.warns(FutureWarning, match="deprecated"):
        config = TranslationConfig.local_hy_mt2_7b(mode="context", translation_brief=complete)
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.NORMAL, config))

    result, _ = _execute_with_fake(request, [tmp_path / "normal.lrc"])

    assert result.translation_mode is TranslationMode.NORMAL


def test_normal_plus_with_semantic_rounds_requires_context_model(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    complete = TranslationBriefInput(summary="Story", characters=[], tone_style="Natural")
    config = TranslationConfig.local_hy_mt2_7b(
        mode=HyMT2Mode.NORMAL_PLUS,
        translation_brief=complete,
        context_llm=ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="reviewer"),
    )
    config.context_llm = None
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.NORMAL_PLUS, config))

    result = WorkflowExecutor().execute(request)

    assert result.status is WorkflowStatus.FAILED
    assert result.error is not None
    assert result.error.category is ErrorCategory.CONFIGURATION


def test_workflow_validation_rejects_context_model_hidden_by_assistance_off(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    complete = TranslationBriefInput(summary="Story", characters=[], tone_style="")
    config = TranslationConfig.local_hy_mt2_7b(
        mode=HyMT2Mode.NORMAL, context_assistance=ContextAssistance.OFF, translation_brief=complete
    )
    config.context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="hidden")
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.NORMAL, config))

    with patch("openlrc.workflow.executor.LRCer") as lrcer_cls:
        result = WorkflowExecutor().execute(request)

    assert result.status is WorkflowStatus.FAILED
    assert result.error is not None
    assert result.error.category is ErrorCategory.CONFIGURATION
    assert "conflicts" in result.error.message
    lrcer_cls.assert_not_called()


def test_mode_config_mismatch_is_structured_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    config = TranslationConfig.local_hy_mt2_7b(mode=HyMT2Mode.FAST)
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.NORMAL, config))

    result = WorkflowExecutor().execute(request)

    assert result.status is WorkflowStatus.FAILED
    assert result.error is not None
    assert result.error.category is ErrorCategory.CONFIGURATION


def test_event_sequence_is_strict_and_sink_failure_is_isolated(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    events = []

    def sink(event) -> None:
        events.append(event)
        if isinstance(event, WorkflowStartedEvent):
            raise RuntimeError("UI failure")

    context = ExecutionContext(WorkflowKind.TRANSCRIBE, event_sink=sink)
    request = TranscribeRequest((source,))
    lrcer = _fake_lrcer([tmp_path / "audio.json"])
    lrcer.close.side_effect = lambda: context.model_event("test-model", "stopped", owned=True)
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(request, context)

    sequences = [event.header.sequence for event in events]
    assert result.status is WorkflowStatus.SUCCEEDED
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))
    assert sum(event.__class__.__name__.startswith("WorkflowCompleted") for event in events) == 1
    assert isinstance(events[-1], WorkflowCompletedEvent)


def test_transcribe_can_export_source_subtitle_without_changing_json_default(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    subtitle = tmp_path / "video.srt"
    source.write_bytes(b"test")
    subtitle.write_text("subtitle", encoding="utf-8")

    subtitle_request = TranscribeRequest((source,), subtitle_output=True, clear_temp=True)
    subtitle_result, subtitle_lrcer = _execute_with_fake(subtitle_request, [subtitle])
    json_result, json_lrcer = _execute_with_fake(TranscribeRequest((source,)), [tmp_path / "video.json"])

    assert subtitle_result.outputs == (subtitle,)
    assert any(artifact.kind is ArtifactKind.SUBTITLE and artifact.primary for artifact in subtitle_result.artifacts)
    subtitle_lrcer.run.assert_called_once()
    assert subtitle_lrcer.run.call_args.kwargs["skip_trans"] is True
    json_lrcer.transcribe.assert_called_once()
    json_lrcer.run.assert_not_called()
    assert json_result.outputs == (tmp_path / "video.json",)


def test_invalid_mode_is_structured_and_does_not_leave_executor_locked(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    config = TranslationConfig(chatbot=ModelConfig(provider=ModelProvider.OPENAI, name="test-model"))
    invalid = TranslateRequest((source,), WorkflowTranslationConfig("legacy", config))  # type: ignore[arg-type]
    executor = WorkflowExecutor()

    failed = executor.execute(invalid)
    valid = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.STANDARD, config))
    lrcer = _fake_lrcer([tmp_path / "source.lrc"])
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        succeeded = executor.execute(valid)

    assert failed.status is WorkflowStatus.FAILED
    assert failed.error is not None
    assert failed.error.category is ErrorCategory.CONFIGURATION
    assert succeeded.status is WorkflowStatus.SUCCEEDED


def test_event_delivery_sequence_is_strict_across_threads() -> None:
    events = []
    context = ExecutionContext(WorkflowKind.RUN, event_sink=events.append)
    barrier = threading.Barrier(8)

    def emit_messages(worker: int) -> None:
        barrier.wait()
        for index in range(25):
            context.log("info", f"{worker}:{index}")

    threads = [threading.Thread(target=emit_messages, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(event, LogMessageEvent) for event in events)
    sequences = [event.header.sequence for event in events]
    assert sequences == list(range(1, 201))


def test_reentrant_event_sink_does_not_deadlock() -> None:
    events = []
    context: ExecutionContext

    def sink(event) -> None:
        events.append(event)
        if len(events) == 1:
            context.log("debug", "nested")

    context = ExecutionContext(WorkflowKind.RUN, event_sink=sink)
    context.log("info", "outer")

    assert [event.header.sequence for event in events] == [1, 2]


def test_cancellation_terminates_registered_owned_process() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    context = ExecutionContext(WorkflowKind.TRANSCRIBE)
    context.processes.register(process)
    try:
        context.cancellation_token.cancel()
        assert process.wait(timeout=2) is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        context.close()


def test_owned_process_registry_escalates_after_terminate_timeout() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="worker", timeout=0), 0]
    registry = OwnedProcessRegistry(terminate_timeout=0)
    registry.register(process)

    registry.terminate_all()

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2


def test_pre_cancelled_request_returns_cancelled_without_creating_lrcer(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    token = CancellationToken()
    token.cancel()
    context = ExecutionContext(WorkflowKind.TRANSCRIBE, cancellation_token=token)

    with patch("openlrc.workflow.executor.LRCer") as lrcer_cls:
        result = WorkflowExecutor().execute(TranscribeRequest((source,)), context)

    assert result.status is WorkflowStatus.CANCELLED
    lrcer_cls.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("worker stopped during cancellation"),
        TranscribeException("whisper exited with code 143"),
        subprocess.CalledProcessError(143, ["worker"]),
    ],
)
def test_cancellation_takes_precedence_over_concurrent_runtime_errors(tmp_path: Path, error: Exception) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    events = []
    token = CancellationToken()
    context = ExecutionContext(WorkflowKind.TRANSCRIBE, event_sink=events.append, cancellation_token=token)
    lrcer = _fake_lrcer([])

    def cancel_then_fail(*_args, **_kwargs):
        token.cancel()
        raise error

    lrcer.transcribe.side_effect = cancel_then_fail
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(TranscribeRequest((source,)), context)

    assert result.status is WorkflowStatus.CANCELLED
    assert result.error is None
    assert sum(isinstance(event, WorkflowCancelledEvent) for event in events) == 1
    assert not any(isinstance(event, WorkflowFailedEvent) for event in events)


def test_runtime_error_without_cancellation_remains_failed(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    lrcer = _fake_lrcer([])
    lrcer.transcribe.side_effect = RuntimeError("worker failed")

    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(TranscribeRequest((source,)))

    assert result.status is WorkflowStatus.FAILED
    assert result.error is not None


def test_executor_rejects_a_second_concurrent_job(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    entered = threading.Event()
    release = threading.Event()
    lrcer = _fake_lrcer([tmp_path / "audio.json"])

    def slow_transcribe(*_args, **_kwargs):
        entered.set()
        release.wait(2)
        return [tmp_path / "audio.json"]

    lrcer.transcribe.side_effect = slow_transcribe
    executor = WorkflowExecutor()
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        thread = threading.Thread(target=executor.execute, args=(TranscribeRequest((source,)),))
        thread.start()
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="already executing"):
            executor.execute(TranscribeRequest((source,)))
        release.set()
        thread.join(2)
    assert not thread.is_alive()


def test_execution_context_is_single_use_and_executor_lock_is_released(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    output = tmp_path / "audio.json"
    source.write_bytes(b"test")
    output.write_text("{}", encoding="utf-8")
    events = []
    context = ExecutionContext(WorkflowKind.TRANSCRIBE, event_sink=events.append)
    executor = WorkflowExecutor()
    lrcer = _fake_lrcer([output])

    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer) as lrcer_cls:
        first = executor.execute(TranscribeRequest((source,)), context)
        with pytest.raises(RuntimeError, match="single-use"):
            executor.execute(TranscribeRequest((source,)), context)
        second = executor.execute(TranscribeRequest((source,)), ExecutionContext(WorkflowKind.TRANSCRIBE))

    assert first.status is WorkflowStatus.SUCCEEDED
    assert second.status is WorkflowStatus.SUCCEEDED
    assert sum(isinstance(event, WorkflowStartedEvent) for event in events) == 1
    assert sum(isinstance(event, WorkflowCompletedEvent) for event in events) == 1
    assert lrcer_cls.call_count == 2


def test_closed_execution_context_is_rejected_without_locking_executor(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    output = tmp_path / "audio.json"
    source.write_bytes(b"test")
    output.write_text("{}", encoding="utf-8")
    context = ExecutionContext(WorkflowKind.TRANSCRIBE)
    context.close()
    context.close()
    executor = WorkflowExecutor()

    with pytest.raises(RuntimeError, match="single-use"):
        executor.execute(TranscribeRequest((source,)), context)

    lrcer = _fake_lrcer([output])
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = executor.execute(TranscribeRequest((source,)))

    assert result.status is WorkflowStatus.SUCCEEDED


def test_run_none_is_untranslated_workflow(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    request = RunRequest((source,), translation=None, clear_temp=False)

    result, lrcer = _execute_with_fake(request, [tmp_path / "audio.lrc"])

    assert result.status is WorkflowStatus.SUCCEEDED
    assert lrcer.run.call_args.kwargs["skip_trans"] is True


def test_run_forwards_explicit_checkpoint_cleanup_choice(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    request = RunRequest((source,), translation=None, clear_temp=False, clear_checkpoint=False)

    result, lrcer = _execute_with_fake(request, [tmp_path / "audio.lrc"])

    assert result.status is WorkflowStatus.SUCCEEDED
    assert lrcer.run.call_args.kwargs["clear_checkpoint"] is False


def test_incomplete_review_returns_success_with_warnings_and_recovery_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    checkpoint = tmp_path / "source_compare.json"
    source.write_text("{}", encoding="utf-8")
    checkpoint.write_text("{}", encoding="utf-8")
    config = TranslationConfig(chatbot=ModelConfig(provider=ModelProvider.OPENAI, name="test-model"))
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.STANDARD, config))
    lrcer = _fake_lrcer([tmp_path / "source.lrc"])
    lrcer.review_statuses = {"source": {"incomplete": True, "checkpoint": checkpoint}}

    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(request)

    assert result.status is WorkflowStatus.SUCCEEDED_WITH_WARNINGS
    assert result.reviews[0].incomplete is True
    assert any(artifact.kind is ArtifactKind.CHECKPOINT for artifact in result.artifacts)


@pytest.mark.parametrize(("workflow", "expected_primary"), [(WorkflowKind.TRANSCRIBE, True), (WorkflowKind.RUN, False)])
def test_transcription_artifact_primary_depends_on_workflow(
    tmp_path: Path, workflow: WorkflowKind, expected_primary: bool
) -> None:
    audio = tmp_path / "audio_preprocessed.wav"
    transcription = tmp_path / "audio_preprocessed_transcribed.json"
    audio.write_bytes(b"test")
    transcription.write_text("{}", encoding="utf-8")
    context = ExecutionContext(workflow)
    lrcer = LRCer(transcription=TranscriptionConfig(), execution_context=context)
    try:
        result = lrcer._transcribe_single(audio)
    finally:
        lrcer.close()
        context.close()

    assert result == transcription
    artifact = next(item for item in context.artifacts if item.kind is ArtifactKind.TRANSCRIPTION)
    assert artifact.primary is expected_primary


def test_result_keeps_only_retained_artifacts_after_run_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    intermediate = tmp_path / "audio_transcribed.json"
    output = tmp_path / "audio.lrc"
    source.write_bytes(b"test")
    intermediate.write_text("{}", encoding="utf-8")
    output.write_text("subtitle", encoding="utf-8")
    context = ExecutionContext(WorkflowKind.RUN)
    lrcer = _fake_lrcer([output])

    def finish_run(*_args, **_kwargs):
        context.artifact_created(
            WorkflowArtifact(intermediate, ArtifactKind.TRANSCRIPTION, item="audio", primary=False)
        )
        context.artifact_created(WorkflowArtifact(output, ArtifactKind.SUBTITLE, item="audio", primary=True))
        intermediate.unlink()
        return [output]

    lrcer.run.side_effect = finish_run
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(RunRequest((source,), translation=None), context)

    assert result.outputs == (output,)
    assert [artifact.path for artifact in result.artifacts] == [output]


def test_execution_context_owned_paths_are_runtime_only_and_explicit(tmp_path: Path) -> None:
    context = ExecutionContext(WorkflowKind.TRANSCRIBE)
    created = tmp_path / "created.wav"
    existing = tmp_path / "existing.wav"
    existing.touch()

    context.register_owned_path(created)

    assert context.owns_path(created)
    assert not context.owns_path(existing)
    assert context.owned_paths == frozenset({created.resolve()})
    context.unregister_owned_path(created)
    assert not context.owns_path(created)


def test_cancelled_run_partial_outputs_exclude_missing_and_intermediate_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    intermediate = tmp_path / "audio_transcribed.json"
    retained_output = tmp_path / "first.lrc"
    removed_output = tmp_path / "removed.lrc"
    source.write_bytes(b"test")
    intermediate.write_text("{}", encoding="utf-8")
    retained_output.write_text("subtitle", encoding="utf-8")
    removed_output.write_text("subtitle", encoding="utf-8")
    context = ExecutionContext(WorkflowKind.RUN)
    lrcer = _fake_lrcer([])

    def cancel_run(*_args, **_kwargs):
        context.artifact_created(
            WorkflowArtifact(intermediate, ArtifactKind.TRANSCRIPTION, item="audio", primary=False)
        )
        context.artifact_created(WorkflowArtifact(retained_output, ArtifactKind.SUBTITLE, item="first", primary=True))
        context.artifact_created(WorkflowArtifact(removed_output, ArtifactKind.SUBTITLE, item="removed", primary=True))
        removed_output.unlink()
        context.cancellation_token.cancel()
        raise RuntimeError("worker exited during cancellation")

    lrcer.run.side_effect = cancel_run
    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(RunRequest((source,), translation=None), context)

    assert result.status is WorkflowStatus.CANCELLED
    assert result.outputs == (retained_output,)
    assert {artifact.path for artifact in result.artifacts} == {intermediate, retained_output}


def test_file_scope_assigns_brief_timeline_and_model_events_to_each_item(tmp_path: Path) -> None:
    events = []
    context = ExecutionContext(WorkflowKind.RUN, event_sink=events.append)
    lrcer = LRCer(execution_context=context)
    transcribed_paths = [
        tmp_path / "first_preprocessed_transcribed.json",
        tmp_path / "second_preprocessed_transcribed.json",
    ]
    subtitle = MagicMock()

    def build_with_context_events(base_name, *_args, **_kwargs):
        with lrcer._workflow_stage("brief"):
            context.model_event("context-model", "starting", owned=True)
        with lrcer._workflow_stage("timeline"):
            context.model_event("context-model", "ready", owned=True)
        context.model_event("context-model", "stopped", owned=True)
        return None

    try:
        with (
            patch("openlrc.openlrc.Subtitle.from_json", return_value=subtitle),
            patch.object(lrcer, "post_process", return_value=subtitle),
            patch.object(lrcer, "_is_video_transcription", return_value=False),
            patch.object(lrcer, "_build_final_subtitle", side_effect=build_with_context_events),
        ):
            for transcribed_path in transcribed_paths:
                lrcer._process_transcribed_file(transcribed_path, "zh-cn")
    finally:
        lrcer.close()
        context.close()

    relevant_events = [
        event
        for event in events
        if (
            isinstance(event, StageStartedEvent)
            and event.stage.value in {"brief", "timeline"}
            or isinstance(event, ModelLifecycleEvent)
        )
    ]
    assert {event.header.item for event in relevant_events} == {str(path.resolve()) for path in transcribed_paths}
    assert all(event.header.item is not None for event in relevant_events)


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (ChatBotException("Authentication failed: api_key=sk-abcdefghijk"), ErrorCategory.PROVIDER),
        (RuntimeError("Checkpoint is corrupt"), ErrorCategory.CHECKPOINT),
        (RuntimeError("llama-server port is unavailable"), ErrorCategory.MODEL_SERVER),
    ],
)
def test_runtime_failures_are_structured_and_secrets_are_redacted(
    tmp_path: Path, error: Exception, category: ErrorCategory
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    config = TranslationConfig(chatbot=ModelConfig(provider=ModelProvider.OPENAI, name="test-model"))
    request = TranslateRequest((source,), WorkflowTranslationConfig(TranslationMode.STANDARD, config))
    lrcer = _fake_lrcer([])
    lrcer.translate.side_effect = error

    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(request)

    assert result.status is WorkflowStatus.FAILED
    assert result.error is not None
    assert result.error.category is category
    assert "sk-abcdefghijk" not in result.error.message


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        ("Authorization: Bearer ordinary-secret", "ordinary-secret"),
        ("Authorization=Basic base64-secret", "base64-secret"),
        ("X-API-Key: header-secret", "header-secret"),
        ("token=plain-secret", "plain-secret"),
        ("https://example.test/v1?key=AIzaOrdinarySecret&x=1", "AIzaOrdinarySecret"),
        ("https://example.test/v1?access_token=query-secret", "query-secret"),
    ],
)
def test_failure_events_redact_common_credential_shapes(tmp_path: Path, message: str, secret: str) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    events = []
    context = ExecutionContext(WorkflowKind.TRANSCRIBE, event_sink=events.append)
    lrcer = _fake_lrcer([])
    lrcer.transcribe.side_effect = RuntimeError(message)

    with patch("openlrc.workflow.executor.LRCer", return_value=lrcer):
        result = WorkflowExecutor().execute(TranscribeRequest((source,)), context)

    failed_event = next(event for event in events if isinstance(event, WorkflowFailedEvent))
    assert result.error is not None
    assert secret not in result.error.message
    assert secret not in failed_event.error.message
    assert "<redacted>" in result.error.message


def test_progress_percent_is_bounded() -> None:
    event = StageProgressEvent(header=MagicMock(), stage=MagicMock(), completed=125, total=100)
    assert event.percent == 100.0
