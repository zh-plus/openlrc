from __future__ import annotations

import importlib
import stat
import threading
from pathlib import Path

import pytest

from openlrc.application import (
    AppSettings,
    CredentialSource,
    CredentialStore,
    JobController,
    JobRecord,
    JobRecordStatus,
    JobRepository,
    ResolvedCredential,
    SettingsStore,
    WorkflowDraft,
    normalize_input_paths,
    preflight,
)
from openlrc.workflow import (
    RunExecutionStrategy,
    RunRequest,
    TranscribeRequest,
    TranslateRequest,
    TranslationMode,
    WorkflowKind,
    WorkflowResult,
    WorkflowStatus,
    resolve_run_execution_strategy,
)


class MemoryKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


class StaticCredentials:
    last_error = None

    def resolve(self, _provider: str) -> ResolvedCredential:
        return ResolvedCredential("test-key", CredentialSource.KEYCHAIN)


def test_settings_are_atomic_private_and_secret_free(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    settings = AppSettings()
    settings.providers["openai"].model = "test-model"
    settings.general.language = "zh-cn"
    settings.general.reduce_motion = True
    settings.transcription.use_gpu = False
    settings.transcription.flash_attn = False
    store.save(settings)

    payload = path.read_text(encoding="utf-8")
    assert "test-model" in payload
    assert "api_key" not in payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert store.load().providers["openai"].model == "test-model"
    assert store.load().general.language == "zh-cn"
    assert store.load().general.reduce_motion is True
    assert store.load().transcription.use_gpu is False
    assert store.load().transcription.flash_attn is False


def test_settings_language_is_backward_compatible_and_invalid_values_fall_back() -> None:
    legacy = AppSettings.from_dict({"schema_version": 1, "general": {"theme": "openlrc-dark"}})
    invalid = AppSettings.from_dict(
        {"schema_version": 1, "general": {"theme": "missing-theme", "language": "traditional-chinese"}}
    )

    assert legacy.general.language == "en"
    assert invalid.general.theme == "openlrc-dark"
    assert invalid.general.language == "en"


def test_corrupt_settings_are_backed_up_and_defaults_loaded(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{broken", encoding="utf-8")
    store = SettingsStore(path)

    settings = store.load()

    assert settings.schema_version == 1
    assert store.last_warning is not None
    assert not path.exists()
    assert len(list(tmp_path.glob("settings.json.corrupt-*"))) == 1


def test_credentials_prefer_keychain_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    keyring = MemoryKeyring()
    store = CredentialStore()
    monkeypatch.setattr(store, "_keyring", lambda: keyring)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    assert store.resolve("openai").source is CredentialSource.ENVIRONMENT
    store.set("openai", "keychain-secret")
    resolved = store.resolve("openai")
    assert resolved.source is CredentialSource.KEYCHAIN
    assert resolved.value == "keychain-secret"
    store.delete("openai")
    assert store.resolve("openai").value == "environment-secret"


def test_keychain_failure_never_falls_back_to_plaintext(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenKeyring:
        def get_password(self, *_args):
            raise RuntimeError("locked")

        def set_password(self, *_args):
            raise RuntimeError("locked")

    store = CredentialStore()
    monkeypatch.setattr(store, "_keyring", BrokenKeyring)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert store.resolve("openai").source is CredentialSource.MISSING
    assert store.last_error and "Keychain" in store.last_error
    with pytest.raises(RuntimeError, match="not saved"):
        store.set("openai", "must-not-be-written")


def test_history_marks_interrupted_caps_records_and_removes_secrets(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "jobs.json")
    records = [
        JobRecord(
            job_id=str(index),
            workflow="run",
            name=f"job-{index}",
            status=JobRecordStatus.RUNNING if index == 104 else JobRecordStatus.SUCCEEDED,
            input_paths=[f"/a/{index}.mp4"],
            recipe={"api_key": "secret", "path": f"/a/{index}.mp4"},
            started_at=f"2026-01-01T00:{index:03d}:00+00:00",
        )
        for index in range(105)
    ]
    repository.save(records)

    loaded = repository.load()

    assert len(loaded) == 100
    assert loaded[0].status is JobRecordStatus.INTERRUPTED
    raw = (tmp_path / "jobs.json").read_text(encoding="utf-8")
    assert "api_key" not in raw
    assert "secret" not in raw


def test_draft_resolves_online_pipeline_and_all_local_modes_to_memory_saver(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()
    settings = AppSettings()
    credentials = StaticCredentials()

    online = WorkflowDraft(
        workflow="run", paths=[str(media)], translation_backend="online", mode="standard", provider="openai"
    ).build_request(settings, credentials)
    assert resolve_run_execution_strategy(online) is RunExecutionStrategy.PIPELINE

    with pytest.raises(ValueError, match="not Standard"):
        WorkflowDraft(workflow="run", paths=[str(media)], translation_backend="local", mode="standard").build_request(
            settings, credentials
        )

    with pytest.raises(ValueError, match="Standard mode only"):
        WorkflowDraft(workflow="run", paths=[str(media)], translation_backend="online", mode="fast").build_request(
            settings, credentials
        )

    for mode in ("fast", "normal", "normal-plus", "pro"):
        draft = WorkflowDraft(
            workflow="run",
            paths=[str(media)],
            translation_backend="local",
            mode=mode,
            context_provider="local" if mode != "fast" else "",
            context_model="qwen3.5-9b" if mode != "fast" else "",
        )
        request = draft.build_request(settings, credentials)
        assert resolve_run_execution_strategy(request) is RunExecutionStrategy.MEMORY_SAVER


def test_tui_transcribe_draft_exports_a_source_subtitle(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()

    request = WorkflowDraft(workflow="transcribe", paths=[str(media)]).build_request(AppSettings(), StaticCredentials())

    assert isinstance(request, TranscribeRequest)
    assert request.subtitle_output is True
    assert request.clear_temp is True
    assert request.transcription.vad_model == ""


def test_tui_v2_raw_transcription_and_source_subtitle_are_distinct_requests(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()
    settings = AppSettings()

    raw = WorkflowDraft.defaults(settings, WorkflowKind.TRANSCRIBE)
    raw.task = "transcribe-json"
    raw.paths = [str(media)]
    source_subtitle = WorkflowDraft.defaults(settings, WorkflowKind.RUN)
    source_subtitle.task = "source-subtitle"
    source_subtitle.translation_backend = "none"
    source_subtitle.paths = [str(media)]

    raw_request = raw.build_request(settings, StaticCredentials())
    subtitle_request = source_subtitle.build_request(settings, StaticCredentials())

    assert isinstance(raw_request, TranscribeRequest)
    assert raw_request.subtitle_output is False
    assert raw_request.transcription.asr_options == {"use_gpu": True, "flash_attn": True}
    assert isinstance(subtitle_request, RunRequest)
    assert subtitle_request.translation is None


def test_tui_v2_transcription_hardware_options_reach_whisper_config(tmp_path: Path) -> None:
    media = tmp_path / "episode.wav"
    media.touch()
    settings = AppSettings()
    settings.transcription.use_gpu = False
    settings.transcription.flash_attn = False
    draft = WorkflowDraft.defaults(settings, WorkflowKind.TRANSCRIBE)
    draft.task = "transcribe-json"
    draft.paths = [str(media)]

    request = draft.build_request(settings, StaticCredentials())

    assert isinstance(request, TranscribeRequest)
    assert request.transcription.asr_options == {"use_gpu": False, "flash_attn": False}


def test_tui_v2_local_qwen_uses_shared_classic_factory(tmp_path: Path) -> None:
    transcription = tmp_path / "episode.json"
    transcription.touch()
    settings = AppSettings()
    draft = WorkflowDraft(
        task="translate-existing",
        workflow="translate",
        paths=[str(transcription)],
        translation_backend="local-qwen",
        mode="standard",
        qwen_model="qwen.gguf",
    )

    request = draft.build_request(settings, StaticCredentials())

    assert isinstance(request, TranslateRequest)
    assert request.translation.mode is TranslationMode.STANDARD
    assert request.translation.config.local_llm is not None
    assert request.translation.config.local_llm.model_path == "qwen.gguf"


def test_disabled_provider_cannot_be_selected_by_workflow_draft(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()
    settings = AppSettings()
    settings.providers["openai"].enabled = False

    with pytest.raises(ValueError, match="disabled in Settings"):
        WorkflowDraft(workflow="run", paths=[str(media)], translation_backend="online", mode="standard").build_request(
            settings, StaticCredentials()
        )


def test_transcribe_recipe_omits_translation_fields_and_ignores_their_values(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()
    draft = WorkflowDraft(
        workflow="transcribe", paths=[str(media)], translation_backend="online", mode="pro", brief_characters="invalid"
    )

    request = draft.build_request(AppSettings(), StaticCredentials())
    recipe = draft.to_recipe()

    assert isinstance(request, TranscribeRequest)
    assert "translation_backend" not in recipe
    assert "brief_characters" not in recipe


def test_brief_characters_use_simple_source_target_lines(tmp_path: Path) -> None:
    transcription = tmp_path / "episode.json"
    transcription.write_text('{"language":"en","segments":[]}', encoding="utf-8")
    draft = WorkflowDraft(
        workflow="translate",
        paths=[str(transcription)],
        translation_backend="local",
        mode="normal",
        context_provider="local",
        context_model="qwen.gguf",
        brief_summary="Story",
        brief_characters="Alice = 爱丽丝\nBob = 鲍勃",
        brief_tone_style="Natural",
    )

    brief = draft._brief()
    assert brief is not None
    assert [(item.source_name, item.target_name) for item in brief.characters or []] == [
        ("Alice", "爱丽丝"),
        ("Bob", "鲍勃"),
    ]

    draft.brief_characters = "missing separator"
    with pytest.raises(ValueError, match="Source Name = Target Name"):
        draft._brief()


def test_normalize_input_paths_uses_full_identity_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "one" / "episode.mp4"
    second = tmp_path / "two" / "episode.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    monkeypatch.setattr("openlrc.media_utils.get_file_type", lambda _path: "video")

    paths, issues = normalize_input_paths([str(first), str(first), str(second)], "run")

    assert paths == [str(first.resolve()), str(second.resolve())]
    assert len(issues) == 1
    assert issues[0].startswith("Duplicate ignored")


def test_preflight_blocks_sidecar_and_same_stem_collisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "episode.mp4"
    audio = tmp_path / "episode.m4a"
    video.touch()
    audio.touch()
    monkeypatch.setattr(
        "openlrc.media_utils.get_file_type", lambda path: "video" if Path(path).suffix == ".mp4" else "audio"
    )
    monkeypatch.setattr(
        "openlrc.workflow.planning.get_file_type", lambda path: "video" if Path(path).suffix == ".mp4" else "audio"
    )
    preflight_module = importlib.import_module("openlrc.application.preflight")
    monkeypatch.setattr(preflight_module, "_check_resources", lambda *_args: None)

    video.with_suffix(".wav").touch()
    sidecar_report = preflight(
        WorkflowDraft(workflow="transcribe", paths=[str(video)]), AppSettings(), StaticCredentials()
    )
    assert sidecar_report.blocked
    assert any("sidecar WAV" in issue.message for issue in sidecar_report.issues)

    video.with_suffix(".wav").unlink()
    collision_report = preflight(
        WorkflowDraft(workflow="transcribe", paths=[str(video), str(audio)]), AppSettings(), StaticCredentials()
    )
    assert collision_report.blocked
    assert any("map to the same" in issue.message for issue in collision_report.issues)


def test_preflight_validates_translation_json_schema(tmp_path: Path) -> None:
    transcription = tmp_path / "episode.json"
    transcription.write_text("{}", encoding="utf-8")
    report = preflight(
        WorkflowDraft(workflow="translate", paths=[str(transcription)], translation_backend="online", mode="standard"),
        AppSettings(),
        StaticCredentials(),
    )

    assert report.blocked
    assert any("segments array" in issue.message for issue in report.issues)


def test_preflight_rejects_json_that_is_not_an_openlrc_transcription_candidate(tmp_path: Path) -> None:
    transcription = tmp_path / "episode.json"
    transcription.write_text('{"language":"en","segments":[]}', encoding="utf-8")

    report = preflight(
        WorkflowDraft(workflow="translate", paths=[str(transcription)], translation_backend="online", mode="standard"),
        AppSettings(),
        StaticCredentials(),
    )

    assert report.blocked
    assert any("transcription candidate" in issue.message for issue in report.issues)


def test_preflight_requires_explicit_confirmation_for_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "episode.wav"
    output = tmp_path / "episode.lrc"
    audio.touch()
    output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr("openlrc.media_utils.get_file_type", lambda _path: "audio")
    monkeypatch.setattr("openlrc.workflow.planning.get_file_type", lambda _path: "audio")
    preflight_module = importlib.import_module("openlrc.application.preflight")
    monkeypatch.setattr(preflight_module, "_check_resources", lambda *_args: None)

    report = preflight(WorkflowDraft(workflow="transcribe", paths=[str(audio)]), AppSettings(), StaticCredentials())

    assert report.status == "Warning"
    assert any("Existing output will be replaced" in issue.message for issue in report.warnings)


def test_history_preserves_same_basename_inputs_from_different_directories(tmp_path: Path) -> None:
    first = tmp_path / "one" / "episode.mp4"
    second = tmp_path / "two" / "episode.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    draft = WorkflowDraft.defaults(AppSettings())
    draft.paths = [str(first), str(second)]

    recipe = WorkflowDraft.from_recipe(draft.to_recipe())

    assert recipe.paths == [str(first), str(second)]


def test_job_controller_enforces_one_active_job_and_cancels_with_workflow_token(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()
    repository = JobRepository(tmp_path / "jobs.json")
    controller = JobController(repository)
    started = threading.Event()
    release = threading.Event()

    class BlockingExecutor:
        def execute(self, request, context):
            context.workflow_started()
            started.set()
            release.wait(timeout=5)
            status = WorkflowStatus.CANCELLED if context.cancellation_token.is_cancelled else WorkflowStatus.SUCCEEDED
            if status is WorkflowStatus.CANCELLED:
                context.workflow_cancelled()
            else:
                context.workflow_completed(status, 0)
            context.close()
            return WorkflowResult(job_id=context.job_id, workflow=context.workflow, status=status)

    controller._executor = BlockingExecutor()
    draft = WorkflowDraft(workflow="transcribe", paths=[str(media)])
    result_holder: list[WorkflowResult] = []
    thread = threading.Thread(
        target=lambda: result_holder.append(controller.run(draft, AppSettings(), StaticCredentials())), daemon=True
    )
    thread.start()
    assert started.wait(timeout=2)

    with pytest.raises(RuntimeError, match="already running"):
        controller.run(draft, AppSettings(), StaticCredentials())
    assert controller.cancel()
    release.set()
    thread.join(timeout=5)

    assert result_holder[0].status is WorkflowStatus.CANCELLED
    assert controller.records[0].status is JobRecordStatus.CANCELLED
    assert any("cancelled" in line for line in controller.records[0].event_log)
    assert next(iter(controller.records[0].items.values())).state == "cancelled"


def test_job_controller_releases_active_state_when_final_history_save_fails(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()

    class FailingFinalRepository(JobRepository):
        calls = 0

        def upsert(self, records, record):
            self.calls += 1
            if self.calls == 2:
                raise OSError("disk full")
            return super().upsert(records, record)

    class SuccessfulExecutor:
        def execute(self, request, context):
            context.workflow_started()
            context.workflow_completed(WorkflowStatus.SUCCEEDED, 0)
            context.close()
            return WorkflowResult(job_id=context.job_id, workflow=context.workflow, status=WorkflowStatus.SUCCEEDED)

    controller = JobController(FailingFinalRepository(tmp_path / "jobs.json"))
    controller._executor = SuccessfulExecutor()

    result = controller.run(
        WorkflowDraft(workflow="transcribe", paths=[str(media)]), AppSettings(), StaticCredentials()
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    assert not controller.is_running
    assert controller.records[0].status is JobRecordStatus.SUCCEEDED
    assert controller.persistence_warning == "History not saved: disk full"


def test_job_controller_does_not_start_when_initial_history_save_fails(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()

    class FailingRepository(JobRepository):
        def upsert(self, records, record):
            raise OSError("read only")

    controller = JobController(FailingRepository(tmp_path / "jobs.json"))
    with pytest.raises(RuntimeError, match="workflow was not started"):
        controller.run(WorkflowDraft(workflow="transcribe", paths=[str(media)]), AppSettings(), StaticCredentials())

    assert not controller.is_running
    assert controller.active_record is None
