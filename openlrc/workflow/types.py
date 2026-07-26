"""Public request, result, and event contracts for OpenLRC workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias

from openlrc.config import SubtitleOptimizationMode, TranscriptionConfig, TranslationConfig


class WorkflowKind(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    RUN = "run"


class TranslationMode(str, Enum):
    """Canonical product modes exposed by the Workflow API."""

    STANDARD = "standard"
    FAST = "fast"
    NORMAL = "normal"
    NORMAL_PLUS = "normal-plus"
    PRO = "pro"


class RunExecutionStrategy(str, Enum):
    """Resolved scheduling policy for a :class:`RunRequest`."""

    TRANSCRIBE_ONLY = "transcribe-only"
    PIPELINE = "pipeline"
    MEMORY_SAVER = "memory-saver"


class WorkflowStatus(str, Enum):
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_WARNINGS = "succeeded_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStage(str, Enum):
    VALIDATE = "validate"
    PREPROCESS = "preprocess"
    TRANSCRIBE = "transcribe"
    SOURCE_OPTIMIZE = "source-optimize"
    GUIDELINE = "guideline"
    BRIEF = "brief"
    TIMELINE = "timeline"
    TRANSLATE = "translate"
    DETERMINISTIC_REPAIR = "deterministic-repair"
    SEMANTIC_REVIEW = "semantic-review"
    TARGET_OPTIMIZE = "target-optimize"
    EXPORT = "export"
    CLEANUP = "cleanup"


class StageOutcome(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    RESUMED = "resumed"


class ErrorCategory(str, Enum):
    CONFIGURATION = "configuration"
    INPUT = "input"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    SUBPROCESS = "subprocess"
    PROVIDER = "provider"
    MODEL_SERVER = "model-server"
    CHECKPOINT = "checkpoint"
    VALIDATION = "validation"
    INTERNAL = "internal"


class ArtifactKind(str, Enum):
    TRANSCRIPTION = "transcription"
    SUBTITLE = "subtitle"
    BILINGUAL_SUBTITLE = "bilingual-subtitle"
    SOURCE_SUBTITLE = "source-subtitle"
    CHECKPOINT = "checkpoint"
    REVIEW_REPORT = "review-report"
    EDIT_SESSION = "edit-session"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class WorkflowTranslationConfig:
    mode: TranslationMode
    config: TranslationConfig


@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    paths: tuple[str | Path, ...]
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    src_lang: str | None = None
    noise_suppress: bool = False
    skip_preprocess: bool = False
    subtitle_output: bool = False
    subtitle_optimization: SubtitleOptimizationMode = SubtitleOptimizationMode.AGGRESSIVE
    clear_temp: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _path_tuple(self.paths))
        object.__setattr__(self, "subtitle_optimization", SubtitleOptimizationMode(self.subtitle_optimization))


@dataclass(frozen=True, slots=True)
class TranslateRequest:
    transcribed_paths: tuple[str | Path, ...]
    translation: WorkflowTranslationConfig
    target_lang: str = "zh-cn"
    bilingual_sub: bool = False
    subtitle_optimization: SubtitleOptimizationMode = SubtitleOptimizationMode.AGGRESSIVE
    clear_checkpoint: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "transcribed_paths", _path_tuple(self.transcribed_paths))
        object.__setattr__(self, "subtitle_optimization", SubtitleOptimizationMode(self.subtitle_optimization))


@dataclass(frozen=True, slots=True)
class RunRequest:
    paths: tuple[str | Path, ...]
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    translation: WorkflowTranslationConfig | None = None
    src_lang: str | None = None
    target_lang: str = "zh-cn"
    noise_suppress: bool = False
    bilingual_sub: bool = False
    subtitle_optimization: SubtitleOptimizationMode = SubtitleOptimizationMode.AGGRESSIVE
    clear_temp: bool = True
    clear_checkpoint: bool | None = None
    skip_preprocess: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _path_tuple(self.paths))
        object.__setattr__(self, "subtitle_optimization", SubtitleOptimizationMode(self.subtitle_optimization))


WorkflowRequest: TypeAlias = TranscribeRequest | TranslateRequest | RunRequest


def _path_tuple(value) -> tuple[str | Path, ...]:
    if isinstance(value, (str, Path)):
        return (value,)
    return tuple(value)


@dataclass(frozen=True, slots=True)
class WorkflowError:
    category: ErrorCategory
    message: str
    stage: WorkflowStage | None = None
    item: str | None = None
    retryable: bool = False
    hint: str | None = None
    exception_type: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowArtifact:
    path: Path
    kind: ArtifactKind
    item: str | None = None
    primary: bool = False


@dataclass(frozen=True, slots=True)
class ReviewStatus:
    item: str
    incomplete: bool = False
    checkpoint: Path | None = None
    report: Path | None = None
    session: Path | None = None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    job_id: str
    workflow: WorkflowKind
    status: WorkflowStatus
    translation_mode: TranslationMode | None = None
    outputs: tuple[Path, ...] = ()
    artifacts: tuple[WorkflowArtifact, ...] = ()
    reviews: tuple[ReviewStatus, ...] = ()
    api_fee: float = 0.0
    elapsed_seconds: float = 0.0
    error: WorkflowError | None = None


@dataclass(frozen=True, slots=True)
class EventHeader:
    job_id: str
    sequence: int
    timestamp: datetime
    workflow: WorkflowKind
    item: str | None = None
    translation_mode: TranslationMode | None = None


@dataclass(frozen=True, slots=True)
class WorkflowStartedEvent:
    header: EventHeader


@dataclass(frozen=True, slots=True)
class WorkflowCompletedEvent:
    header: EventHeader
    status: WorkflowStatus
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class WorkflowFailedEvent:
    header: EventHeader
    error: WorkflowError


@dataclass(frozen=True, slots=True)
class WorkflowCancelledEvent:
    header: EventHeader
    message: str = "Workflow cancelled."


@dataclass(frozen=True, slots=True)
class StageStartedEvent:
    header: EventHeader
    stage: WorkflowStage


@dataclass(frozen=True, slots=True)
class StageProgressEvent:
    header: EventHeader
    stage: WorkflowStage
    completed: float
    total: float
    message: str | None = None

    @property
    def percent(self) -> float:
        return 0.0 if self.total <= 0 else min(100.0, max(0.0, self.completed / self.total * 100.0))


@dataclass(frozen=True, slots=True)
class StageCompletedEvent:
    header: EventHeader
    stage: WorkflowStage
    outcome: StageOutcome = StageOutcome.COMPLETED
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ModelLifecycleEvent:
    header: EventHeader
    model: str
    state: str
    owned: bool | None = None
    role: str | None = None
    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactCreatedEvent:
    header: EventHeader
    artifact: WorkflowArtifact


@dataclass(frozen=True, slots=True)
class LogMessageEvent:
    header: EventHeader
    level: str
    message: str


WorkflowEvent: TypeAlias = (
    WorkflowStartedEvent
    | WorkflowCompletedEvent
    | WorkflowFailedEvent
    | WorkflowCancelledEvent
    | StageStartedEvent
    | StageProgressEvent
    | StageCompletedEvent
    | ModelLifecycleEvent
    | ArtifactCreatedEvent
    | LogMessageEvent
)


class EventSink(Protocol):
    def __call__(self, event: WorkflowEvent) -> None: ...
