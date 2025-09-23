from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


Vec2 = Tuple[float, float]


class DictLikeBaseModel(BaseModel, Mapping[str, Any]):
    """Base model that exposes dict-like access for backward compatibility."""

    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, item: str) -> Any:  # pragma: no cover - fallback path
        return getattr(self, item)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - mapping protocol
        return iter(self.__class__.model_fields)

    def __len__(self) -> int:  # pragma: no cover - mapping protocol
        return len(self.__class__.model_fields)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class BotSnapshot(DictLikeBaseModel):
    id: int
    x: float
    y: float
    theta: float
    vx: float
    vy: float
    hp: int
    alive: bool
    team: int


class ProjectileSnapshot(DictLikeBaseModel):
    id: int
    x: float
    y: float
    vx: float
    vy: float
    team: int
    age: int
    shooter_id: int
    ttl: float


class ShotEvent(DictLikeBaseModel):
    type: Literal["shot"] = "shot"
    tick: int
    bot_id: int
    projectile_id: int
    pos: Vec2
    heading: float
    cooldown_remaining: float
    total_shots: int


class ProjectileRemovedEvent(DictLikeBaseModel):
    type: Literal["projectile_removed"] = "projectile_removed"
    tick: int
    projectile_id: int
    shooter_id: Optional[int] = None
    reason: Literal["expired", "out_of_bounds", "hit", "unknown"]
    pos: Optional[Vec2] = None


class HitEvent(DictLikeBaseModel):
    type: Literal["hit"] = "hit"
    tick: int
    projectile_shooter: int
    target: int
    damage: float
    pos: Vec2
    shooter_accuracy: float


class DeathEvent(DictLikeBaseModel):
    type: Literal["death"] = "death"
    tick: int
    bot_id: int
    killer_id: int
    pos: Vec2
    killer_kills: int


BattleEvent = (
    ShotEvent
    | ProjectileRemovedEvent
    | HitEvent
    | DeathEvent
)


class BattleFrame(DictLikeBaseModel):
    tick: int
    time: float
    bots: List[BotSnapshot] = Field(default_factory=list)
    projectiles: List[ProjectileSnapshot] = Field(default_factory=list)
    events: List[BattleEvent] = Field(default_factory=list)
    precomp_visible_by_bot: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict
    )


class ScoreBreakdown(DictLikeBaseModel):
    accuracy: float
    damage: float
    kills: float
    survival: float
    damage_penalty: float


class BotScore(DictLikeBaseModel):
    bot_id: int
    team: int
    total_score: float
    hit_rate: float
    damage_efficiency: float
    survival_rate: float
    shots_fired: int
    shots_hit: int
    damage_dealt: float
    damage_taken: float
    kills: int
    deaths: int
    score_breakdown: ScoreBreakdown


class TeamScore(DictLikeBaseModel):
    bots: List[BotScore] = Field(default_factory=list)
    total_score: float = 0.0
    avg_hit_rate: float = 0.0
    total_kills: int = 0
    total_deaths: int = 0
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    bots_alive: int = 0


class MVPHighlight(DictLikeBaseModel):
    bot_id: Optional[int] = None
    score: float = 0.0
    team: Optional[int] = None


class AccuracyHighlight(DictLikeBaseModel):
    bot_id: Optional[int] = None
    hit_rate: float = 0.0
    shots: str = "0/0"


class DamageHighlight(DictLikeBaseModel):
    bot_id: Optional[int] = None
    damage: float = 0.0


class KillHighlight(DictLikeBaseModel):
    bot_id: Optional[int] = None
    kills: int = 0


class BotFunctionSummary(DictLikeBaseModel):
    function_errors: int = 0
    avg_exec_time_ms: float = 0.0
    version: int = 0


class BattleSummary(DictLikeBaseModel):
    total_shots: int
    total_hits: int
    total_deaths: int
    hit_rate: float
    final_hp: List[int]
    final_positions: List[Optional[Vec2]]
    bot_scores: List[BotScore]
    team_scores: Dict[int, TeamScore]
    mvp: MVPHighlight = Field(default_factory=MVPHighlight)
    best_accuracy: AccuracyHighlight = Field(default_factory=AccuracyHighlight)
    most_damage: DamageHighlight = Field(default_factory=DamageHighlight)
    most_kills: KillHighlight = Field(default_factory=KillHighlight)
    battle_intensity: float
    lethality: float
    overall_accuracy: float
    control_system: Optional[str] = None
    function_executions: Optional[int] = None
    function_timeouts: Optional[int] = None
    function_errors: Optional[int] = None
    avg_function_time_ms: Optional[float] = None
    bot_functions: Dict[int, BotFunctionSummary] = Field(default_factory=dict)
    numba_enabled: Optional[bool] = None
    sandbox_enabled: Optional[bool] = None


class RunnerStats(DictLikeBaseModel):
    total_executions: int
    total_timeouts: int
    total_errors: int
    avg_execution_time: float
    bot_count: int
    cache_size: int
    numba_available: bool
    sandbox_enabled: bool


class WallDefinition(DictLikeBaseModel):
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float


class BattleMetadata(DictLikeBaseModel):
    seed: int
    duration: float
    winner: str
    reason: str
    arena_size: Vec2
    total_ticks: int
    real_time: float
    walls: List[WallDefinition] = Field(default_factory=list)
    control_system: Optional[str] = None
    compilation_success: Optional[Dict[int, bool]] = None
    runner_stats: Optional[RunnerStats] = None
    sandbox_enabled: Optional[bool] = None


class BattleData(DictLikeBaseModel):
    metadata: BattleMetadata
    timeline: List[BattleFrame]
    summary: BattleSummary

    def model_dump_jsonable(self) -> Dict[str, Any]:
        """Dump battle data to a JSON-serializable dict."""
        return self.model_dump(mode="json")
