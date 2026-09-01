"""
Official public-game setup and scorecard replay orchestration.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

from .adapter import (
    OfficialArcEnvironment,
    check_official_versions,
    pinned_official_environment,
    require_pinned_official_client,
)
from .scoring import level_rhae, score_game
from .trace import ReplayResult, load_trace, replay_into_environment, trace_sha256, validate_replay
from .types import ArcAction, ArcEnvironmentFactory, GameStatus


class OfficialSetupError(RuntimeError):
    """
    Raised when the complete official public-game cache cannot be prepared.
    """


class HumanConfirmationRequired(RuntimeError):
    """
    Raised before Competition mode unless the caller records human approval.
    """


class ScorecardSubmissionError(RuntimeError):
    """
    Raised when official scorecard replay cannot safely complete.
    """


class ScorecardMode(StrEnum):
    """
    Explicit online scorecard modes supported by the harness.
    """

    DRY_RUN = "dry-run"
    COMPETITION = "competition"


@dataclass(frozen=True, slots=True)
class OfficialGameDescriptor:
    """
    Versioned public-game metadata intended for the host orchestrator.
    """

    game_id: str
    title: str | None
    levels: int


@dataclass(frozen=True, slots=True)
class CacheFile:
    """
    Digest of one downloaded cache artifact.
    """

    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class EnvironmentCacheManifest:
    """
    Auditable result of preparing the public environment cache.
    """

    environments_dir: str
    package_versions: tuple[tuple[str, str], ...]
    game_ids: tuple[str, ...]
    files: tuple[CacheFile, ...]
    aggregate_sha256: str


@dataclass(frozen=True, slots=True)
class ScorecardReplay:
    """
    Replay evidence for one game submitted to an official scorecard.
    """

    game_id: str
    actions: int
    levels_completed: int
    win_levels: int
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class _OfficialReplayExpectation:
    """
    Sensitive host-only values used to verify a scorecard close response.
    """

    game_id: str
    actions: int
    resets: int
    levels: int
    per_level_actions: tuple[int, ...]
    baseline_actions: tuple[int, ...]
    rhae_percent: float


@dataclass(frozen=True, slots=True)
class ScorecardReport:
    """
    Closed official scorecard and its per-game replay evidence.
    """

    mode: ScorecardMode
    scorecard_id: str
    replays: tuple[ScorecardReplay, ...]
    official_response: Any


def _response_attribute(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _validate_official_response(
    response: Any,
    *,
    scorecard_id: str,
    mode: ScorecardMode,
    replays: Sequence[ScorecardReplay],
    expectations: Sequence[_OfficialReplayExpectation],
) -> None:
    """
    Reject missing or internally inconsistent official close responses.
    """

    if response is None:
        raise ScorecardSubmissionError("official scorecard close returned no response")
    if _response_attribute(response, "api_key") is not None:
        raise ScorecardSubmissionError("official scorecard response retained an API key")
    expected = {
        "card_id": scorecard_id,
        "total_environments": len(replays),
        "total_environments_completed": len(replays),
        "total_levels": sum(item.win_levels for item in replays),
        "total_levels_completed": sum(item.levels_completed for item in replays),
        "total_actions": sum(item.actions for item in replays),
    }
    for name, wanted in expected.items():
        if _response_attribute(response, name) != wanted:
            raise ScorecardSubmissionError(
                f"official scorecard response has unexpected {name}"
            )
    competition_mode = _response_attribute(response, "competition_mode")
    expected_competition_mode = mode is ScorecardMode.COMPETITION
    if (
        type(competition_mode) is not bool
        or competition_mode is not expected_competition_mode
    ):
        raise ScorecardSubmissionError("official scorecard response mode does not match the request")
    raw_environments = _response_attribute(response, "environments")
    if not isinstance(raw_environments, Sequence) or isinstance(
        raw_environments, (str, bytes)
    ):
        raise ScorecardSubmissionError("official scorecard response lacks an environment roster")
    by_id = {item.game_id: item for item in expectations}
    returned_ids = [_response_attribute(item, "id") for item in raw_environments]
    if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(by_id):
        raise ScorecardSubmissionError("official scorecard response roster differs from replayed traces")
    for environment in raw_environments:
        game_id = _response_attribute(environment, "id")
        expectation = by_id[str(game_id)]
        raw_runs = _response_attribute(environment, "runs")
        if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)) or len(raw_runs) != 1:
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} does not contain exactly one run"
            )
        run = raw_runs[0]
        if not isinstance(_response_attribute(run, "guid"), str) or not _response_attribute(
            run, "guid"
        ):
            raise ScorecardSubmissionError(f"official scorecard game {game_id!r} lacks a run GUID")
        if _response_attribute(run, "message") is not None:
            raise ScorecardSubmissionError(f"official scorecard game {game_id!r} reports an error")
        try:
            state = GameStatus.coerce(_response_attribute(run, "state"))
        except ValueError as error:
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} has an invalid terminal state"
            ) from error
        scalar_expectations = {
            "actions": expectation.actions,
            "resets": expectation.resets,
            "levels_completed": expectation.levels,
        }
        if state is not GameStatus.WIN or _response_attribute(run, "completed") is not True:
            raise ScorecardSubmissionError(f"official scorecard game {game_id!r} is not a WIN")
        for name, wanted in scalar_expectations.items():
            actual = _response_attribute(run, name)
            if isinstance(actual, bool) or actual != wanted:
                raise ScorecardSubmissionError(
                    f"official scorecard game {game_id!r} has unexpected {name}"
                )
        level_actions = _response_attribute(run, "level_actions")
        baselines = _response_attribute(run, "level_baseline_actions")
        level_scores = _response_attribute(run, "level_scores")
        if not all(
            isinstance(values, Sequence) and not isinstance(values, (str, bytes))
            for values in (level_actions, baselines, level_scores)
        ):
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} lacks per-level evidence"
            )
        assert isinstance(level_actions, Sequence)
        assert isinstance(baselines, Sequence)
        assert isinstance(level_scores, Sequence)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in level_actions
        ):
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} has invalid per-level action counts"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in baselines
        ):
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} has invalid baseline metadata"
            )
        if tuple(level_actions) != expectation.per_level_actions:
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} action allocation differs from replay"
            )
        if tuple(baselines) != expectation.baseline_actions:
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} baseline metadata differs from the pinned cache"
            )
        expected_level_scores = tuple(
            level_rhae(baseline, actions) * 100.0
            for baseline, actions in zip(
                expectation.baseline_actions,
                expectation.per_level_actions,
                strict=True,
            )
        )
        if len(level_scores) != expectation.levels or any(
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(float(actual), wanted, abs_tol=1e-9)
            for actual, wanted in zip(level_scores, expected_level_scores, strict=True)
        ):
            raise ScorecardSubmissionError(
                f"official scorecard game {game_id!r} per-level scores are inconsistent"
            )
        expected_score = score_game(
            expectation.baseline_actions,
            expectation.per_level_actions,
        ).percent
        for holder in (run, environment):
            actual_score = _response_attribute(holder, "score")
            if (
                isinstance(actual_score, bool)
                or not isinstance(actual_score, (int, float))
                or not math.isclose(float(actual_score), expected_score, abs_tol=1e-9)
            ):
                raise ScorecardSubmissionError(
                    f"official scorecard game {game_id!r} aggregate score is inconsistent"
                )
        if not math.isclose(expected_score, expectation.rhae_percent, abs_tol=1e-9):
            raise ScorecardSubmissionError(
                f"local score evidence for game {game_id!r} is internally inconsistent"
            )
    score = _response_attribute(response, "score")
    expected_board = sum(item.rhae_percent for item in expectations) / len(expectations)
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0.0 <= float(score) <= 100.0
        or not math.isclose(float(score), expected_board, abs_tol=1e-9)
    ):
        raise ScorecardSubmissionError("official scorecard response has an invalid RHAE score")


def _load_official_types() -> tuple[Any, Any, Any]:
    arc_module = import_module("arc_agi")
    base_module = import_module("arc_agi.base")
    engine_module = import_module("arcengine")
    return arc_module.Arcade, base_module.OperationMode, engine_module.GameAction


def _new_arcade(
    environments_dir: Path,
    mode_name: str,
    *,
    arcade_builder: Callable[..., Any] | None = None,
) -> tuple[Any, Any]:
    arcade_type, operation_mode, game_action_type = _load_official_types()
    builder = arcade_builder or arcade_type
    mode = getattr(operation_mode, mode_name)
    with pinned_official_environment(mode_name):
        arcade = builder(environments_dir=str(environments_dir), operation_mode=mode)
    require_pinned_official_client(arcade, mode)
    return arcade, game_action_type


def list_official_games(
    environments_dir: str | Path,
    *,
    online_catalog: bool = False,
    verify_versions: bool = True,
    arcade_builder: Callable[..., Any] | None = None,
) -> tuple[OfficialGameDescriptor, ...]:
    """
    List versioned public games without exposing human action baselines.

    Use the online catalog during setup and the offline catalog during play.
    """

    if verify_versions:
        check_official_versions()
    mode_name = "NORMAL" if online_catalog else "OFFLINE"
    arcade, _ = _new_arcade(Path(environments_dir).resolve(), mode_name, arcade_builder=arcade_builder)
    descriptors = []
    for metadata in getattr(arcade, "available_environments", ()):
        game_id = str(metadata.game_id)
        baselines = tuple(getattr(metadata, "baseline_actions", ()) or ())
        descriptors.append(
            OfficialGameDescriptor(
                game_id=game_id,
                title=getattr(metadata, "title", None),
                levels=len(baselines),
            )
        )
    return tuple(sorted(descriptors, key=lambda item: item.game_id))


def inspect_environment_cache(environments_dir: str | Path) -> tuple[tuple[CacheFile, ...], str]:
    """
    Hash every regular cache artifact using relative paths and file bytes.
    """

    root = Path(environments_dir).resolve()
    files: list[CacheFile] = []
    aggregate = sha256(b"ARDEA-ARC-CACHE-v1\0")
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            digest = sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
            files.append(CacheFile(relative_path=relative, size=size, sha256=digest))
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(str(size).encode("ascii"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
    return tuple(files), aggregate.hexdigest()


def setup_public_games(
    environments_dir: str | Path,
    *,
    expected_game_count: int = 25,
    arcade_builder: Callable[..., Any] | None = None,
    verify_versions: bool = True,
) -> EnvironmentCacheManifest:
    """
    Download all catalogued public games and return a hashed cache manifest.

    The official SDK reads ``ARC_API_KEY`` from the process environment. This
    function never accepts, logs, or persists the credential itself.
    """

    if not os.environ.get("ARC_API_KEY"):
        raise OfficialSetupError("ARC_API_KEY is required to download the public game cache")
    if isinstance(expected_game_count, bool) or expected_game_count <= 0:
        raise ValueError("expected_game_count must be positive")
    versions = check_official_versions() if verify_versions else {
        "arc-agi": "unchecked",
        "arcengine": "unchecked",
    }
    root = Path(environments_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    arcade, _ = _new_arcade(root, "NORMAL", arcade_builder=arcade_builder)
    games: list[str] = []
    failed: list[str] = []
    try:
        games = sorted(str(item.game_id) for item in getattr(arcade, "available_environments", ()))
        if len(games) != expected_game_count:
            raise OfficialSetupError(
                f"official catalog contains {len(games)} games; expected exactly {expected_game_count}"
            )
        for game_id in games:
            environment = arcade.make(game_id)
            if environment is None:
                failed.append(game_id)
                continue
            close = getattr(environment, "close", None)
            if callable(close):
                close()
    finally:
        default_card = getattr(arcade, "_default_scorecard_id", None)
        if default_card is not None:
            with suppress(Exception):
                arcade.close_scorecard(default_card)
    if failed:
        raise OfficialSetupError(f"failed to download public games: {', '.join(failed)}")
    files, aggregate = inspect_environment_cache(root)
    if not files:
        raise OfficialSetupError("official setup completed without producing cache artifacts")
    return EnvironmentCacheManifest(
        environments_dir=str(root),
        package_versions=tuple(sorted(versions.items())),
        game_ids=tuple(games),
        files=files,
        aggregate_sha256=aggregate,
    )


def _preflight_traces(
    trace_paths: Iterable[str | Path],
    factory: ArcEnvironmentFactory,
    *,
    expected_game_count: int | None,
) -> tuple[tuple[Path, ReplayResult], ...]:
    paths = tuple(Path(path).resolve() for path in trace_paths)
    if not paths:
        raise ScorecardSubmissionError("at least one trace is required")
    if expected_game_count is not None and len(paths) != expected_game_count:
        raise ScorecardSubmissionError(
            f"submission has {len(paths)} traces; expected exactly {expected_game_count}"
        )
    seen: set[str] = set()
    validated: list[tuple[Path, ReplayResult]] = []
    for path in paths:
        result = validate_replay(path, factory)
        if result.game_id in seen:
            raise ScorecardSubmissionError(f"duplicate game trace {result.game_id!r}")
        if result.status is not GameStatus.WIN or result.levels_completed != result.win_levels:
            raise ScorecardSubmissionError(f"trace for {result.game_id!r} is not a complete WIN")
        seen.add(result.game_id)
        validated.append((path, result))
    return tuple(validated)


def submit_scorecard(
    trace_paths: Sequence[str | Path],
    environments_dir: str | Path,
    *,
    mode: ScorecardMode,
    local_factory: ArcEnvironmentFactory,
    human_confirmed: bool = False,
    tags: Sequence[str] = ("ardea-avo",),
    expected_game_count: int | None = 25,
    verify_versions: bool = True,
    arcade_builder: Callable[..., Any] | None = None,
    on_scorecard_opened: Callable[[str], None] | None = None,
) -> ScorecardReport:
    """
    Preflight every trace locally, then replay all of them into one scorecard.

    ``human_confirmed`` is deliberately supplied by the caller. A CLI must read
    a controlling terminal and pass ``True`` only after the human types its
    explicit confirmation phrase; this library never prompts or reads stdin.
    """

    mode = ScorecardMode(mode)
    if mode is ScorecardMode.COMPETITION and human_confirmed is not True:
        raise HumanConfirmationRequired(
            "Competition submission requires caller-verified human confirmation"
        )
    if verify_versions:
        versions = check_official_versions()
        engine_version = versions["arcengine"]
    else:
        engine_version = "unchecked"
    preflight = _preflight_traces(
        trace_paths,
        local_factory,
        expected_game_count=expected_game_count,
    )
    mode_name = "COMPETITION" if mode is ScorecardMode.COMPETITION else "ONLINE"
    arcade, game_action_type = _new_arcade(
        Path(environments_dir).resolve(),
        mode_name,
        arcade_builder=arcade_builder,
    )
    scorecard_id: str | None = None
    response: Any = None
    replays: list[ScorecardReplay] = []
    expectations: list[_OfficialReplayExpectation] = []
    try:
        scorecard_id = str(arcade.open_scorecard(tags=list(tags)))
        if on_scorecard_opened is not None:
            on_scorecard_opened(scorecard_id)
        for path, local_result in preflight:
            data = load_trace(path)
            raw_environment = arcade.make(data.header.game_id, scorecard_id=scorecard_id)
            if raw_environment is None:
                raise ScorecardSubmissionError(
                    f"official service could not create game {data.header.game_id!r}"
                )
            environment = OfficialArcEnvironment(
                raw_environment,
                game_action_type,
                engine_version=engine_version,
            )
            try:
                frames = replay_into_environment(data, environment)
            finally:
                environment.close()
            final = frames[-1]
            if final.status is not GameStatus.WIN or final.levels_completed != final.win_levels:
                raise ScorecardSubmissionError(
                    f"official replay for {data.header.game_id!r} did not finish all levels"
                )
            replays.append(
                ScorecardReplay(
                    game_id=data.header.game_id,
                    actions=local_result.actions,
                    levels_completed=final.levels_completed,
                    win_levels=final.win_levels,
                    trace_sha256=trace_sha256(path),
                )
            )
            expectations.append(
                _OfficialReplayExpectation(
                    game_id=data.header.game_id,
                    actions=local_result.actions,
                    resets=sum(step.action is ArcAction.RESET for step in data.steps),
                    levels=local_result.win_levels,
                    per_level_actions=local_result.per_level_actions,
                    baseline_actions=environment.info.baseline_actions,
                    rhae_percent=local_result.rhae_percent,
                )
            )
    except ScorecardSubmissionError:
        raise
    except Exception as exc:
        raise ScorecardSubmissionError(
            f"official {mode.value} replay failed"
            + (f" after opening scorecard {scorecard_id}" if scorecard_id is not None else "")
        ) from exc
    finally:
        if scorecard_id is not None:
            try:
                response = arcade.close_scorecard(scorecard_id)
            except Exception as exc:
                if not replays:
                    raise ScorecardSubmissionError(
                        f"could not close official scorecard {scorecard_id}"
                    ) from exc
                raise ScorecardSubmissionError(
                    f"replays completed but official scorecard {scorecard_id} could not be closed"
                ) from exc
    assert scorecard_id is not None
    _validate_official_response(
        response,
        scorecard_id=scorecard_id,
        mode=mode,
        replays=replays,
        expectations=expectations,
    )
    return ScorecardReport(
        mode=mode,
        scorecard_id=scorecard_id,
        replays=tuple(replays),
        official_response=response,
    )
