"""
Offline validation of a complete ARC-AGI-3 campaign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ardea_avo.arc import (
    CampaignBank,
    OfficialArcadeFactory,
    OfficialGameDescriptor,
    board_rhae,
    validate_replay,
)
from ardea_avo.runtime import RunContext
from ardea_avo.runtime._io import atomic_write_json, utc_now

PRIMARY_GAME_COUNT = 25
PRIMARY_LEVEL_COUNT = 183
PRIMARY_RHAE_PERCENT = 100.0


@dataclass(frozen=True, slots=True)
class ValidatedGame:
    """
    Fresh-replay result for one selected game trace.
    """

    game_id: str
    trace_path: str
    trace_sha256: str
    actions: int
    levels_completed: int
    win_levels: int
    per_level_actions: tuple[int, ...]
    rhae_percent: float


@dataclass(frozen=True, slots=True)
class CampaignValidation:
    """
    JSON-friendly result of all local Competition-eligibility gates.
    """

    schema: str
    run_id: str
    validated_at: str
    valid: bool
    eligible_for_competition: bool
    expected_games: int
    expected_levels: int
    solved_games: int
    solved_levels: int
    submitted_actions: int
    board_rhae_percent: float
    games: tuple[ValidatedGame, ...]
    errors: tuple[str, ...]
    contamination: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a canonical JSON-compatible representation.
        """

        return asdict(self)


def validate_campaign(
    context: RunContext,
    descriptors: tuple[OfficialGameDescriptor, ...],
    factory: OfficialArcadeFactory,
    *,
    contamination: tuple[str, ...] = (),
) -> CampaignValidation:
    """
    Validate provenance, roster, selected bytes, and every fresh replay offline.
    """

    errors: list[str] = []
    try:
        context.validate()
    except Exception as error:
        errors.append(f"run provenance failed: {type(error).__name__}: {error}")

    expected_by_id = {item.game_id: item for item in descriptors}
    if len(expected_by_id) != len(descriptors):
        errors.append("offline catalog contains duplicate game identifiers")
    if len(descriptors) != PRIMARY_GAME_COUNT:
        errors.append(
            f"offline catalog has {len(descriptors)} games; primary protocol requires {PRIMARY_GAME_COUNT}"
        )
    catalog_levels = sum(item.levels for item in descriptors)
    if catalog_levels != PRIMARY_LEVEL_COUNT:
        errors.append(
            f"offline catalog has {catalog_levels} levels; primary protocol requires {PRIMARY_LEVEL_COUNT}"
        )

    configured = context.manifest.config.get("game_roster")
    current_roster = [
        {"game_id": item.game_id, "levels": item.levels}
        for item in sorted(descriptors, key=lambda descriptor: descriptor.game_id)
    ]
    if configured != current_roster:
        errors.append("current offline game roster differs from the immutable run manifest")

    bank = CampaignBank(context.directory / "bank.json")
    entries = bank.entries()
    if {entry.game_id for entry in entries} != set(expected_by_id):
        extra = sorted({entry.game_id for entry in entries} - set(expected_by_id))
        if extra:
            errors.append(f"selected trace roster contains unexpected games: {extra}")

    games: list[ValidatedGame] = []
    run_root = context.directory.resolve()
    for entry in entries:
        try:
            trace_path = Path(entry.trace_path)
            resolved = trace_path.resolve(strict=True)
            if trace_path.is_symlink() or not resolved.is_relative_to(run_root):
                raise ValueError("banked trace must be a regular file inside its run directory")
            if not resolved.is_file():
                raise ValueError("banked trace is not a regular file")
            bank.validate_selected(entry.game_id)
            replay = validate_replay(resolved, factory)
            descriptor = expected_by_id.get(entry.game_id)
            if descriptor is None:
                raise ValueError("banked game is absent from the offline catalog")
            if replay.win_levels != descriptor.levels:
                raise ValueError("replayed level count differs from the offline catalog")
            games.append(
                ValidatedGame(
                    game_id=replay.game_id,
                    trace_path=str(resolved),
                    trace_sha256=replay.trace_sha256,
                    actions=replay.actions,
                    levels_completed=replay.levels_completed,
                    win_levels=replay.win_levels,
                    per_level_actions=replay.per_level_actions,
                    rhae_percent=replay.rhae_percent,
                )
            )
        except Exception as error:
            errors.append(f"trace {entry.game_id!r} failed: {type(error).__name__}: {error}")

    games.sort(key=lambda game: game.game_id)
    solved_games = len(games)
    solved_levels = sum(game.levels_completed for game in games)
    submitted_actions = sum(game.actions for game in games)
    scores = {game.game_id: game.rhae_percent / 100.0 for game in games}
    rhae = (
        board_rhae([scores.get(item.game_id, 0.0) for item in descriptors])
        if descriptors
        else 0.0
    )
    complete = (
        solved_games == PRIMARY_GAME_COUNT
        and solved_levels == PRIMARY_LEVEL_COUNT
        and not errors
    )
    eligible = complete and not contamination and rhae == PRIMARY_RHAE_PERCENT
    return CampaignValidation(
        schema="ardea.arc.campaign-validation.v2",
        run_id=context.manifest.run_id,
        validated_at=utc_now(),
        valid=complete,
        eligible_for_competition=eligible,
        expected_games=PRIMARY_GAME_COUNT,
        expected_levels=PRIMARY_LEVEL_COUNT,
        solved_games=solved_games,
        solved_levels=solved_levels,
        submitted_actions=submitted_actions,
        board_rhae_percent=rhae,
        games=tuple(games),
        errors=tuple(errors),
        contamination=tuple(contamination),
    )


def write_validation(context: RunContext, validation: CampaignValidation) -> Path:
    """
    Atomically store a validation result when the run remains mutable.
    """

    path = context.directory / "validation.json"
    if context.is_sealed:
        return path
    atomic_write_json(path, validation.to_dict())
    return path
