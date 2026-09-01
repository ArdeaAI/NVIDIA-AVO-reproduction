"""
Lazy adapter for the official ARC-AGI and arcengine packages.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .types import (
    ArcAction,
    ArcContractError,
    ArcEnvironment,
    ArcFrame,
    ArcGameInfo,
    GameStatus,
    normalize_grid,
)

REQUIRED_ARC_AGI_VERSION = "0.9.9"
REQUIRED_ARCENGINE_VERSION = "0.9.3"
OFFICIAL_ARC_BASE_URL = "https://arcprize.org"
_OFFICIAL_ENVIRONMENT_LOCK = threading.RLock()


@contextmanager
def pinned_official_environment(mode_name: str) -> Iterator[None]:
    """
    Pin SDK mode and endpoint while constructing an official Arcade client.

    arc-agi 0.9.9 gives an ambient Competition mode priority over an explicit
    constructor mode and reads its base URL from the environment. Serializing
    and restoring these two variables closes that SDK-level override without
    retaining credentials or changing the caller's environment permanently.
    """

    normalized = mode_name.strip().casefold()
    if normalized not in {"normal", "online", "offline", "competition"}:
        raise ValueError(f"unsupported official operation mode: {mode_name!r}")
    names = ("OPERATION_MODE", "ARC_BASE_URL")
    with _OFFICIAL_ENVIRONMENT_LOCK:
        previous = {name: os.environ.get(name) for name in names}
        os.environ["OPERATION_MODE"] = normalized
        os.environ["ARC_BASE_URL"] = OFFICIAL_ARC_BASE_URL
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


@contextmanager
def level_reset_environment() -> Iterator[None]:
    """
    Scope the SDK's process-global level-reset switch to one serialized action.
    """

    with _OFFICIAL_ENVIRONMENT_LOCK:
        previous = os.environ.get("ONLY_RESET_LEVELS")
        os.environ["ONLY_RESET_LEVELS"] = "true"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("ONLY_RESET_LEVELS", None)
            else:
                os.environ["ONLY_RESET_LEVELS"] = previous


class OfficialDependencyError(RuntimeError):
    """
    Raised when an optional official ARC dependency is unavailable.
    """


class OfficialVersionError(OfficialDependencyError):
    """
    Raised when an official package does not match the pinned version.
    """


def require_pinned_official_client(client: Any, expected_mode: Any) -> None:
    """
    Verify the SDK honored the host-selected operation mode and endpoint.
    """

    actual_mode = getattr(client, "operation_mode", expected_mode)
    if actual_mode != expected_mode:
        raise OfficialDependencyError("official SDK did not honor the requested operation mode")
    actual_url = str(getattr(client, "arc_base_url", OFFICIAL_ARC_BASE_URL)).rstrip("/")
    if actual_url != OFFICIAL_ARC_BASE_URL.rstrip("/"):
        raise OfficialDependencyError("official SDK did not honor the pinned ARC endpoint")


def check_official_versions(
    *,
    arc_agi_version: str = REQUIRED_ARC_AGI_VERSION,
    arcengine_version: str = REQUIRED_ARCENGINE_VERSION,
    version_reader: Callable[[str], str] = version,
) -> dict[str, str]:
    """
    Verify exact package versions without importing either runtime package.
    """

    required = {"arc-agi": arc_agi_version, "arcengine": arcengine_version}
    installed: dict[str, str] = {}
    for distribution, expected in required.items():
        try:
            actual = version_reader(distribution)
        except PackageNotFoundError as exc:
            raise OfficialDependencyError(
                f"optional dependency {distribution} is not installed; run the project setup command"
            ) from exc
        if actual != expected:
            raise OfficialVersionError(
                f"{distribution} version drift: expected {expected}, found {actual}"
            )
        installed[distribution] = actual
    return installed


def _engine_state(value: Any) -> GameStatus:
    raw = getattr(value, "value", value)
    try:
        return GameStatus.coerce(raw)
    except ArcContractError:
        return GameStatus.coerce(getattr(value, "name", value))


def _extract_action_name(value: Any, game_action_type: Any) -> ArcAction:
    if isinstance(value, int):
        value = game_action_type.from_id(value)
    return ArcAction.coerce(getattr(value, "name", value))


class OfficialArcEnvironment:
    """
    Adapt one official engine environment to the stable host protocol.
    """

    def __init__(
        self,
        raw_environment: Any,
        game_action_type: Any,
        *,
        engine_version: str,
        sdk_lock: Any | None = None,
    ) -> None:
        self._raw = raw_environment
        self._game_action_type = game_action_type
        self._sdk_lock = sdk_lock or threading.RLock()
        raw_info = getattr(raw_environment, "info", None)
        if raw_info is None:
            raise ArcContractError("official environment has no info metadata")
        raw_baselines = getattr(raw_info, "baseline_actions", None)
        raw_game_id = getattr(raw_info, "game_id", None)
        if raw_game_id is None:
            raise ArcContractError("official environment has no versioned game_id")
        environment_version_value = getattr(
            raw_info,
            "version",
            getattr(raw_info, "environment_version", raw_game_id),
        )
        environment_version = (
            str(raw_game_id)
            if environment_version_value in {None, "", "unknown"}
            else str(environment_version_value)
        )
        self._info = ArcGameInfo(
            game_id=str(raw_game_id),
            baseline_actions=tuple(int(item) for item in raw_baselines or ()),
            engine_version=engine_version,
            environment_version=environment_version,
        )
        raw_frame = getattr(raw_environment, "observation_space", None)
        if raw_frame is None:
            with self._sdk_lock, level_reset_environment():
                raw_frame = raw_environment.reset()
        self._frame = self._convert_frame(raw_frame)

    @property
    def info(self) -> ArcGameInfo:
        """
        Return host-only game metadata.
        """

        return self._info

    @property
    def frame(self) -> ArcFrame:
        """
        Return the latest settled frame.
        """

        return self._frame

    def _convert_frame(self, raw_frame: Any) -> ArcFrame:
        if raw_frame is None:
            raise ArcContractError("official engine returned no frame")
        animation = getattr(raw_frame, "frame", None)
        if animation is None or len(animation) == 0:
            raise ArcContractError("official engine returned an empty animation")
        settled = animation[-1]
        raw_available = getattr(raw_frame, "available_actions", ()) or ()
        available: list[ArcAction] = []
        for item in raw_available:
            action = _extract_action_name(item, self._game_action_type)
            if action is not ArcAction.RESET and action not in available:
                available.append(action)
        return ArcFrame(
            grid=normalize_grid(settled),
            status=_engine_state(getattr(raw_frame, "state", "NOT_FINISHED")),
            levels_completed=int(getattr(raw_frame, "levels_completed", 0)),
            win_levels=int(getattr(raw_frame, "win_levels", len(self.info.baseline_actions))),
            available_actions=tuple(available),
        )

    def step(
        self,
        action: ArcAction,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> ArcFrame:
        """
        Commit one action, converting ARC row/column to engine y/x data.
        """

        normalized = ArcAction.coerce(action)
        engine_action = self._game_action_type[normalized.value]
        data = {"x": col, "y": row} if normalized.uses_coordinates else None
        with self._sdk_lock:
            if normalized is ArcAction.RESET:
                with level_reset_environment():
                    raw_frame = self._raw.step(engine_action, data=data)
            else:
                raw_frame = self._raw.step(engine_action, data=data)
            self._frame = self._convert_frame(raw_frame)
        return self._frame

    def close(self) -> None:
        """
        Close the underlying environment when it supports explicit cleanup.
        """

        with self._sdk_lock:
            close = getattr(self._raw, "close", None)
            if callable(close):
                close()


class OfficialArcadeFactory:
    """
    Lazily construct official ARC environments with exact dependency pins.
    """

    def __init__(
        self,
        environments_dir: str | Path,
        *,
        operation_mode: Any | None = None,
        scorecard_id: str | None = None,
        verify_versions: bool = True,
    ) -> None:
        self.environments_dir = Path(environments_dir).expanduser().resolve()
        self.operation_mode = operation_mode
        self.scorecard_id = scorecard_id
        self._arcade: Any | None = None
        self._game_action_type: Any | None = None
        self._lock = threading.RLock()
        self._versions = check_official_versions() if verify_versions else {
            "arc-agi": "unchecked",
            "arcengine": "unchecked",
        }

    def _load(self) -> None:
        with self._lock:
            if self._arcade is not None:
                return
            try:
                arcade_type = import_module("arc_agi").Arcade
                operation_mode_type = import_module("arc_agi.base").OperationMode
                game_action_type = import_module("arcengine").GameAction
            except (ImportError, AttributeError) as exc:
                raise OfficialDependencyError(
                    "could not import the official ARC runtime"
                ) from exc
            kwargs: dict[str, Any] = {
                "environments_dir": str(self.environments_dir)
            }
            kwargs["operation_mode"] = (
                self.operation_mode
                if self.operation_mode is not None
                else operation_mode_type.OFFLINE
            )
            mode_name = str(getattr(kwargs["operation_mode"], "name", "OFFLINE"))
            with pinned_official_environment(mode_name):
                arcade = arcade_type(**kwargs)
            require_pinned_official_client(arcade, kwargs["operation_mode"])
            self._game_action_type = game_action_type
            self._arcade = arcade

    def __call__(self, game_id: str) -> ArcEnvironment:
        """
        Create a fresh adapter for a full versioned game identifier.
        """

        with self._lock:
            self._load()
            assert self._arcade is not None
            assert self._game_action_type is not None
            kwargs = (
                {"scorecard_id": self.scorecard_id}
                if self.scorecard_id is not None
                else {}
            )
            with level_reset_environment():
                raw_environment = self._arcade.make(game_id, **kwargs)
        if raw_environment is None:
            raise OfficialDependencyError(f"official ARC runtime could not load game {game_id!r}")
        return OfficialArcEnvironment(
            raw_environment,
            self._game_action_type,
            engine_version=self._versions["arcengine"],
            sdk_lock=self._lock,
        )
