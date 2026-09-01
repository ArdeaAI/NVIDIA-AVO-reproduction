"""
Command-line entry point for ARC campaigns and generic AVO targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ardea_avo.arc import (
    OFFICIAL_ARC_BASE_URL,
    REQUIRED_ARC_AGI_VERSION,
    REQUIRED_ARCENGINE_VERSION,
    CampaignBank,
    OfficialArcadeFactory,
    OfficialGameDescriptor,
    ScorecardMode,
    list_official_games,
    setup_public_games,
    submit_scorecard,
    validate_replay,
)
from ardea_avo.campaign_runner import (
    AnthropicEpisodeDriver,
    CampaignRunner,
    CodexEpisodeDriver,
    EpisodeDriver,
    OpenAIEpisodeDriver,
    import_warm_memory,
)
from ardea_avo.doctor import checks_pass, run_checks
from ardea_avo.integrity import (
    IntegrityError,
    capture_agent_bundle,
    capture_cache_manifest,
    capture_git_provenance,
    provenance_payload,
    scan_agent_bundle,
)
from ardea_avo.reporting import (
    RunReport,
    SubmissionSummary,
    build_run_report,
    write_report,
)
from ardea_avo.runtime import (
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_SERVICE_TIER,
    AnthropicMessagesBackend,
    BudgetExceeded,
    BudgetLedger,
    CodexOAuthBackend,
    OpenAIResponsesBackend,
    ResultsManager,
    RunContext,
    RunLease,
    RunMode,
    pricing_for_model,
)
from ardea_avo.runtime._io import atomic_write_json, canonical_json, sha256_json, utc_now
from ardea_avo.target_config import TargetFile, load_target
from ardea_avo.validation import CampaignValidation, validate_campaign, write_validation

MODEL = "gpt-5.6-sol"
SUPPORTED_BACKENDS = ("codex-oauth", "openai-api", "anthropic-api")
DEFAULT_MAX_COST_USD = Decimal("20.00")
PRIMARY_GAME_COUNT = 25
PRIMARY_LEVEL_COUNT = 183
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = REPOSITORY_ROOT / "targets" / "arc_agi_3" / "agent_bundle"
PUBLIC_ROSTER_PATH = REPOSITORY_ROOT / "targets" / "arc_agi_3" / "public_roster.json"
SETUP_MANIFEST_SCHEMA = "ardea.arc.public-cache.v1"
EXPECTED_PUBLIC_ROSTER_SHA256 = (
    "19611e0ad29479c9fd84b759d0468ac7293830d9a6db02e4c03c0275828316da"
)
CANONICAL_AGENT_BUNDLE_SHA256 = (
    "cd1ea4a7b840d30a9a3f16fa233a39f7d6310f9d88173f0852d42aac61828617"
)


def _model_for_backend(backend: str) -> str:
    """
    Return the immutable model identity selected by a backend lane.
    """

    if backend == "anthropic-api":
        return ANTHROPIC_MODEL
    if backend in {"codex-oauth", "openai-api"}:
        return MODEL
    raise ValueError(f"unsupported backend: {backend}")


def _auth_for_backend(backend: str) -> str:
    """
    Return the non-secret authentication method recorded in run provenance.
    """

    return "chatgpt-oauth" if backend == "codex-oauth" else "api-key"


@dataclass(frozen=True, slots=True)
class IntegrityComparison:
    """
    Current-state findings relative to a run's immutable provenance.
    """

    contamination: tuple[str, ...]
    errors: tuple[str, ...]


def _uses_disclosed_model_lane(context: RunContext) -> bool:
    """
    Return whether a run uses the backend and model disclosed for the full result.
    """

    return (
        context.manifest.backend == "anthropic-api"
        and context.manifest.model == ANTHROPIC_MODEL
    )


def _validate_campaign_jobs(backend: str, jobs: int) -> None:
    """
    Require serial scheduling for the qualifying provider-ambiguity boundary.
    """

    if backend == "anthropic-api" and jobs != 1:
        raise ValueError(
            "the qualifying anthropic-api lane requires --jobs 1 so an ambiguous "
            "provider outcome stops all subsequent model calls"
        )


def _budget_ambiguity_contamination(context: RunContext) -> tuple[str, ...]:
    """
    Report permanent ambiguity markers and live worst-case budget holds.
    """

    findings: list[str] = []
    if any(
        event["kind"]
        in {
            "provider.usage_ambiguous",
            "budget.reservations_unreconciled",
            "budget.reservations_recovered",
        }
        and event["payload"].get("provider_usage_may_be_unreported", True) is True
        for event in context.events()
    ):
        findings.append(
            "provider usage may be unreported after an ambiguous or interrupted request"
        )
    active_reservations = BudgetLedger(
        context.directory,
        max_cost_usd=context.manifest.max_cost_usd,
        pricing=pricing_for_model(context.manifest.model),
    ).active_reservations()
    if active_reservations:
        findings.append(
            "budget retains worst-case holds for unreconciled provider requests"
        )
    return tuple(findings)


def _unreconciled_reservation_payload(
    active: Mapping[str, Decimal],
) -> dict[str, Any]:
    """
    Build a stable public receipt for conservative unresolved budget holds.
    """

    return {
        "reservations": [
            {
                "reservation_id": reservation_id,
                "max_unreported_usd": str(amount),
            }
            for reservation_id, amount in sorted(active.items())
        ],
        "max_unreported_usd": str(sum(active.values(), Decimal("0"))),
        "provider_usage_may_be_unreported": True,
        "full_reservations_retained": True,
    }


def _record_unreconciled_reservations(
    context: RunContext,
    budget: BudgetLedger,
) -> dict[str, Any] | None:
    """
    Idempotently anchor active worst-case holds without releasing their capacity.
    """

    active = budget.active_reservations()
    if not active:
        return None
    payload = _unreconciled_reservation_payload(active)
    if not any(
        event["kind"] == "budget.reservations_unreconciled"
        and event["payload"] == payload
        for event in context.events()
    ):
        context.append_event("budget.reservations_unreconciled", payload)
    return payload


def _positive_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a decimal amount") from error
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _add_path_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--bundle-dir", type=Path, default=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the complete CLI grammar without reading credentials or files.
    """

    parser = argparse.ArgumentParser(
        prog="app",
        description="Auditable AVO reproduction harness for ARC-AGI-3",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cold", action="store_true", help="start a state-cold ARC campaign")
    mode.add_argument("--results", metavar="PARENT_RUN", help="start a warm child of a prior run")
    mode.add_argument("--resume", metavar="RUN_ID", help="resume the same ARC run")
    parser.add_argument("--slug", default="arc-agi-3")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    parser.add_argument("--max-cost-usd", type=_positive_decimal)
    parser.add_argument("--jobs", type=_positive_int, default=1)
    parser.add_argument("--attempts", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--episodes-per-attempt", type=_positive_int, default=argparse.SUPPRESS
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("ARDEA_AVO_ENVIRONMENTS_DIR", "environment_files")),
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)

    commands = parser.add_subparsers(dest="command")
    setup = commands.add_parser("setup", help="download and hash the complete public game cache")
    _add_path_overrides(setup)
    doctor = commands.add_parser("doctor", help="run read-only dependency and authentication checks")
    doctor.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=argparse.SUPPRESS)
    _add_path_overrides(doctor)
    validate = commands.add_parser("validate", help="fresh-replay a run entirely offline")
    validate.add_argument("run_id")
    _add_path_overrides(validate)
    report = commands.add_parser("report", help="print an offline evidence report")
    report.add_argument("run_id")
    _add_path_overrides(report)
    compete = commands.add_parser("compete", help="replay a complete bank into one online scorecard")
    compete.add_argument("run_id")
    compete.add_argument("--dry-run", action="store_true")
    _add_path_overrides(compete)

    evolve = commands.add_parser("evolve", help="run AVO against a generic target YAML")
    evolve.add_argument("target", type=Path)
    evolve_mode = evolve.add_mutually_exclusive_group(required=True)
    evolve_mode.add_argument("--cold", action="store_true")
    evolve_mode.add_argument("--results", metavar="PARENT_RUN")
    evolve_mode.add_argument("--resume", metavar="RUN_ID")
    evolve.add_argument("--slug", default="evolve")
    evolve.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=argparse.SUPPRESS)
    evolve.add_argument("--max-cost-usd", type=_positive_decimal, default=argparse.SUPPRESS)
    evolve.add_argument("--attempts", type=_positive_int, default=argparse.SUPPRESS)
    _add_path_overrides(evolve)
    return parser


def _load_local_environment() -> None:
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _setup_manifest_path(cache_dir: Path) -> Path:
    """
    Place host-only catalog provenance beside, rather than inside, the cache.
    """

    return cache_dir.parent / f".{cache_dir.name}.ardea-setup.json"


def _game_roster(
    descriptors: Sequence[OfficialGameDescriptor],
) -> list[dict[str, Any]]:
    """
    Return the deterministic full public-game roster for host provenance.
    """

    return [
        {"game_id": item.game_id, "levels": item.levels}
        for item in sorted(descriptors, key=lambda descriptor: descriptor.game_id)
    ]


def _public_roster_sha256(
    roster: Sequence[Mapping[str, Any]],
    package_versions: Mapping[str, str],
) -> str:
    """
    Commit the sorted public roster and exact SDK contract with domain separation.
    """

    games = [dict(item) for item in roster]
    if any(
        set(game) != {"game_id", "levels"}
        or not isinstance(game["game_id"], str)
        or not game["game_id"]
        or isinstance(game["levels"], bool)
        or not isinstance(game["levels"], int)
        or game["levels"] <= 0
        for game in games
    ):
        raise ValueError("public roster entries must contain a game ID and positive level count")
    if len({game["game_id"] for game in games}) != len(games):
        raise ValueError("public roster contains duplicate game IDs")
    games.sort(key=lambda game: str(game["game_id"]))
    payload = {
        "schema": "ardea.arc.public-roster.v1",
        "sdk": dict(package_versions),
        "games": games,
    }
    encoded = b"ARDEA-ARC-PUBLIC-ROSTER-v1\0" + canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_public_roster() -> list[dict[str, Any]]:
    """
    Load and validate the committed host-only public benchmark identity.
    """

    try:
        document = json.loads(PUBLIC_ROSTER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("frozen public roster is unreadable") from error
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "snapshot",
        "sdk",
        "roster_sha256",
        "games",
    }:
        raise RuntimeError("frozen public roster fields do not match the protocol")
    expected_versions = {
        "arc-agi": REQUIRED_ARC_AGI_VERSION,
        "arcengine": REQUIRED_ARCENGINE_VERSION,
    }
    games = document["games"]
    if (
        document["schema"] != "ardea.arc.public-roster-snapshot.v1"
        or not isinstance(document["snapshot"], Mapping)
        or document["sdk"] != expected_versions
        or not isinstance(games, list)
    ):
        raise RuntimeError("frozen public roster metadata is invalid")
    digest = _public_roster_sha256(games, expected_versions)
    if (
        digest != EXPECTED_PUBLIC_ROSTER_SHA256
        or document["roster_sha256"] != EXPECTED_PUBLIC_ROSTER_SHA256
        or len(games) != PRIMARY_GAME_COUNT
        or sum(int(game["levels"]) for game in games) != PRIMARY_LEVEL_COUNT
    ):
        raise RuntimeError("frozen public roster identity is invalid")
    return [dict(game) for game in games]


def _validate_setup_manifest(
    cache_dir: Path,
    descriptors: Sequence[OfficialGameDescriptor],
) -> dict[str, Any]:
    """
    Bind an offline catalog to the roster downloaded from the pinned endpoint.
    """

    path = _setup_manifest_path(cache_dir)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"public cache setup manifest is missing or unsafe; run `app setup`: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("public cache setup manifest is malformed") from error
    required = {
        "schema",
        "created_at",
        "official_arc_base_url",
        "package_versions",
        "game_roster",
        "game_roster_sha256",
        "cache_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("public cache setup manifest fields do not match the protocol")
    roster = _game_roster(descriptors)
    canonical_roster = _canonical_public_roster()
    if value["schema"] != SETUP_MANIFEST_SCHEMA:
        raise RuntimeError("public cache setup manifest schema is unsupported")
    if value["official_arc_base_url"] != OFFICIAL_ARC_BASE_URL:
        raise RuntimeError("public cache was not downloaded from the pinned official endpoint")
    expected_versions = {
        "arc-agi": REQUIRED_ARC_AGI_VERSION,
        "arcengine": REQUIRED_ARCENGINE_VERSION,
    }
    if value["package_versions"] != expected_versions:
        raise RuntimeError("public cache setup manifest has unexpected SDK versions")
    if (
        value["game_roster"] != roster
        or roster != canonical_roster
        or value["game_roster_sha256"]
        != _public_roster_sha256(roster, expected_versions)
        or value["game_roster_sha256"] != EXPECTED_PUBLIC_ROSTER_SHA256
    ):
        raise RuntimeError(
            "offline game roster differs from the frozen public benchmark protocol"
        )
    cache = capture_cache_manifest(cache_dir)
    if value["cache_sha256"] != cache.sha256:
        raise RuntimeError("offline environment cache differs from its setup-time digest")
    return value


def _official_catalog(cache_dir: Path) -> tuple[OfficialGameDescriptor, ...]:
    descriptors = list_official_games(cache_dir, online_catalog=False)
    if len(descriptors) != PRIMARY_GAME_COUNT:
        raise RuntimeError(
            f"offline cache exposes {len(descriptors)} public games; expected {PRIMARY_GAME_COUNT}; run `app setup`"
        )
    levels = sum(item.levels for item in descriptors)
    if levels != PRIMARY_LEVEL_COUNT:
        raise RuntimeError(
            f"offline cache exposes {levels} levels; expected {PRIMARY_LEVEL_COUNT}"
        )
    _validate_setup_manifest(cache_dir, descriptors)
    return descriptors


def _backend_preflight(cache_dir: Path, backend: str) -> None:
    checks = run_checks(cache_dir, backend)
    failed = [check for check in checks if check.required and not check.ok]
    if failed:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        raise RuntimeError(f"backend preflight failed: {detail}")


def _capture_provenance(
    cache_dir: Path,
    bundle_dir: Path,
    descriptors: Sequence[OfficialGameDescriptor],
    *,
    backend: str,
) -> dict[str, Any]:
    game_ids = tuple(item.game_id for item in descriptors)
    repository = capture_git_provenance(REPOSITORY_ROOT)
    bundle = capture_agent_bundle(bundle_dir, game_ids)
    cache = capture_cache_manifest(cache_dir)
    payload = dict(provenance_payload(repository=repository, agent_bundle=bundle, cache=cache))
    setup_manifest = _validate_setup_manifest(cache_dir, descriptors)
    payload["isolation"] = (
        "tool_api_enforced"
        if backend in {"openai-api", "anthropic-api"}
        else "native_best_effort"
    )
    payload["backend_contract"] = {
        "backend": backend,
        "model": _model_for_backend(backend),
        "reasoning_effort": "max",
        **(
            {
                "max_tokens": ANTHROPIC_MAX_TOKENS,
                "thinking": "adaptive",
                "inference_geo": "global",
                "cache_ttl": "ephemeral-5m",
                "parallel_tool_calls": False,
            }
            if backend == "anthropic-api"
            else (
                {
                    "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
                    "service_tier": OPENAI_SERVICE_TIER,
                    "parallel_tool_calls": False,
                }
                if backend == "openai-api"
                else {}
            )
        ),
    }
    payload["official_arc_base_url"] = OFFICIAL_ARC_BASE_URL
    payload["setup_manifest_sha256"] = sha256_json(setup_manifest)
    payload["public_roster_protocol_sha256"] = EXPECTED_PUBLIC_ROSTER_SHA256
    payload["canonical_agent_bundle_sha256"] = CANONICAL_AGENT_BUNDLE_SHA256
    payload["canonical_agent_bundle"] = (
        bundle.sha256 == CANONICAL_AGENT_BUNDLE_SHA256
    )
    return payload


def _run_configuration(
    cache_dir: Path,
    bundle_dir: Path,
    descriptors: Sequence[OfficialGameDescriptor],
    *,
    attempts: int,
    episodes_per_attempt: int,
    jobs: int,
) -> dict[str, Any]:
    return {
        "target": "arc-agi-3-public",
        "official_arc_base_url": OFFICIAL_ARC_BASE_URL,
        "cache_dir": str(cache_dir),
        "bundle_dir": str(bundle_dir),
        "attempts_per_game": attempts,
        "episodes_per_attempt": episodes_per_attempt,
        "jobs": jobs,
        "game_roster": _game_roster(descriptors),
        "game_roster_sha256": _public_roster_sha256(
            _game_roster(descriptors),
            {
                "arc-agi": REQUIRED_ARC_AGI_VERSION,
                "arcengine": REQUIRED_ARCENGINE_VERSION,
            },
        ),
        "expected_games": PRIMARY_GAME_COUNT,
        "expected_levels": PRIMARY_LEVEL_COUNT,
    }


def _new_arc_context(
    args: argparse.Namespace,
    manager: ResultsManager,
    descriptors: tuple[OfficialGameDescriptor, ...],
    cache_dir: Path,
    bundle_dir: Path,
) -> RunContext:
    backend = args.backend or "codex-oauth"
    _validate_campaign_jobs(backend, args.jobs)
    _backend_preflight(cache_dir, backend)
    maximum = args.max_cost_usd or DEFAULT_MAX_COST_USD
    config = _run_configuration(
        cache_dir,
        bundle_dir,
        descriptors,
        attempts=getattr(args, "attempts", 3),
        episodes_per_attempt=getattr(args, "episodes_per_attempt", 12),
        jobs=args.jobs,
    )
    provenance = _capture_provenance(
        cache_dir,
        bundle_dir,
        descriptors,
        backend=backend,
    )
    repository_provenance = provenance.get("repository")
    if (
        backend in {"openai-api", "anthropic-api"}
        and isinstance(repository_provenance, Mapping)
        and repository_provenance.get("dirty") is True
    ):
        raise RuntimeError(
            "qualifying API runs require a committed, clean source repository"
        )
    auth_method = _auth_for_backend(backend)
    model = _model_for_backend(backend)
    if args.cold:
        return manager.create_cold(
            args.slug,
            backend=backend,
            auth_method=auth_method,
            model=model,
            reasoning_effort="max",
            max_cost_usd=maximum,
            observation_mode="text",
            config=config,
            provenance=provenance,
        )
    assert args.results is not None
    parent = manager.open(args.results)
    if parent.manifest.config.get("target") != "arc-agi-3-public":
        raise ValueError("warm parent is not an ARC-AGI-3 campaign")
    parent_bundle = parent.manifest.provenance.get("agent_bundle", {})
    parent_cache = parent.manifest.provenance.get("cache", {})
    if not isinstance(parent_bundle, Mapping) or parent_bundle.get("sha256") != provenance["agent_bundle"]["sha256"]:
        raise ValueError("warm child must use the parent's exact frozen agent bundle")
    if not isinstance(parent_cache, Mapping) or parent_cache.get("sha256") != provenance["cache"]["sha256"]:
        raise ValueError("warm child must use the parent's exact environment cache")
    return manager.create_warm(
        args.results,
        args.slug,
        backend=backend,
        auth_method=auth_method,
        model=model,
        reasoning_effort="max",
        max_cost_usd=maximum,
        observation_mode="text",
        config=config,
        provenance=provenance,
    )


def _resume_arc_context(args: argparse.Namespace, manager: ResultsManager) -> RunContext:
    context = manager.resume(args.resume)
    if context.manifest.config.get("target") != "arc-agi-3-public":
        raise ValueError("run is not an ARC-AGI-3 campaign")
    if args.backend is not None and args.backend != context.manifest.backend:
        raise ValueError("resume cannot change model backend")
    if args.max_cost_usd is not None:
        current = BudgetLedger(
            context.directory,
            max_cost_usd=context.manifest.max_cost_usd,
            pricing=pricing_for_model(context.manifest.model),
        )
        if args.max_cost_usd < current.max_cost_usd:
            raise ValueError("resume cannot lower the lifetime model budget")
        if args.max_cost_usd > current.max_cost_usd:
            current.revise_cap(args.max_cost_usd)
            context.append_event("budget.revised", {"max_cost_usd": str(args.max_cost_usd)})
    attempts, episodes = _campaign_limits(context)
    requested_attempts = getattr(args, "attempts", attempts)
    requested_episodes = getattr(args, "episodes_per_attempt", episodes)
    if requested_attempts < attempts or requested_episodes < episodes:
        raise ValueError("resume cannot lower campaign attempt or episode limits")
    if requested_attempts > attempts or requested_episodes > episodes:
        context.append_event(
            "campaign.limits_revised",
            {
                "attempts_per_game": requested_attempts,
                "episodes_per_attempt": requested_episodes,
            },
        )
    return context


def _campaign_limits(context: RunContext) -> tuple[int, int]:
    """
    Resolve append-only campaign limit increases from the immutable baseline.
    """

    config = context.manifest.config
    attempts = config.get("attempts_per_game")
    episodes = config.get("episodes_per_attempt")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (attempts, episodes)
    ):
        raise ValueError("run manifest has invalid campaign limits")
    assert isinstance(attempts, int)
    assert isinstance(episodes, int)
    for event in context.events():
        if event["kind"] != "campaign.limits_revised":
            continue
        payload = event["payload"]
        if not isinstance(payload, Mapping) or set(payload) != {
            "attempts_per_game",
            "episodes_per_attempt",
        }:
            raise ValueError("campaign limit revision is malformed")
        revised_attempts = payload["attempts_per_game"]
        revised_episodes = payload["episodes_per_attempt"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (revised_attempts, revised_episodes)
        ):
            raise ValueError("campaign limit revision is malformed")
        assert isinstance(revised_attempts, int)
        assert isinstance(revised_episodes, int)
        if revised_attempts < attempts or revised_episodes < episodes:
            raise ValueError("campaign limit revisions must be monotonic")
        attempts = revised_attempts
        episodes = revised_episodes
    return attempts, episodes


def _config_path(context: RunContext, key: str) -> Path:
    value = context.manifest.config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run manifest lacks {key}")
    return Path(value).resolve()


def _integrity_comparison(
    context: RunContext,
    descriptors: Sequence[OfficialGameDescriptor],
) -> IntegrityComparison:
    errors: list[str] = []
    contamination: list[str] = []
    stored = context.manifest.provenance
    stored_repository = stored.get("repository")
    stored_bundle = stored.get("agent_bundle")
    stored_cache = stored.get("cache")
    if not all(isinstance(item, Mapping) for item in (stored_repository, stored_bundle, stored_cache)):
        return IntegrityComparison((), ("run manifest lacks complete integrity provenance",))
    assert isinstance(stored_repository, Mapping)
    assert isinstance(stored_bundle, Mapping)
    assert isinstance(stored_cache, Mapping)

    if stored.get("official_arc_base_url") != OFFICIAL_ARC_BASE_URL:
        errors.append("run provenance does not bind the pinned official ARC endpoint")
    expected_backend_contract = {
        "backend": context.manifest.backend,
        "model": context.manifest.model,
        "reasoning_effort": context.manifest.reasoning_effort,
        **(
            {
                "max_tokens": ANTHROPIC_MAX_TOKENS,
                "thinking": "adaptive",
                "inference_geo": "global",
                "cache_ttl": "ephemeral-5m",
                "parallel_tool_calls": False,
            }
            if context.manifest.backend == "anthropic-api"
            else (
                {
                    "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
                    "service_tier": OPENAI_SERVICE_TIER,
                    "parallel_tool_calls": False,
                }
                if context.manifest.backend == "openai-api"
                else {}
            )
        ),
    }
    if stored.get("backend_contract") != expected_backend_contract:
        errors.append("run provenance does not match the pinned model backend contract")
    if stored.get("isolation") != "tool_api_enforced":
        contamination.append(
            "native model execution is not an enforced confidentiality boundary"
        )
    if not _uses_disclosed_model_lane(context):
        contamination.append(
            "run did not use the Claude Opus 5 backend disclosed for NVIDIA's full result"
        )
    if stored.get("canonical_agent_bundle") is not True:
        contamination.append("run used a noncanonical agent prompt bundle")
    if stored.get("canonical_agent_bundle_sha256") != CANONICAL_AGENT_BUNDLE_SHA256:
        errors.append("run provenance does not bind the frozen agent bundle protocol")
    if stored.get("public_roster_protocol_sha256") != EXPECTED_PUBLIC_ROSTER_SHA256:
        errors.append("run provenance does not bind the frozen public roster protocol")

    if stored_repository.get("dirty") is True:
        contamination.append("source repository was dirty when the run was created")
    try:
        current_repository = capture_git_provenance(REPOSITORY_ROOT)
        if current_repository.dirty:
            contamination.append("source repository is currently dirty")
        if (
            current_repository.commit != stored_repository.get("commit")
            or current_repository.diff_sha256 != stored_repository.get("diff_sha256")
        ):
            errors.append("source repository differs from the run's immutable provenance")
    except Exception as error:
        errors.append(f"repository provenance failed: {type(error).__name__}: {error}")

    game_ids = tuple(item.game_id for item in descriptors)
    bundle_dir = _config_path(context, "bundle_dir")
    bundle_report = scan_agent_bundle(bundle_dir, game_ids)
    contamination.extend(
        f"agent bundle {finding.kind.value} at {finding.path}: {finding.detail}"
        for finding in bundle_report.findings
    )
    if bundle_report.clean:
        try:
            current_bundle = capture_agent_bundle(bundle_dir, game_ids)
            if current_bundle.sha256 != stored_bundle.get("sha256"):
                errors.append("agent bundle differs from the run's immutable provenance")
        except Exception as error:
            errors.append(f"agent bundle capture failed: {type(error).__name__}: {error}")
    else:
        errors.append("agent bundle failed the contamination scan")

    try:
        current_cache = capture_cache_manifest(_config_path(context, "cache_dir"))
        if current_cache.sha256 != stored_cache.get("sha256"):
            errors.append("environment cache differs from the run's immutable provenance")
        setup_manifest = _validate_setup_manifest(
            _config_path(context, "cache_dir"),
            descriptors,
        )
        if sha256_json(setup_manifest) != stored.get("setup_manifest_sha256"):
            errors.append("public roster setup manifest differs from run provenance")
    except Exception as error:
        errors.append(f"environment cache capture failed: {type(error).__name__}: {error}")

    try:
        contamination.extend(_budget_ambiguity_contamination(context))
    except Exception as error:
        errors.append(f"budget integrity failed: {type(error).__name__}: {error}")
    return IntegrityComparison(
        contamination=tuple(dict.fromkeys(contamination)),
        errors=tuple(dict.fromkeys(errors)),
    )


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """
    Durably create a JSON claim without a check-then-create race.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _submission_replay_roster(context: RunContext) -> list[dict[str, Any]]:
    """
    Return exact selected replay evidence for artifact binding.
    """

    return [
        {
            "game_id": entry.game_id,
            "actions": entry.actions,
            "levels_completed": entry.levels_completed,
            "win_levels": entry.win_levels,
            "trace_sha256": entry.trace_sha256,
        }
        for entry in CampaignBank(context.directory / "bank.json").entries()
    ]


def _read_submission(context: RunContext) -> SubmissionSummary | None:
    for name in ("competition-submission.json", "dry-run-submission.json"):
        path = context.directory / name
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema",
            "run_id",
            "manifest_sha256",
            "official_arc_base_url",
            "mode",
            "completed_at",
            "summary",
            "replays",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"submission artifact is malformed: {path}")
        mode = name.removesuffix("-submission.json")
        if (
            value["schema"] != "ardea.arc.submission.v2"
            or value["run_id"] != context.manifest.run_id
            or value["manifest_sha256"] != sha256_json(context.manifest.to_dict())
            or value["official_arc_base_url"] != OFFICIAL_ARC_BASE_URL
            or value["mode"] != mode
            or value["replays"] != _submission_replay_roster(context)
            or not isinstance(value["summary"], dict)
        ):
            raise ValueError(f"submission artifact does not match run evidence: {path}")
        summary = SubmissionSummary(**value["summary"])
        artifact_sha256 = sha256_json(value)
        claim_path = context.directory / f"{mode}-submission.claim.json"
        try:
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"submission claim is missing or malformed: {claim_path}") from error
        if (
            not isinstance(claim, dict)
            or claim.get("status") != "completed"
            or claim.get("run_id") != context.manifest.run_id
            or claim.get("mode") != mode
            or claim.get("scorecard_id") != summary.scorecard_id
            or claim.get("artifact_sha256") != artifact_sha256
        ):
            raise ValueError("submission claim does not match the completed artifact")
        anchored = any(
            event["kind"] == "scorecard.completed"
            and event["payload"].get("mode") == mode
            and event["payload"].get("artifact_sha256") == artifact_sha256
            for event in context.events()
        )
        if not anchored:
            raise ValueError("submission artifact is not anchored in the run event chain")
        if summary.mode != mode or not summary.completed:
            raise ValueError("submission summary mode or completion state is inconsistent")
        return summary
    return None


def _validate_and_build_report(
    context: RunContext,
    descriptors: tuple[OfficialGameDescriptor, ...],
    factory: OfficialArcadeFactory,
    *,
    persist: bool,
    submission: SubmissionSummary | None = None,
) -> tuple[CampaignValidation, RunReport]:
    integrity = _integrity_comparison(context, descriptors)
    validation = validate_campaign(
        context,
        descriptors,
        factory,
        contamination=integrity.contamination,
    )
    combined_errors = tuple(dict.fromkeys((*validation.errors, *integrity.errors)))
    validation = replace(
        validation,
        valid=validation.valid and not combined_errors,
        eligible_for_competition=(
            validation.eligible_for_competition
            and not combined_errors
            and context.manifest.mode is RunMode.COLD
            and _uses_disclosed_model_lane(context)
        ),
        errors=combined_errors,
    )
    bank = CampaignBank(context.directory / "bank.json")
    budget = BudgetLedger(
        context.directory,
        max_cost_usd=context.manifest.max_cost_usd,
        pricing=pricing_for_model(context.manifest.model),
    )
    report = build_run_report(
        context,
        bank,
        budget,
        expected_games={item.game_id: item.levels for item in descriptors},
        trace_paths=tuple(context.directory.glob("games/*/attempt-*/trace.jsonl")),
        contamination=validation.contamination,
        fresh_replay_validated=validation.valid,
        validation_errors=validation.errors,
        submission=submission or _read_submission(context),
    )
    if persist and not context.is_sealed:
        write_validation(context, validation)
        write_report(context.directory / "report.json", report)
        context.append_event(
            "validation.completed",
            {
                "valid": validation.valid,
                "eligible_for_competition": validation.eligible_for_competition,
                "solved_games": validation.solved_games,
                "solved_levels": validation.solved_levels,
                "rhae_percent": validation.board_rhae_percent,
            },
        )
    return validation, report


def _report_summary(report: RunReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "mode": report.mode,
        "status": report.status.value,
        "games": f"{report.solved_games}/{report.expected_games}",
        "levels": f"{report.solved_levels}/{report.expected_levels}",
        "rhae_percent": report.rhae_percent,
        "submitted_actions": report.submitted_actions,
        "exploratory_actions": report.exploratory_actions,
        "estimated_cost_usd": report.usage.estimated_cost_usd,
        "competition_acceptance_met": report.submission.acceptance_met,
        "validation_errors": list(report.validation.errors),
        "contamination": list(report.contamination),
    }


def _command_setup(args: argparse.Namespace) -> int:
    cache_dir = _resolved(args.cache_dir)
    manifest = setup_public_games(cache_dir)
    descriptors = list_official_games(cache_dir, online_catalog=False)
    if len(descriptors) != PRIMARY_GAME_COUNT or sum(item.levels for item in descriptors) != PRIMARY_LEVEL_COUNT:
        raise RuntimeError(
            "downloaded public catalog does not match the 25-game, 183-level protocol"
        )
    roster = _game_roster(descriptors)
    if roster != _canonical_public_roster():
        raise RuntimeError(
            "downloaded catalog does not match the frozen public benchmark roster"
        )
    roster_digest = _public_roster_sha256(
        roster,
        dict(manifest.package_versions),
    )
    if roster_digest != EXPECTED_PUBLIC_ROSTER_SHA256:
        raise RuntimeError(
            "downloaded catalog does not match the frozen public benchmark roster"
        )
    setup_document = {
        "schema": SETUP_MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "official_arc_base_url": OFFICIAL_ARC_BASE_URL,
        "package_versions": dict(manifest.package_versions),
        "game_roster": roster,
        "game_roster_sha256": roster_digest,
        "cache_sha256": capture_cache_manifest(cache_dir).sha256,
    }
    setup_path = _setup_manifest_path(cache_dir)
    atomic_write_json(setup_path, setup_document)
    os.chmod(setup_path, 0o444)
    print(
        json.dumps(
            {
                "environments_dir": manifest.environments_dir,
                "game_count": len(manifest.game_ids),
                "file_count": len(manifest.files),
                "aggregate_sha256": manifest.aggregate_sha256,
                "game_roster_sha256": setup_document["game_roster_sha256"],
                "setup_manifest": str(setup_path),
                "package_versions": dict(manifest.package_versions),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    checks = run_checks(_resolved(args.cache_dir), args.backend or "codex-oauth")
    for check in checks:
        marker = "PASS" if check.ok else ("WARN" if not check.required else "FAIL")
        print(f"{marker:4}  {check.name:16} {check.detail}")
    return 0 if checks_pass(checks) else 2


def _validate_existing_traces(
    context: RunContext,
    factory: OfficialArcadeFactory,
) -> None:
    """
    Replay every durable attempt before a resumed run can spend or mutate.
    """

    for trace_path in sorted(context.directory.glob("games/*/attempt-*/trace.jsonl")):
        resolved = trace_path.resolve(strict=True)
        if trace_path.is_symlink() or not resolved.is_relative_to(context.directory.resolve()):
            raise RuntimeError(f"existing trace is outside the run boundary: {trace_path}")
        validate_replay(resolved, factory)


def _run_campaign_under_lease(
    args: argparse.Namespace,
    manager: ResultsManager,
    context: RunContext,
    cache_dir: Path,
    bundle_dir: Path,
    descriptors: tuple[OfficialGameDescriptor, ...],
) -> int:
    """
    Execute one ARC campaign while its caller holds exclusive run ownership.
    """

    budget = BudgetLedger(
        context.directory,
        max_cost_usd=context.manifest.max_cost_usd,
        pricing=pricing_for_model(context.manifest.model),
    )
    if args.resume is not None:
        integrity = _integrity_comparison(context, descriptors)
        if integrity.errors:
            raise RuntimeError(
                "resume integrity preflight failed: " + "; ".join(integrity.errors)
            )
        _validate_existing_traces(context, OfficialArcadeFactory(cache_dir))
        _record_unreconciled_reservations(context, budget)
    if context.manifest.mode is RunMode.WARM:
        import_warm_memory(context, manager.root)

    config = context.manifest.config
    jobs = int(config["jobs"])
    if _uses_disclosed_model_lane(context):
        _validate_campaign_jobs(context.manifest.backend, jobs)
    if context.manifest.backend == "codex-oauth":
        driver: EpisodeDriver = CodexEpisodeDriver(budget, cache_dir)
    elif context.manifest.backend == "openai-api":
        driver = OpenAIEpisodeDriver(budget, cache_dir)
    elif context.manifest.backend == "anthropic-api":
        driver = AnthropicEpisodeDriver(budget, cache_dir)
    else:
        raise ValueError(f"unsupported run backend: {context.manifest.backend}")
    attempts_per_game, episodes_per_attempt = _campaign_limits(context)
    runner = CampaignRunner(
        context,
        descriptors,
        OfficialArcadeFactory(cache_dir),
        driver,
        bundle_dir=bundle_dir,
        attempts_per_game=attempts_per_game,
        episodes_per_attempt=episodes_per_attempt,
        jobs=jobs,
    )
    runner.run()
    validation, report = _validate_and_build_report(
        context,
        descriptors,
        OfficialArcadeFactory(cache_dir),
        persist=True,
    )
    print(json.dumps(_report_summary(report), indent=2, sort_keys=True))
    print(f"results: {context.directory}")
    return 0 if validation.eligible_for_competition else 1


def _command_campaign(args: argparse.Namespace) -> int:
    manager = ResultsManager(_resolved(args.results_root))
    if args.resume is not None:
        preview = manager.resume(args.resume)
        with RunLease(preview.directory):
            context = _resume_arc_context(args, manager)
            cache_dir = _config_path(context, "cache_dir")
            bundle_dir = _config_path(context, "bundle_dir")
            descriptors = _official_catalog(cache_dir)
            _backend_preflight(cache_dir, context.manifest.backend)
            return _run_campaign_under_lease(
                args,
                manager,
                context,
                cache_dir,
                bundle_dir,
                descriptors,
            )

    cache_dir = _resolved(args.cache_dir)
    bundle_dir = _resolved(args.bundle_dir)
    descriptors = _official_catalog(cache_dir)
    context = _new_arc_context(args, manager, descriptors, cache_dir, bundle_dir)
    with RunLease(context.directory):
        return _run_campaign_under_lease(
            args,
            manager,
            context,
            cache_dir,
            bundle_dir,
            descriptors,
        )


def _open_run(args: argparse.Namespace) -> tuple[RunContext, tuple[OfficialGameDescriptor, ...], OfficialArcadeFactory]:
    context = ResultsManager(_resolved(args.results_root)).open(args.run_id)
    if context.manifest.config.get("target") != "arc-agi-3-public":
        raise ValueError("run is not an ARC-AGI-3 campaign")
    cache_dir = _config_path(context, "cache_dir")
    descriptors = _official_catalog(cache_dir)
    return context, descriptors, OfficialArcadeFactory(cache_dir)


def _command_validate(args: argparse.Namespace) -> int:
    preview = ResultsManager(_resolved(args.results_root)).open(args.run_id)
    with RunLease(preview.directory):
        context, descriptors, factory = _open_run(args)
        validation, report = _validate_and_build_report(
            context,
            descriptors,
            factory,
            persist=not context.is_sealed,
        )
        print(json.dumps(_report_summary(report), indent=2, sort_keys=True))
        return 0 if validation.eligible_for_competition else 1


def _command_report(args: argparse.Namespace) -> int:
    preview = ResultsManager(_resolved(args.results_root)).open(args.run_id)
    with RunLease(preview.directory):
        context, descriptors, factory = _open_run(args)
        _validation, report = _validate_and_build_report(
            context,
            descriptors,
            factory,
            persist=False,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0


def _human_confirmation(run_id: str, actions: int) -> bool:
    prompt = (
        f"Competition mode will open one official scorecard for {run_id} and replay {actions} actions. "
        "This external action cannot be undone. Type YES to continue: "
    )
    try:
        with Path("/dev/tty").open("r+", encoding="utf-8") as terminal:
            terminal.write(prompt)
            terminal.flush()
            return terminal.readline().strip() == "YES"
    except OSError as error:
        raise RuntimeError("Competition submission requires a controlling terminal") from error


def _attribute(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _submission_summary(scorecard: Any) -> SubmissionSummary:
    response = scorecard.official_response
    score = _attribute(response, "score")
    games = _attribute(response, "total_environments_completed")
    levels = _attribute(response, "total_levels_completed")
    actions = _attribute(response, "total_actions")
    summary = SubmissionSummary.from_scorecard(
        scorecard,
        official_rhae_percent=float(score) if score is not None else None,
        official_games_solved=int(games) if games is not None else None,
        official_levels_solved=int(levels) if levels is not None else None,
    )
    if actions is not None:
        summary = replace(summary, official_submitted_actions=int(actions))
    return summary


def _command_compete_under_lease(args: argparse.Namespace) -> int:
    context, descriptors, factory = _open_run(args)
    if context.is_sealed:
        raise RuntimeError("sealed parent runs cannot create scorecards")
    mode = ScorecardMode.DRY_RUN if args.dry_run else ScorecardMode.COMPETITION
    artifact = context.directory / f"{mode.value}-submission.json"
    claim_path = context.directory / f"{mode.value}-submission.claim.json"
    if artifact.exists() or claim_path.exists():
        raise RuntimeError(
            f"this run already has a {mode.value} submission or pending claim; "
            "automatic retry is refused"
        )
    validation, _report = _validate_and_build_report(
        context,
        descriptors,
        factory,
        persist=True,
    )
    if not validation.valid:
        raise RuntimeError("submission requires a complete, fresh-replay-valid campaign")
    if mode is ScorecardMode.COMPETITION and not validation.eligible_for_competition:
        raise RuntimeError("Competition submission requires a clean cold run at local 100.00 RHAE")
    if not os.environ.get("ARC_API_KEY"):
        raise RuntimeError("ARC_API_KEY is required for an online scorecard")
    confirmed = mode is ScorecardMode.DRY_RUN or _human_confirmation(
        context.manifest.run_id,
        validation.submitted_actions,
    )
    if not confirmed:
        raise RuntimeError("Competition submission cancelled; confirmation was not exactly YES")
    replay_roster = _submission_replay_roster(context)
    claim: dict[str, Any] = {
        "schema": "ardea.arc.submission-claim.v1",
        "run_id": context.manifest.run_id,
        "manifest_sha256": sha256_json(context.manifest.to_dict()),
        "official_arc_base_url": OFFICIAL_ARC_BASE_URL,
        "mode": mode.value,
        "status": "claimed",
        "claimed_at": utc_now(),
        "scorecard_id": None,
        "artifact_sha256": None,
        "replays_sha256": sha256_json(replay_roster),
    }
    _write_exclusive_json(claim_path, claim)
    try:
        context.append_event(
            "scorecard.claimed",
            {
                "mode": mode.value,
                "claim_sha256": sha256_json(claim),
                "replays_sha256": claim["replays_sha256"],
            },
        )

        def record_opened(scorecard_id: str) -> None:
            claim.update(
                {
                    "status": "opened",
                    "opened_at": utc_now(),
                    "scorecard_id": scorecard_id,
                }
            )
            atomic_write_json(claim_path, claim)
            context.append_event(
                "scorecard.opened",
                {"mode": mode.value, "scorecard_id": scorecard_id},
            )

        scorecard = submit_scorecard(
            [game.trace_path for game in validation.games],
            _config_path(context, "cache_dir"),
            mode=mode,
            local_factory=factory,
            human_confirmed=confirmed,
            expected_game_count=PRIMARY_GAME_COUNT,
            on_scorecard_opened=record_opened,
        )
        summary = _submission_summary(scorecard)
        document = {
            "schema": "ardea.arc.submission.v2",
            "run_id": context.manifest.run_id,
            "manifest_sha256": sha256_json(context.manifest.to_dict()),
            "official_arc_base_url": OFFICIAL_ARC_BASE_URL,
            "mode": mode.value,
            "completed_at": utc_now(),
            "summary": asdict(summary),
            "replays": [asdict(replay) for replay in scorecard.replays],
        }
        if document["replays"] != replay_roster:
            raise RuntimeError("official replay evidence differs from the selected campaign bank")
        atomic_write_json(artifact, document)
        artifact_sha256 = sha256_json(document)
        context.append_event(
            "scorecard.completed",
            {
                "mode": mode.value,
                "scorecard_id": summary.scorecard_id,
                "artifact_sha256": artifact_sha256,
                "official_rhae_percent": summary.official_rhae_percent,
                "official_games_solved": summary.official_games_solved,
                "official_levels_solved": summary.official_levels_solved,
            },
        )
        claim.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "artifact_sha256": artifact_sha256,
            }
        )
        atomic_write_json(claim_path, claim)
    except Exception as error:
        claim.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "failure_type": type(error).__name__,
            }
        )
        with suppress(Exception):
            atomic_write_json(claim_path, claim)
        with suppress(Exception):
            context.append_event(
                "scorecard.failed",
                {
                    "mode": mode.value,
                    "scorecard_id": claim.get("scorecard_id"),
                    "failure_type": type(error).__name__,
                },
            )
        raise
    _validation, report = _validate_and_build_report(
        context,
        descriptors,
        factory,
        persist=True,
        submission=summary,
    )
    print(json.dumps(_report_summary(report), indent=2, sort_keys=True))
    return 0 if mode is ScorecardMode.DRY_RUN or report.submission.acceptance_met else 1


def _command_compete(args: argparse.Namespace) -> int:
    """
    Hold exclusive ownership across validation and the irreversible scorecard.
    """

    preview = ResultsManager(_resolved(args.results_root)).open(args.run_id)
    with RunLease(preview.directory):
        return _command_compete_under_lease(args)


def _command_evolve_impl(args: argparse.Namespace) -> int:
    from ardea_avo.evolve import EvolutionLayout, resume_evolution, start_evolution

    target_source = _resolved(args.target)
    target = load_target(target_source, require_inputs=args.resume is None)
    target_sha256 = hashlib.sha256(target_source.read_bytes()).hexdigest()
    manager = ResultsManager(_resolved(args.results_root))
    maximum = getattr(args, "max_cost_usd", None)
    attempts = int(getattr(args, "attempts", 3))
    requested_backend = getattr(args, "backend", None)

    evaluator_cwd: Path | None = None
    if args.resume is not None:
        context = manager.resume(args.resume)
        if context.manifest.config.get("target") != "generic-avo":
            raise ValueError("run is not a generic AVO evolution")
        if context.manifest.config.get("target_yaml_sha256") != target_sha256:
            raise ValueError("target YAML differs from the immutable run manifest")
        backend_name = context.manifest.backend
        if requested_backend is not None and requested_backend != backend_name:
            raise ValueError("resume cannot change model backend")
        _backend_preflight(_resolved(args.cache_dir), backend_name)
        effective_target = _effective_generic_target(target, context.manifest.config)
        raw_evaluator_cwd = context.manifest.config.get("evaluator_cwd")
        if isinstance(raw_evaluator_cwd, str):
            evaluator_cwd = Path(raw_evaluator_cwd)
        layout = EvolutionLayout.for_run(context.directory)
        definition = json.loads(layout.definition.read_text(encoding="utf-8"))
        definition_sha256 = sha256_json(definition)
        initialized = [
            event
            for event in context.events()
            if event["kind"] == "evolution.initialized"
        ]
        if (
            len(initialized) != 1
            or initialized[0]["payload"].get("definition_sha256")
            != definition_sha256
        ):
            raise ValueError("evolution definition is not anchored by the outer run event chain")
        factory = _generic_backend_factory(backend_name)
        run = resume_evolution(
            effective_target,
            context.directory,
            backend_factory=factory,
            max_cost_usd=maximum,
            pricing=pricing_for_model(context.manifest.model),
            evaluator_cwd=evaluator_cwd,
        )
        if run.released_reservations:
            context.append_event(
                "budget.reservations_recovered",
                {
                    "count": len(run.released_reservations),
                    "provider_usage_may_be_unreported": True,
                },
            )
        return _advance_evolution(context, run, attempts)
    else:
        maximum = maximum or DEFAULT_MAX_COST_USD
        effective_target = target
        knowledge_paths = list(target.knowledge)
        parent: RunContext | None = None
        if args.results is not None:
            parent = manager.open(args.results)
            if parent.manifest.config.get("target") != "generic-avo":
                raise ValueError("warm parent is not a generic AVO evolution")
            if parent.manifest.config.get("target_yaml_sha256") != target_sha256:
                raise ValueError("warm parent uses a different target YAML")
            parent_layout = EvolutionLayout.for_run(parent.directory)
            parent_budget = BudgetLedger(parent_layout.budget)
            if parent_budget.active_reservations():
                raise RuntimeError("warm parent has active model reservations")
            effective_target = target.model_copy(update={"seed": parent_layout.workspace})
            steps_path = parent_layout.host / "evolution-steps.jsonl"
            if steps_path.exists():
                knowledge_paths.append(steps_path)
                effective_target = effective_target.model_copy(
                    update={"knowledge": tuple(knowledge_paths)}
                )
            evaluator_cwd = _parent_evaluator_cwd(parent_layout)
        backend_name = requested_backend or (
            parent.manifest.backend if parent is not None else "codex-oauth"
        )
        _backend_preflight(_resolved(args.cache_dir), backend_name)
        auth_method = _auth_for_backend(backend_name)
        model = _model_for_backend(backend_name)
        config = {
            "target": "generic-avo",
            "target_yaml": str(target_source),
            "target_yaml_sha256": target_sha256,
            "effective_seed": str(effective_target.seed),
            "knowledge": [str(path) for path in effective_target.knowledge],
            "evaluator_cwd": str(evaluator_cwd) if evaluator_cwd is not None else None,
        }
        provenance = {"repository": capture_git_provenance(REPOSITORY_ROOT).to_dict()}
        slug = getattr(args, "slug", "evolve")
        if args.cold:
            context = manager.create_cold(
                slug,
                backend=backend_name,
                auth_method=auth_method,
                model=model,
                max_cost_usd=maximum,
                config=config,
                provenance=provenance,
            )
        else:
            assert args.results is not None
            context = manager.create_warm(
                args.results,
                slug,
                backend=backend_name,
                auth_method=auth_method,
                model=model,
                max_cost_usd=maximum,
                config=config,
                provenance=provenance,
            )
        with RunLease(context.directory):
            run = start_evolution(
                effective_target,
                context.directory,
                backend_factory=_generic_backend_factory(backend_name),
                max_cost_usd=maximum,
                pricing=pricing_for_model(model),
                evaluator_cwd=evaluator_cwd,
            )
            context.append_event(
                "evolution.initialized",
                {
                    "candidate_id": run.state.accepted_candidate.candidate_id,
                    "definition_sha256": sha256_json(
                        json.loads(run.layout.definition.read_text(encoding="utf-8"))
                    ),
                },
            )
            return _advance_evolution(context, run, attempts)


def _advance_evolution(context: RunContext, run: Any, attempts: int) -> int:
    """
    Advance and report a generic evolution under an exclusive run lease.
    """

    exhausted = False
    try:
        state = run.advance(attempts)
    except BudgetExceeded:
        exhausted = True
        state = run.state
    snapshot = run.budget.snapshot()
    report = {
        "schema": "ardea.avo.generic-report.v1",
        "run_id": context.manifest.run_id,
        "mode": context.manifest.mode.value,
        "parent_run_id": context.manifest.parent_run_id,
        "target": state.target_name,
        "attempts": state.attempts,
        "accepted_candidate": state.accepted_candidate.model_dump(mode="json"),
        "accepted_evaluation": state.accepted_evaluation.model_dump(mode="json"),
        "accepted_commit": state.accepted_commit,
        "last_decision": state.records[-1].decision.value if state.records else "seed",
        "budget_exhausted": exhausted,
        "spent_usd": str(snapshot.spent_usd),
        "max_cost_usd": str(snapshot.cap_usd),
    }
    atomic_write_json(context.directory / "generic-report.json", report)
    context.append_event(
        "evolution.completed",
        {
            "attempts": state.attempts,
            "accepted_candidate": state.accepted_candidate.candidate_id,
            "budget_exhausted": exhausted,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"results: {context.directory}")
    return 1 if exhausted else 0


def _command_evolve(args: argparse.Namespace) -> int:
    """
    Establish exclusive recovery ownership before resuming generic evolution.
    """

    if args.resume is not None:
        preview = ResultsManager(_resolved(args.results_root)).resume(args.resume)
        with RunLease(preview.directory):
            return _command_evolve_impl(args)
    return _command_evolve_impl(args)


def _generic_backend_factory(backend: str) -> Any:
    if backend == "codex-oauth":
        return lambda ledger: CodexOAuthBackend(ledger)
    if backend == "openai-api":
        return lambda ledger: OpenAIResponsesBackend(ledger)
    if backend == "anthropic-api":
        return lambda ledger: AnthropicMessagesBackend(ledger)
    raise ValueError(f"unsupported generic backend: {backend}")


def _effective_generic_target(target: TargetFile, config: Mapping[str, Any]) -> TargetFile:
    seed = config.get("effective_seed")
    knowledge = config.get("knowledge")
    if not isinstance(seed, str) or not isinstance(knowledge, list) or not all(
        isinstance(item, str) for item in knowledge
    ):
        raise ValueError("generic run manifest lacks its effective target paths")
    return target.model_copy(
        update={
            "seed": Path(seed),
            "knowledge": tuple(Path(item) for item in knowledge),
        }
    )


def _parent_evaluator_cwd(layout: Any) -> Path:
    definition = json.loads(layout.definition.read_text(encoding="utf-8"))
    value = definition.get("evaluator_cwd")
    if value == "snapshot":
        return layout.evaluator_snapshot
    if not isinstance(value, str) or not value:
        raise ValueError("parent evolution has invalid evaluator provenance")
    return Path(value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse one command, execute it, and return a process exit status.
    """

    _load_local_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            return _command_setup(args)
        if args.command == "doctor":
            return _command_doctor(args)
        if args.command == "validate":
            return _command_validate(args)
        if args.command == "report":
            return _command_report(args)
        if args.command == "compete":
            return _command_compete(args)
        if args.command == "evolve":
            return _command_evolve(args)
        if not (args.cold or args.results or args.resume):
            parser.error("an ARC campaign requires exactly one of --cold, --results, or --resume")
        return _command_campaign(args)
    except (IntegrityError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
