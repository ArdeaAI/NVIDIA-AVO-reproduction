from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ardea_avo.core import (
    Candidate,
    ExternalEvaluator,
    ExternalEvaluatorError,
    StaleArtifactError,
    tree_digest,
)


def _candidate(workspace: Path) -> Candidate:
    return Candidate(
        candidate_id="candidate-1",
        parent_id="v0",
        generation=1,
        artifact_digest=tree_digest(workspace),
    )


def test_external_evaluator_uses_exact_argv_json_contract_and_minimal_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "solution.txt").write_text("answer", encoding="utf-8")
    script = tmp_path / "evaluator.py"
    script.write_text(
        """
import json
import os
import sys

print(json.dumps({
    "candidate_id": sys.argv[1],
    "artifact_digest": sys.argv[2],
    "correct": True,
    "metrics": {"score": 7},
    "evaluator": "fixture-evaluator",
    "evidence": [{"path": "solution.txt"}],
    "metadata": {
        "root_matches": sys.argv[3] == os.environ["AVO_CANDIDATE_ROOT"],
        "secret_leaked": "UNRELATED_SECRET" in os.environ,
    },
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-inherit")
    candidate = _candidate(workspace)
    evaluator = ExternalEvaluator(
        [
            sys.executable,
            str(script),
            "{candidate_id}",
            "{artifact_digest}",
            "{candidate_root}",
        ]
    )

    result = evaluator.evaluate(candidate, workspace)

    assert result.candidate_id == candidate.candidate_id
    assert result.artifact_digest == candidate.artifact_digest
    assert result.score.metrics == {"score": 7.0}
    assert result.metadata == {"root_matches": True, "secret_leaked": False}


@pytest.mark.parametrize(
    ("payload_update", "error"),
    [
        ({"candidate_id": "wrong"}, StaleArtifactError),
        ({"artifact_digest": "0" * 64}, StaleArtifactError),
        ({"metrics": {"score": "7"}}, ExternalEvaluatorError),
        ({"correct": "true"}, ExternalEvaluatorError),
        ({"unexpected": True}, ExternalEvaluatorError),
    ],
)
def test_external_evaluator_rejects_identity_digest_and_schema_drift(
    tmp_path: Path,
    payload_update: dict[str, object],
    error: type[Exception],
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "file").write_text("seed", encoding="utf-8")
    candidate = _candidate(workspace)
    payload: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "artifact_digest": candidate.artifact_digest,
        "correct": True,
        "metrics": {"score": 7},
    }
    payload.update(payload_update)
    evaluator = ExternalEvaluator([sys.executable, "-c", f"print({json.dumps(payload)!r})"])

    with pytest.raises(error):
        evaluator.evaluate(candidate, workspace)


def test_external_evaluator_detects_worktree_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "file").write_text("seed", encoding="utf-8")
    candidate = _candidate(workspace)
    script = tmp_path / "mutating_evaluator.py"
    script.write_text(
        """
import json
import os
from pathlib import Path

Path(os.environ["AVO_CANDIDATE_ROOT"], "file").write_text("mutated", encoding="utf-8")
print(json.dumps({
    "candidate_id": os.environ["AVO_CANDIDATE_ID"],
    "artifact_digest": os.environ["AVO_ARTIFACT_DIGEST"],
    "correct": True,
    "metrics": {"score": 1},
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StaleArtifactError, match="mutated"):
        ExternalEvaluator([sys.executable, str(script)]).evaluate(candidate, workspace)


def test_external_evaluator_reports_invalid_json_and_nonzero_exit(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    candidate = _candidate(workspace)

    with pytest.raises(ExternalEvaluatorError, match="JSON object"):
        ExternalEvaluator([sys.executable, "-c", "print('not-json')"]).evaluate(
            candidate, workspace
        )
    with pytest.raises(ExternalEvaluatorError, match="status 3"):
        ExternalEvaluator(
            [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(3)"]
        ).evaluate(candidate, workspace)


def test_external_evaluator_rejects_duplicate_keys_and_nonstandard_numbers(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    candidate = _candidate(workspace)
    common = (
        f'"candidate_id":"{candidate.candidate_id}",'
        f'"artifact_digest":"{candidate.artifact_digest}",'
        '"correct":true,'
    )
    duplicate = "{" + common + '"metrics":{"score":1},"metrics":{"score":2}}'
    nonstandard = "{" + common + '"metrics":{"score":NaN}}'

    with pytest.raises(ExternalEvaluatorError, match="JSON object"):
        ExternalEvaluator([sys.executable, "-c", f"print({duplicate!r})"]).evaluate(
            candidate, workspace
        )
    with pytest.raises(ExternalEvaluatorError, match="JSON object"):
        ExternalEvaluator([sys.executable, "-c", f"print({nonstandard!r})"]).evaluate(
            candidate, workspace
        )
