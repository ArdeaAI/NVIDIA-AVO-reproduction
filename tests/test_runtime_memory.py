"""
Tests for evidence-linked memory and warm imports.
"""

import pytest

from ardea_avo.runtime.memory import MemoryStatus, MemoryStore

EVIDENCE = "a" * 64
CONTRADICTION = "b" * 64


def test_resolved_memory_requires_sha256_evidence(tmp_path) -> None:
    """
    Verified and falsified claims cannot be ungrounded prose.
    """

    with MemoryStore(tmp_path / "memory.sqlite", run_id="cold") as store:
        with pytest.raises(ValueError, match="supporting evidence"):
            store.add(scope="game", claim="door opens", status="verified")
        with pytest.raises(ValueError, match="SHA-256"):
            store.add(
                scope="game",
                claim="door opens",
                status="verified",
                evidence=("frame-1",),
            )


def test_hypothesis_resolution_and_approval(tmp_path) -> None:
    """
    A hypothesis becomes warm-eligible only after an evidence-backed review.
    """

    with MemoryStore(tmp_path / "memory.sqlite", run_id="cold") as store:
        hypothesis = store.add(scope="game", claim="red is hazardous")
        with pytest.raises(ValueError):
            store.approve_for_warm(hypothesis.id)
        resolved = store.resolve(
            hypothesis.id,
            status=MemoryStatus.FALSIFIED,
            contradictions=(CONTRADICTION,),
            confidence=0.9,
        )
        assert resolved.status is MemoryStatus.FALSIFIED
        approved = store.approve_for_warm(resolved.id)
        assert approved.approved_for_warm
        assert store.search("red hazardous") == (approved,)


def test_warm_import_copies_only_approved_validated_resolved_records(tmp_path) -> None:
    """
    Warm children receive claims, not live state, sessions, or hypotheses.
    """

    with MemoryStore(tmp_path / "parent.sqlite", run_id="parent") as parent:
        verified = parent.add(
            scope="game",
            scope_id="game-1",
            claim="green toggles the gate",
            status="verified",
            evidence=(EVIDENCE,),
        )
        parent.approve_for_warm(verified.id)
        parent.add(scope="game", claim="unverified idea")
        unapproved = parent.add(
            scope="game",
            claim="not reviewed",
            status="verified",
            evidence=(CONTRADICTION,),
        )
        assert not unapproved.approved_for_warm

        with MemoryStore(tmp_path / "child.sqlite", run_id="child") as child:
            imported = child.import_from_parent(
                parent, evidence_validator=lambda digest: digest == EVIDENCE
            )
            assert len(imported) == 1
            assert imported[0].claim == verified.claim
            assert imported[0].id != verified.id
            assert imported[0].origin_record_id == verified.id
            assert imported[0].imported_from_run == "parent"
            assert not imported[0].approved_for_warm
            assert child.import_from_parent(
                parent, evidence_validator=lambda _: True
            ) == ()


def test_database_is_bound_to_one_run(tmp_path) -> None:
    """
    Reopening a memory database under another run id is rejected.
    """

    path = tmp_path / "memory.sqlite"
    MemoryStore(path, run_id="first").close()
    with pytest.raises(ValueError, match="does not match"):
        MemoryStore(path, run_id="second")


def test_read_only_parent_store_cannot_mutate_database(tmp_path) -> None:
    """
    Warm imports read a sealed parent through SQLite's immutable read-only mode.
    """

    path = tmp_path / "parent.sqlite"
    MemoryStore(path, run_id="parent").close()
    before = path.read_bytes()
    with MemoryStore(path, run_id="parent", read_only=True) as store:
        assert store.list() == ()
        with pytest.raises(RuntimeError, match="read-only"):
            store.add(scope="run", claim="cannot write")
    assert path.read_bytes() == before
