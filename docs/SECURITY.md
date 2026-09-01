# Security and experiment integrity

## ARC qualification boundary

In the Anthropic and OpenAI API lanes, the model receives only host-defined ARC and memory tools. It
does not receive a shell, the official SDK, cache paths, evaluator internals, human baselines, or
credentials. Tool schemas disallow parallel mutating actions; the host validates inputs, performs the
single action, commits the trace durably, and returns the observation. Provider SDK retries are disabled.

The official SDK and scorecard objects remain host-owned. A shared lock serializes SDK mutations, and
the process-global level-reset setting is scoped and restored even on exceptions. One run lease prevents
concurrent resume, validation, reporting, sealing, or submission against the same directory.

Codex CLI runs natively with workspace-write restrictions. That reduces accidental mutation but is not
a certified confidentiality boundary: a native process can potentially observe files permitted by the
operating system. Reports therefore label it `native_best_effort`, and it cannot qualify for the primary
claim.

## Credentials and external side effects

The harness reads `ARC_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` only in the host process and
never stores or forwards them through prompts, manifests, traces, or memory. Competition submission is
the only irreversible external operation. It requires terminal confirmation and an exclusive one-shot
claim bound to the exact replay roster. An ambiguous failure after the claim requires manual forensic
reconciliation; automatic replay could create a second billable scorecard.

ARC campaign provider responses are charged to the append-only ledger before another round begins,
with a conservative worst-case reservation. A connection failure before a response receipt is inherently
ambiguous: the provider may have completed a billable request that the host could not record. Recovery
marks that possibility rather than pretending the ledger is complete. The current campaign invocation
stops, and the full worst-case reservation stays held against the lifetime ceiling. A resumed run can
continue only with its remaining capacity and is permanently excluded from the primary claim.

## Durable evidence

Manifests are immutable, events and budget records are hash-chained, checkpoints are atomically
replaced, provider transcript contracts are digested, and warm-parent seals cover every durable file and
symbolic link. Parent memory is read through immutable SQLite mode and checked before and after import.
These hashes detect mutation; they are not signatures or independent third-party attestations.

Raw action traces and learned manuals remain local and Git-ignored because they are replayable
public-game solutions. Fresh replay through the exact pinned environment plus the official scorecard is
the empirical validation boundary.

## Generic AVO targets

Generic evolution is a trusted-code workflow, not the ARC clean boundary. File tools reject traversal,
Git metadata, and symbolic links, but the configured evaluator and `run_check` execute candidate code as
native host subprocesses. A candidate can therefore access operating-system resources available to the
user. Do not run generic targets from untrusted seeds or on a host containing secrets; use an external
VM/container sandbox when that threat model matters.
