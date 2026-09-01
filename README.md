# NVIDIA AVO reproduction for ARC-AGI-3

This repository implements an auditable AVO-style harness for the 25-game, 183-level ARC-AGI-3
public set. The primary reproduction lane uses `claude-opus-5`, the model in
[NVIDIA's reported 100.00 RHAE run](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/).
NVIDIA reports 25/25 games, 183/183 levels, and 6,624 submitted actions.

The underlying [AVO paper](https://arxiv.org/abs/2603.24517v1) is an NVIDIA attention-kernel study,
not an ARC experiment. NVIDIA has not published the exact ARC prompts, runtime,
or artifacts, so this project distinguishes architectural fidelity, model parity, and an independently
verified benchmark result.

## What is implemented

- A domain-neutral AVO engine with a scored seed, correctness-gated single lineage, full-tree Git
  commits, exact-tie acceptance, rejected-patch archives, supervisor redirects, and recovery.
- An ARC adapter with exact text grids, one counted action surface, evidence-bearing traces,
  deterministic replay, official RHAE parity, multi-attempt search, and best-trace banking. A sub-100
  `WIN` is retained as fallback evidence while search continues for exact 100.
- A direct Anthropic Messages backend pinned to `claude-opus-5`, adaptive thinking, maximum effort,
  global inference, serial host tools, prompt caching, streamed long responses, and durable transcripts.
- Optional OpenAI Responses and native Codex lanes. OpenAI is a supported alternative; native Codex is
  exploratory and is deliberately marked nonqualifying because its filesystem boundary is best effort.
- Evidence-linked SQLite memory, immutable cold/warm provenance, whole-run artifact seals, append-only
  limit and budget revisions, ARC campaign cost reservations, exclusive leases, and crash-resumable
  campaigns.
- A host-only benchmark snapshot fixing the exact public roster and SDK versions. Endpoint drift fails
  closed instead of silently redefining the benchmark.

No official campaign was run in this development environment because the ARC cache and ARC API key were
unavailable. This code therefore makes no independent 100% claim. A clean cold local 100 makes a run
eligible for Competition submission; the end-to-end claim exists only after the official scorecard is
accepted at 100.00 RHAE, 25/25 games, and 183/183 levels.

## Install

Python 3.12 and `uv` are required.

```bash
uv sync --all-groups --locked
cp .env.example .env
```

Set `ARC_API_KEY` and `ANTHROPIC_API_KEY` in `.env`, then prepare and inspect the frozen environment:

```bash
uv run app setup
uv run app doctor --backend anthropic-api
```

`app setup` downloads through the pinned official endpoint and accepts only the committed 25-game,
183-level roster. Credentials are never written to run artifacts or passed to the model.

## Run the disclosed-model lane

A qualifying API run must start from a committed, clean tree. Commit intentional changes, then confirm
that `git status --porcelain` prints nothing. Choose the cost ceiling deliberately: the following $1,500
value is a hard lifetime ceiling, not a cost forecast or a guarantee of completion.

```bash
git status --porcelain
uv run app --cold --backend anthropic-api --slug opus5-cold \
  --max-cost-usd 1500 --attempts 3 --episodes-per-attempt 12 --jobs 1
```

The default $20 ceiling is useful for smoke tests, not a realistic full public-set campaign. Each
provider round reserves a conservative worst-case allowance before it starts and settles actual usage
afterward. An ambiguous network failure can still represent provider usage that the host never received;
automatic SDK retries are disabled, the campaign stops, and the full reservation remains charged against
the lifetime ceiling. Resuming such a run is allowed for research but remains permanently nonqualifying.
The disclosed-model lane requires one campaign worker so no peer can launch another provider round after
an ambiguous outcome.

Resume the same run with monotonic budget/search extensions:

```bash
uv run app --resume 260901-132026_opus5-cold \
  --max-cost-usd 2000 --attempts 6 --episodes-per-attempt 24
```

Create a warm research child that imports only approved, evidence-backed claims:

```bash
uv run app --results 260901-132026_opus5-cold --backend anthropic-api --slug opus5-warm
```

Warm children never qualify for the primary cold claim. They do not inherit traces, workspaces, provider
sessions, or live environments. Generic AVO warm runs may inherit an accepted candidate tree by design;
that is a separate target contract.

Validate, inspect, and submit:

```bash
uv run app validate 260901-132026_opus5-cold
uv run app report 260901-132026_opus5-cold
uv run app compete 260901-132026_opus5-cold --dry-run
uv run app compete 260901-132026_opus5-cold
```

Campaign and validation success mean the local evidence is eligible for submission. The final command
is the acceptance boundary: its report must show `competition_acceptance_met: true`.

Competition submission requires a literal `YES` on the controlling terminal. The host writes a durable
one-shot claim before opening the scorecard. A crash or ambiguous external failure after that point is
not retried automatically; preserve the run and reconcile it manually with the official service.

Alternative and exploratory lanes are explicit:

```bash
uv run app --cold --backend openai-api
uv run app --cold --backend codex-oauth
uv run app evolve path/to/target.yaml --cold --backend anthropic-api
```

## Integrity terminology

- **Cold** starts without a parent, prior trace, provider session, or game-specific memory.
- **Warm** creates a child from a sealed parent and imports only validated knowledge allowed by the
  target contract.
- **Resume** reopens the same run after validating its manifest, event chain, artifacts, environment,
  budget ledger, checkpoints, and every existing action trace.

The public set is visible and hosted models may have encountered related material during training, so
state-cold does not mean held-out. See the [benchmark protocol](docs/BENCHMARK_PROTOCOL.md),
[paper map](docs/PAPER_MAP.md), [source matrix](docs/SOURCE_MATRIX.md), and
[security model](docs/SECURITY.md).

## Development

```bash
uv run ruff check .
uv run mypy src/ardea_avo
uv run pytest
```

All Python docstrings place their opening and closing triple quotes on separate lines.
