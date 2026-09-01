# ARC-AGI-3 benchmark protocol

## Primary claim

A qualifying result must originate from `app --cold --backend anthropic-api`, use the frozen default
agent bundle and `claude-opus-5`, start from a clean committed repository with no parent knowledge or
sessions, and finish with an accepted official Competition scorecard containing all 25 public games,
all 183 levels, and exactly 100.00 RHAE. Complete replay evidence below 100 remains a fallback and does
not produce a successful campaign or validation exit status.

The disclosed-model backend pins adaptive thinking, maximum effort, global inference, 65,536 maximum
output tokens, ephemeral five-minute prompt caching, serial mutating tools, and zero automatic SDK
retries. It also requires `--jobs 1`, making an ambiguity stop global to the qualifying campaign.
NVIDIA disclosed the model and text-only 64×64 observation format but not its exact prompt,
reasoning setting, memory representation, or runtime; these local choices are reproduction protocol,
not claimed NVIDIA source.

The exact public roster is committed in `targets/arc_agi_3/public_roster.json`. It is anchored to
official published scorecard `9fb9db8d-3734-4885-987a-a250445c0690`, `arc-agi==0.9.9`,
`arcengine==0.9.3`, and roster digest
`19611e0ad29479c9fd84b759d0468ac7293830d9a6db02e4c03c0275828316da`. Any version rotation requires an
explicit protocol update.

## Observations and actions

- The authoritative observation is the last settled 64×64 frame serialized as hexadecimal cell values.
- Images are disabled unless the run is explicitly marked as an image ablation.
- Only `play` advances the environment. Inspection, history, diff, segmentation, and memory tools do not.
- Invalid payloads and host-rejected actions return protocol-level tool errors and never step the engine.
- Human baselines remain host-only and are used solely for scoring and action limits.
- Per-attempt action budget is `min(5 × sum(level baselines), 2500)`.
- The official SDK client and its process-global level-reset switch are serialized and restored around
  each affected operation, so parallel model calls cannot corrupt level state.

## Scoring

For a completed level with one or more charged actions, its score is
`min((human_baseline_actions / agent_actions)^2, 1.15)`. A zero-action completed level scores zero,
matching `arc-agi==0.9.9`. Game scoring applies the official one-indexed level weights,
incomplete-level ceiling, and 100-point game cap. Board RHAE is the mean over the frozen roster. The
strictly validated official server response is authoritative for a submitted scorecard.

## Cold, warm, and resume

- Cold runs import no memory, traces, workspaces, candidate trees, or provider sessions.
- ARC warm children import only approved verified/falsified records whose evidence hashes replay.
  Sealing snapshots every durable parent artifact, including SQLite WAL state.
- Resume validates dependencies, roster, prompt bundle, source revision, environment cache, manifests,
  checkpoints, event chains, provider transcript contracts, and trace replay before further spend.
- Budget ceilings, attempts, and episodes can only increase through append-only revision events.
- An unreceipted provider outcome stops new model work, retains its full worst-case budget hold, and
  permanently makes that run nonqualifying; a research resume can use only the remaining capacity.

## Submission

Offline replay is mandatory and a normal online dry run is strongly recommended. Competition mode
refuses warm, incomplete, dirty, contaminated, divergent, sub-100, or previously claimed runs. It reads
literal `YES` from a controlling terminal, writes an exclusive artifact-bound claim, then opens exactly
one scorecard. That claim is never automatically retried after an ambiguous failure.

NVIDIA's reported 6,624 actions are a separate stretch comparison. The acceptance threshold here is the
official 100.00/25/183 outcome, not matching that action count.
