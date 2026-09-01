# Paper-to-code map

The pinned source is [*AVO: Agentic Variation Operators for Autonomous Evolutionary Search*,
arXiv:2603.24517v1](https://arxiv.org/abs/2603.24517v1), submitted March 25, 2026. It defines AVO as an
agentic replacement for a conventional variation operator: `Vary(P_t) = Agent(P_t, K, f)`. The paper
reports an NVIDIA B200 attention-kernel experiment, not ARC-AGI-3.

The ARC adaptation is described separately in NVIDIA's August 21, 2026
[technical report](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/).
That report discloses Claude Opus 5, text-only 64×64 grids, persistent memory, supervision, 25/25 games,
183/183 levels, 100.00 RHAE, and 6,624 actions, but not reproducible source or exact prompts.

| Paper mechanism | Implementation |
| --- | --- |
| Candidate population supplied to an agent | The accepted candidate tree, lineage, knowledge, and failed-attempt evidence form each variation request |
| Autonomous inspect/edit/evaluate/repair loop | `EvolutionEngine` delegates proposal work to `AgentBackend`; the host performs authoritative evaluation |
| Correctness-gated promotion | Protocol validity and correctness are mandatory before comparison |
| Match-or-improve commits | The configured lexicographic comparator accepts exact equality |
| Single committed lineage | The lineage store commits the complete candidate tree and score metadata |
| Failed exploration retained internally | Rejected patches, evaluation payloads, and hypotheses are archived outside the accepted branch |
| Persistent context | Backend session identifiers plus durable evidence-linked memory survive compaction and restarts |
| Stagnation supervisor | Host-owned deterministic triggers request exactly three distinct directions; the main agent chooses |
| External evaluator | Target evaluators execute outside the candidate directory and bind evaluations to tree digests |

The paper does not publish its prompts, exact model, memory schema, compaction policy, supervisor
thresholds, evaluator payloads, or budget policy. The later ARC report adds a model identity but still
does not publish those implementation details. Local choices are explicit and versioned here rather
than presented as recovered NVIDIA code.

The ARC campaign runner shares backend, memory, supervisor, budget, receipt, and replay services with
the generic engine. It is not described as the paper's attention-kernel lineage. ARC bundles evolved
against public-game traces are warm artifacts and remain separate from the primary state-cold run.
