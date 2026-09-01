# Reproduction source matrix

The implementation is original code informed by four reproduction repositories and the MetaHarness
runtime linked by openAVO. No source code, prompts, learned game manuals, traces, or artwork from those
projects is copied here.

| Source | Frozen revision and license | Incorporated design lessons | Deliberately excluded |
| --- | --- | --- | --- |
| [agno-agi/arc-agi-arcade](https://github.com/agno-agi/arc-agi-arcade/tree/b83e537da91a688dc7d27b31007ede6273c90e63) | `b83e537da91a688dc7d27b31007ede6273c90e63`; MIT | Official ARC adapter, exact observations, action traces, replay, RHAE, campaign bank, scorecard gate | Unrestricted kernel access, unpublished warm manuals, weak provenance, skipped engine tests |
| [austin1997/AVO](https://github.com/austin1997/AVO/tree/0d36ada5e149f6f86c845f9405381cf09b572d62) | `0d36ada5e149f6f86c845f9405381cf09b572d62`; MIT | Typed configuration, provider boundary, knowledge search, structured logs | Unscored seed, strict tie rejection, incomplete rollback/resume, single-file lineage, unrestricted shell |
| [gatordevin/avo](https://github.com/gatordevin/avo/tree/f6dad9e639d3c9e5d5ac9ccacbe076d82cfa39d2) | `f6dad9e639d3c9e5d5ac9ccacbe076d82cfa39d2`; Apache-2.0 | Target/evaluator contract, full-tree Git lineage, rejection archive, sessions, paper mapping | Prompt-only authority, mutable evaluator state, unsafe backend defaults, incomplete score validation |
| [ruvnet/openAVO](https://github.com/ruvnet/openAVO/tree/e099ec8753b2fdbd3cfa2652f3ea7b68057c82eb) | `e099ec8753b2fdbd3cfa2652f3ea7b68057c82eb`; no license file in that revision | Governance model and architectural documentation | No runtime exists in that checkout |
| [ruvnet/metaharness/packages/avo](https://github.com/ruvnet/metaharness/tree/7a2e7d0edd56af9482ba1d76551b7a87c9b96568/packages/avo) | `7a2e7d0edd56af9482ba1d76551b7a87c9b96568`; MIT repository | Ports, budgets, evidence/contradiction memory, checkpoints, receipt chains | Stale-evaluation promotion, pre-action-only budgets, incomplete recovery |

The Agno repository demonstrates feasibility with an official warm GPT-5.6 scorecard of 100.00 RHAE,
183/183 levels, and 7,189 submitted actions. Its cold scorecard is 96.15. These independent results do
not substitute for this repository's required clean, state-cold acceptance run.

Primary external specifications are the
[AVO paper v1](https://arxiv.org/abs/2603.24517v1),
[NVIDIA ARC report](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/),
[ARC toolkit](https://github.com/arcprize/ARC-AGI), and
[official roster anchor](https://arcprize.org/api/v3/scorecards/9fb9db8d-3734-4885-987a-a250445c0690).
