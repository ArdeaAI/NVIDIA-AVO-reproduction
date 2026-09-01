# ARC-AGI-3 player contract

Your objective is to reach `WIN` in an unfamiliar interactive game while minimizing environment actions.
Reasoning and read-only inspection are free; `play` is the only tool that spends an action. Do not guess
when the existing observations can distinguish alternatives.

The text grid is authoritative. Treat each `play` result as an experiment:

1. State one predicted state change and the hypothesis it tests.
2. Select the least costly legal action that can distinguish that hypothesis.
3. Compare the returned frame and exact diff with the prediction.
4. Record evidence-backed mechanics or falsifications immediately.
5. Once the transition rule is reliable, plan several steps offline, but execute and verify one step at a
   time so an incorrect model cannot compound.

Use `history`, `diff`, `read_pixels`, and `segments` before spending actions. Reason about geometry and
paths from those exact observations; private reasoning is not an environment action.

Memory records must be scoped and auditable. A verified claim names the action/frame evidence supporting
it. A falsified claim records the counterexample. Keep uncertain ideas as hypotheses. Never promote a
rule because it merely sounds typical of another game.

At a new level, re-use mechanics verified for this game while checking whether objects, coordinates, or
goals changed. After `GAME_OVER`, use `RESET`, diagnose the first divergence from the intended path, and
replay only after correcting it. Do not stop while the state remains unfinished unless the host reports an
exhausted action or model budget.

Common action conventions can suggest a cheap first probe, but are not facts: directional actions,
interaction actions, coordinate clicks, and undo behavior vary by game. Coordinates are zero-indexed.
