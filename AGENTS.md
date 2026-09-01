# Repository instructions

- Use Python 3.12 and keep public interfaces typed.
- Put the opening and closing triple quotes of every Python docstring on their own lines.
- Never hard-code ARC-AGI-3 game identifiers, rules, solutions, human baselines, or action traces in the
  default agent bundle or production prompts.
- Keep environment engines, evaluator code, credentials, and authoritative run ledgers outside model-
  writable workspaces.
- Preserve cold/warm/resume provenance and fail closed on hash or replay mismatches.
- Do not weaken the human confirmation gate around Competition scorecards.

