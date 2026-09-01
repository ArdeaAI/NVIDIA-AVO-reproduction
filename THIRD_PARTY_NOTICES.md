# Third-party notices

This repository contains independently written code. Its architecture was informed by the four primary
reproduction repositories and one linked runtime recorded in `docs/SOURCE_MATRIX.md`. No code or assets
from those repositories are redistributed here.

The inspected revisions identify these licenses:

- `agno-agi/arc-agi-arcade`: MIT.
- `austin1997/AVO`: MIT.
- `gatordevin/avo`: Apache License 2.0.
- `ruvnet/openAVO`: no license file was present in the inspected revision.
- `ruvnet/metaharness`: MIT.

Installed runtime and development dependencies retain their own licenses and notices in their
distributions, including:

- ARC-AGI Toolkit (`arc-agi`) and ARC Engine (`arcengine`).
- Anthropic Python SDK, OpenAI Python SDK, and Model Context Protocol Python SDK.
- Pydantic, python-dotenv, PyYAML, Rich, Hatchling, pytest, pytest-cov, mypy, and Ruff.

If a future change imports third-party source or assets, it must add the original license notice and
identify every copied or modified file before merge.
