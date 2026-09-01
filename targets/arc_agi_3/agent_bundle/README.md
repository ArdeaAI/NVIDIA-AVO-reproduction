# Default ARC agent bundle

This directory is the frozen state-cold seed supplied to each campaign. It contains only generic
observation, experimentation, planning, and memory policy. It must not contain game identifiers,
solutions, human baselines, or knowledge derived from prior public-game traces.

Every run records the complete directory digest. Any bundle evolved with public-game evidence is a warm
candidate and cannot replace this seed for the primary cold claim without an explicit protocol revision.

