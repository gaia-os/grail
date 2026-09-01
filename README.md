# GRAIL

_Graph Reasoning from AI Learning_

GRAIL is a toolkit for building probabilistic world/causal models with explicit graph structure.
It combines declarative model definitions (**Frames**) with executable inference tooling (**Engine/Runner**) and an LLM-assisted operation-generation foundation (**Elixir**).

## Purpose

GRAIL is designed to make world-model construction and Bayesian analysis practical without hiding model structure.
The core idea is simple: keep the model definition explicit and reviewable, then layer inference and AI assistance on top.

## Current State (v0.x)

- **Frames are live**: YAML model specs define variables, distributions, and dependencies, then validate/compile through `FrameRepository`.
- **Inference pipeline is live**: `Engine` builds executable Pyro models and `Runner` supports prior predictive sampling, SVI training, posterior predictive sampling, and strategy-based inference.
- **Posterior state is live**: observations can be persisted and reused for posterior updates (including an exact Beta-Bernoulli strategy path).
- **Elixir foundation is live**: actor-critic generation, structural validation, and review loops exist for statistical operation code.
- **Still in progress**: full Frame-aware operation registration/provenance, broader operation coverage, and hardened execution boundaries for generated code.

## Near-Term Focus

- Tighten the Frame + Engine + Runner workflow with stronger examples and test coverage.
- Evolve Elixir from code-generation substrate into reliable Frame-aware operation artifacts.
- Expand and stabilize the core operation set around prior predictive, posterior inference, and posterior predictive flows.
- Build out Builder-style frame lifecycle workflows (create/read/update/delete patterns).

## Longer-Term Direction

- Add richer operation classes (likelihood, sensitivity, interventional, counterfactual).
- Introduce interpreter/agentic orchestration once the core tooling layer is stable.
- Improve persistence/provenance for generated operations and run artifacts.
- Scale architecture from local-first defaults (SQLite/files) to multi-user/remote backends as needed.

## Documentation

- Overview and roadmap notes: [`docs/sketch.md`](docs/sketch.md)
- Frames: [`docs/modules/frames.md`](docs/modules/frames.md)
- Engine: [`docs/modules/engine.md`](docs/modules/engine.md)
- Runner: [`docs/modules/runner.md`](docs/modules/runner.md)
- Observations/posteriors: [`docs/modules/observations.md`](docs/modules/observations.md)
- Elixir: [`docs/modules/elixir.md`](docs/modules/elixir.md)
- Tooling/storage architecture: [`docs/systems-arch.md`](docs/systems-arch.md)
