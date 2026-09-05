# GRAIL

_Graph Reasoning from AI Learning_

GRAIL is a toolkit for building probabilistic world/causal models with explicit graph structure.
It combines declarative model definitions (**Frames**) with executable inference tooling (**Engine/Runner**) and an LLM-assisted operation-generation foundation (**Elixir**).

## Purpose

GRAIL is designed to make world-model construction and Bayesian analysis practical without hiding model structure.
The core idea is simple: keep the model definition explicit and reviewable, then layer inference and AI assistance on top.

## Illustrative Example: Health and Exercise

Highly recommended is to read this short [Illustrative Example](docs/illustrative_health_example.md)
to get a sense of the usage flow of GRAIL and its purpose.

## Installation

```bash
# Base
uv pip install .
# Extras
uv pip install ".[extra1,extra2]"
# All extras
uv pip install -r pyproject.toml --all-extras
```

---

## Visualization Dashboard

A Streamlit dashboard lets you browse registered Frames (metadata, variables, causal graph, posteriors)
and inspect persisted inference runs (diagnostics, results) without writing any code.

Install the extra dependencies, then launch it:

```bash
uv sync --extra viz   # or: uv sync --all-extras
uv run grail-dash
```

This opens the dashboard at `http://localhost:8501`, with a **Frames** page and a **Runs** page in the sidebar.

---

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

- Illustrative walkthrough (start here): [`docs/illustrative_health_example.md`](docs/illustrative_health_example.md)
- Frames: [`docs/modules/frames.md`](docs/modules/frames.md)
- Engine: [`docs/modules/engine.md`](docs/modules/engine.md)
- Runner: [`docs/modules/runner.md`](docs/modules/runner.md)
- Observations/posteriors: [`docs/modules/observations.md`](docs/modules/observations.md)
- Elixir: [`docs/modules/elixir.md`](docs/modules/elixir.md)
