# Frames

Frames are GRAIL's durable unit of world-model definition. A Frame declares the variables a model reasons about, the statistical distributions assigned to them, and the directed dependency graph that connects them. The Engine compiles a Frame into an executable Pyro model; the Runner then performs simulations or inference against that model.

A Frame is the boundary between an application's domain model and GRAIL's statistical execution layer. For example, an energy-planning application may define a Frame for demand, generation capacity, and price; a health application may define one for age, exercise, and outcomes. The application owns its user experience, raw data, and business logic. GRAIL owns the model specification, validation, graph structure, and execution-ready representation.

## Frame lifecycle

1. **Author or construct** a versioned specification for a domain model.
2. **Load and validate** it with `FrameRepository`.
3. **Compile** it to an in-memory `Frame` and directed graph.
4. **Execute** the Frame through `Engine` and `Runner`.
5. **Optionally register** the specification in local SQLite so an application can discover Frames and later associate them with runs or artifacts.

YAML is the source of truth. The runtime `Frame` is a compiled object and should be treated as ephemeral: load it when a process needs to inspect, simulate, or update a model, then save its portable specification if its definition changes.

## Frame as the construction boundary

Frame now absorbs the responsibility formerly described as the Builder's role. A
Frame is not merely a passive graph passed between other modules: it is GRAIL's home
for creating, validating, composing, loading, saving, and eventually evolving world
models.

This gives the project one coherent model boundary:

- **Frame specification and repository** define the portable model contract.
- **Runtime Frame** owns the compiled variables and graph, and provides the small
  imperative construction API needed by applications and tooling.
- **Frame composition** will eventually combine or link reusable model pieces while
  preserving explicit variable names, dependencies, metadata, and provenance.
- **Frame-associated artifacts** will eventually reference generated Elixir
  operations, run configurations, and durable results without making those volatile
  artifacts part of the core YAML definition.

The relocated `grail.frame.builder.Builder` remains transitional convenience
scaffolding. Its `new_frame()` method is useful for existing examples, but its
pickle-based save/load route is legacy behavior rather than the preferred persistence
model. New applications should author YAML or use `Frame` plus `FrameRepository`;
this keeps every durable model portable, inspectable, versionable, and independent of
Python object serialization.

## Where Frame definitions live

Canonical, project-owned Frame specifications live in [`../../data/frames`](../../data/frames). This keeps user-authored model data outside `../../src`, which contains only Python package code. The repository creates this directory on demand when it is used with its default configuration.

An application can also supply a different root directory to `FrameRepository`. This is useful when every user, tenant, or project has an isolated collection of Frames. A repository intentionally rejects paths outside its configured root.

## Anatomy of a Frame YAML file

The following is the complete minimal example from [`../../data/frames/examples/health_model.yaml`](../../data/frames/examples/health_model.yaml):

```yaml
version: 1
name: health-model
metadata:
  description: A minimal causal Frame demonstrating declarative parameter references.
  tags:
    - example
    - health
variables:
  - name: Exercise
    distribution: Bernoulli
    params:
      probs: 0.5
    description: Whether an individual exercises during the observation period.
  - name: Health
    distribution: Normal
    params:
      loc:
        $ref: Exercise
      scale: 0.1
    description: Health outcome conditioned on exercise.
dependencies:
  - source: Exercise
    target: Health
```

### Top-level fields

| Field | Required | Meaning |
| --- | --- | --- |
| `version` | No; defaults to `1` | Frame schema version. Current runtime supports version `1`. |
| `name` | Yes | Stable Frame identifier. It may contain letters, numbers, `_`, and `-`. |
| `metadata` | No | Human/application metadata: `description`, `tags`, and arbitrary `attributes`. |
| `variables` | Yes | One or more statistical variable definitions. |
| `dependencies` | No | Directed variable-to-variable relationships. |

### Variables

Each variable has a unique `name`, a `distribution`, and a `params` mapping passed to GRAIL's distribution layer. `description`, `code`, `observations`, and `attributes` may be added when useful to the calling application.

Variable names must begin with a letter and may then contain letters, numbers, and underscores. This restriction keeps YAML names compatible with generated code and probabilistic-program sample sites.

The currently built-in distribution identifiers are `Normal`, `Bernoulli`, `Uniform`, `Exponential`, and `Gamma`. Their parameter names follow Pyro's distribution constructors. Add a distribution to `DistributionFactory` before using another identifier in a Frame.

### Dependencies and parameter references

`dependencies` is the authoritative declaration of graph structure. A dependency from `Exercise` to `Health` means `Exercise` is a parent of `Health` and the model will evaluate it first.

Use an exact `$ref` mapping to pass a parent variable's sampled value into a distribution parameter:

```yaml
params:
  loc:
    $ref: Exercise
```

Every `$ref` must name a declared variable and must have a matching direct dependency (`Exercise -> Health` in the example). This prevents a distribution's implementation from silently acquiring dependencies that the graph does not show.

The graph must be a directed acyclic graph (DAG): self-links, dangling endpoints, duplicate edges, and cycles are rejected before a model is run.

## Loading a Frame into an application

Use `FrameRepository` to load YAML into the runtime `Frame` object:

```python
from grail.frame import FrameRepository

frames = FrameRepository()
frame = frames.load("health_model.yaml")

health = frame.get_variable("Health")
exercise_node_id = frame.get_variable_id("Exercise")
```

The runtime graph uses generated node IDs internally, while application code should generally use stable variable names. `Frame.get_variable()` and `Frame.get_variable_id()` support this distinction. Variable names are portable across YAML saves; internal IDs are not.

For isolated application workspaces, select a dedicated root:

```python
from pathlib import Path

from grail.frame import FrameRepository

frames = FrameRepository(Path("/path/to/application-data/frames"))
scenario = frames.load("scenario-2026.yaml")
```

## Running a Frame

Pass the compiled Frame to the Engine, then give its callable model to a Runner:

```python
from grail.engine import Engine
from grail.frame import FrameRepository
from grail.runner import Runner

frame = FrameRepository().load("health_model.yaml")
runner = Runner(Engine(frame).get_model())
prior_samples = runner.simulate(num_samples=1_000)
```

`simulate()` is the current prior-predictive path. Data passed to Runner methods can be keyed by a stable variable name (recommended) or the corresponding runtime node ID:

```python
from grail.engine import Engine
from grail.frame import FrameRepository
from grail.runner import Runner

observed_health_values = [0.4, 0.7, 1.1]
runner = Runner(Engine(FrameRepository().load("health_model.yaml")).get_model())
loss_history = runner.train_svi(
    data={"Health": observed_health_values},
    n_steps=2_000,
)
posterior_samples = runner.predict(num_samples=1_000)
```

Small, static observations may be included in YAML using a variable's `observations` field. In most applications, keep sizeable or frequently changing datasets outside the Frame file and pass them at run time. This lets one durable world-model definition be reused for many datasets, experiments, or scenarios.

## Creating and saving Frames in Python

YAML is preferred for authored and reviewed models. The runtime API remains useful for application-generated models, prototypes, and tooling:

```python
from grail.frame import Frame, FrameRepository

frame = Frame("demand-model")
demand_id = frame.add_variable("Demand", "Normal", {"loc": 100.0, "scale": 10.0})
frame.add_variable("Price", "Normal", {"loc": demand_id, "scale": 2.0})
frame.add_dependency("Demand", "Price")

FrameRepository().save(frame)
```

When constructing directly, a parameter reference uses the returned runtime node ID. `FrameRepository.save()` converts known node IDs back into portable `$ref` mappings, so the resulting YAML remains readable and stable.

## SQLite registry

`FrameRegistry` provides an optional local SQLite index at `../../data/grail.sqlite3`. It records a Frame's name, YAML path, schema version, content hash, description, and timestamps. It does **not** store or supersede the model definition itself.

Register a validated spec after saving it:

```python
from grail.frame import FrameRegistry, FrameRepository

frames = FrameRepository()
path = frames.path_for("health-model")
spec = frames.load_spec(path.name)

registry = FrameRegistry()
record = registry.register(spec, path)
```

This index is a foundation for an application-level Frame library, execution run history, generated Elixir operation records, and artifact locations. Simulation samples and large datasets do not belong in this metadata database.

## Current scope and future extensions

Frames currently define variables, distributions, observations, metadata, and an explicit DAG. They compile into the Pyro-backed Engine used by the existing Runner. The `grail.frame` package is also now the authoritative location for Frame construction; the former Builder API is nested beneath it only for transition. Frames do not yet persist run state, generated Elixir code, posterior state, or result artifacts as part of the Frame format.

That separation is intentional. As GRAIL adds Elixir-generated prior predictive, posterior inference, and posterior predictive operations, those operation records can reference a stable Frame specification and its hash without turning YAML into a database or embedding volatile outputs in the world-model definition.
