# Engine

The **Engine** is the translation boundary between a declarative Frame and an
executable probabilistic model. It does not decide what a world model contains and it
does not decide how long to run an analysis. Its job is to take the validated graph
and variable definitions already held by a Frame, initialize the execution-ready
pieces, and return a model that the Runner can execute.

Today, the Engine targets Pyro. In the longer term, the same boundary gives GRAIL a
place to prepare generated operations, select execution backends, resolve external
state, and construct a reproducible run package without making Frames depend on a
particular Runner implementation.

## Position in the model lifecycle

A typical application flow is:

1. Load or construct a Frame.
2. Validate its YAML specification and compile its graph.
3. Give the runtime Frame to `Engine`.
4. Obtain an executable model with `Engine.get_model()`.
5. Give that model and explicit run parameters to a Runner.

The separation matters. The Frame is durable model definition; the Engine's model is
an in-memory execution artifact; the Runner carries out a requested statistical
operation. Keeping these roles apart makes it possible to reuse one Frame across
many datasets, run configurations, and later execution backends.

## How the current Engine works

`Engine.get_model()` captures the Frame graph in a Pyro model function. For each
execution trace, the function:

1. obtains a topological ordering of the directed Frame graph;
2. visits each `VariableNode` after its parents;
3. retrieves its configured distribution and parameters;
4. replaces direct runtime node-ID parameter references with sampled parent values;
5. builds the matching Pyro distribution through `DistributionFactory`; and
6. creates a Pyro sample site using the variable's stable name.

The Frame schema keeps authored YAML readable through mappings such as
`{"$ref": "Exercise"}`. During Frame compilation, those references become internal
node IDs. The Engine resolves those IDs within the current trace, so a child
variable's distribution is parameterized by the sampled value of its declared parent.

A Frame graph is validated as a DAG before this stage. That gives the Engine a valid
sampling order and prevents accidental recursive evaluation.

## Observations and external data

The callable model returned by the current Engine accepts an optional data mapping.
Values may be keyed by a Frame variable's stable name—preferred for applications—or
by its internal runtime node ID. Such data is used as an observation at the matching
Pyro sample site.

A variable may also contain small static `observations` in its Frame specification.
Runtime data takes precedence. This supports a useful distinction:

- Use Frame YAML for the relatively stable **structure and assumptions** of a world
  model.
- Supply a dataset at execution time when it is large, mutable, experiment-specific,
  private, or shared across multiple Frames.

The Engine is not yet a data catalog, artifact store, or state manager. Applications
should retain ownership of loading, authorization, and transformation of external
data before it reaches a run.

## What the Engine should eventually prepare

The current implementation is deliberately small. As GRAIL matures, the Engine is a
natural home for an explicit execution-plan or run-package layer containing:

- the selected Frame revision and compiled graph;
- selected variables, target outputs, and operation type;
- validated observations, interventions, and scenario parameters;
- the generated Elixir operation, if one is used;
- random-seed, backend, device, and numerical settings; and
- references to durable input and output artifacts.

That package would make a run inspectable and reproducible while keeping its volatile
state out of the canonical Frame YAML.

## Current boundaries and limitations

The Engine is not yet a general model compiler. Its present behavior should be viewed
as a functional vertical slice for simple directed probabilistic models.

In particular:

- only the distributions implemented by `DistributionFactory` can be executed;
- parameter reference resolution currently serves the simple direct values used by
  those distribution definitions;
- graph edges are meaningful model dependencies, but a causal claim still requires
  domain justification rather than merely drawing an arrow;
- there is no formal run-plan object, backend abstraction, compiled-model cache, or
  operation registry yet; and
- model construction currently occurs in-process and relies on Pyro's normal runtime
  state.

The Engine should remain focused on **preparing a model for execution**. Persistence,
user-facing model authoring, and long-running scheduling belong respectively to the
Frame layer, application layer, and future orchestration/workflow layer.

