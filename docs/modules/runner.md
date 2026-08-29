# Runner

The **Runner** is where a prepared GRAIL model is actually executed. It receives an
executable model from the Engine plus explicit run controls, invokes the relevant
probabilistic-programming procedure, and returns the resulting values or diagnostics.

A Runner is intentionally downstream of Frame construction. A Frame defines the
world model; the Engine prepares its executable form; the Runner applies one
statistical mode to it. This lets one Frame support repeated simulations, inference
runs, forecasts, and—later—other operations without baking a particular run into the
model definition.

## Current execution modes

The current `Runner` exposes a small Pyro-backed baseline:

| Method | Current purpose | Important state |
| --- | --- | --- |
| `simulate(num_samples, data=None)` | Prior-predictive sampling from the model. | Stateless between calls. |
| `train_svi(data=None, n_steps, learning_rate)` | Fits a variational posterior through stochastic variational inference (SVI). | Creates and retains an auto-guide. |
| `predict(num_samples, data=None)` | Posterior-predictive sampling after SVI. | Requires the guide created by `train_svi()`. |
| `do_operation(interventions, num_samples)` | Samples an intervened Pyro model. | Intervention mapping is per call. |

`n_samples` governs the number of draws. `n_steps` governs optimization work during
SVI. They are explicit because computational budget is part of a run request, not a
property of a Frame.

## Prior predictive, inference, and posterior predictive

The three initial GRAIL operation concepts map cleanly to Runner behavior:

1. **Prior predictive** (`simulate`) asks: *If the current Frame assumptions were
   true, what data could plausibly occur?* It is useful for checking distributions,
   parameter scales, and graph assumptions before fitting to observations.
2. **Posterior inference** (`train_svi`) asks: *How should the model's latent beliefs
   change after seeing these observations?* The current implementation uses Pyro's
   `AutoDiagonalNormal` guide as a general initial approximation.
3. **Posterior predictive** (`predict`) asks: *Given what was learned, what future or
   replicated observations does the model predict?* It draws with the retained guide
   and is unavailable until inference has run.

These modes belong to one conceptual workflow, but they are not interchangeable. An
application should save the inputs, run settings, diagnostics, and Frame revision
alongside any result it needs to reproduce.

## Inputs and outputs

The Engine model accepts observations keyed by stable Frame variable names, which is
the recommended application-facing convention. The Runner passes those inputs to
Pyro. Its methods currently return Pyro/Torch-native values:

- predictive methods return a mapping of sample-site names to tensors;
- `train_svi()` returns the loss value for each optimization step; and
- interventions are mapped from Frame variable names to fixed values accepted by
  Pyro.

The Runner does not yet normalize results into application schemas, write Parquet
artifacts, record a run in SQLite, or retain structured diagnostics. An application
can adapt the returned tensors today; a future GRAIL run-record layer should own
these responsibilities consistently.

## State, reproducibility, and isolation

There are two relevant types of state:

- **Frame state**: the declarative model definition and its graph. This is durable
  YAML and should be identified by a version/hash.
- **Run state**: observations, parameter-store values, the fitted guide, random seed,
  run controls, and output artifacts. This is specific to one execution.

The current Runner retains the fitted guide on the Python object. `train_svi()` also
clears Pyro's global parameter store before fitting. This is practical for the
current single-process toolkit, but it means applications should treat a Runner as a
short-lived, single-run object rather than a multi-user or concurrent service
primitive.

A mature runner should receive an immutable run request and emit a durable run record
containing the Frame revision, input artifact references, seed, backend/device
configuration, operation implementation, status, timings, diagnostics, and output
artifact paths.

## Interventions and causal scope

`do_operation()` exposes Pyro's intervention mechanism: it replaces selected sample
sites with specified values and samples the resulting model. It is a useful execution
primitive, not proof that a Frame is causally valid.

An intervention should only be interpreted as a causal `do(...)` query when its Frame
edges encode a domain-justified causal model. A purely predictive dependency graph may
still be useful for simulation, but it does not automatically support causal claims,
counterfactuals, or policy recommendations.

## Relationship to Elixir

Today the Runner calls its built-in Pyro procedures directly. In the intended design,
Elixir will generate and validate additional Frame-derived operations—beginning with
prior predictive, posterior inference, and posterior predictive implementations.
The Runner should eventually execute those operations through a common run contract,
not allow generated code to bypass input validation, provenance, resource limits, or
result recording.

Until that contract exists, consider the current Runner a clear execution baseline:
it demonstrates the separation between model definition and run behavior without
claiming to be the final run-management system.

