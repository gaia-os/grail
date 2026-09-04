# Runner

The **Runner** executes a model that the Engine prepared.

Think of it as the runtime API for your analysis calls.

## Main methods

- `simulate(num_samples, observations=None)`
  - prior predictive sampling
- `train_svi(observations=None, n_steps, learning_rate, guide=None)`
  - fits an in-memory variational posterior from scratch, with an
    `AutoDiagonalNormal` guide unless you supply one
- `predict(num_samples)`
  - posterior predictive sampling (after `train_svi()`)
- `infer(strategy)`
  - runs a Frame-aware strategy against persisted observation history and retains its posterior
- `do_operation(interventions, num_samples, observations=None)`
  - sampling under interventions

## Inputs and outputs

- Observations are mappings keyed by variable name (or node ID). Supply them to
  `simulate()`, `train_svi()`, or `do_operation()` when conditioning the model.
- Predictive calls return tensors by sample-site name.
- `train_svi()` returns a loss curve (one value per step).

`num_samples` controls draw count. `n_steps` controls SVI optimization depth.

## Usage notes

- A single Frame can be run many times with different observations and run settings.
- Keep run settings with saved results so runs are reproducible.
- Treat Runner instances as short-lived per analysis task.
- Construct `Runner(model, frame=frame)` when calling `infer()`. For the exact,
  resumable Beta–Bernoulli strategy, see [Observations and Posterior State](observations.md).

## About interventions

`do_operation()` is useful for intervention-style experiments.

Pyro implements `do` as a Single World Intervention Graph: downstream variables
see the value you imposed, but the trace's own site for an intervened variable
holds a fresh draw from its original distribution that propagates nowhere.
Returning that draw would misreport the world you asked about, so GRAIL replaces
the intervened sites with the values actually imposed. `samples["X"]` after
`do_operation({"X": 10.0})` is therefore `10.0`, not a prior draw.

Interpretation still depends on model quality: intervention results are only causal if
your Frame's structure is a valid causal model.
