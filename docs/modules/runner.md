# Runner

The **Runner** executes a model that the Engine prepared.

Think of it as the runtime API for your analysis calls.

## Main methods

- `simulate(num_samples, data=None)`
  - prior predictive sampling
- `train_svi(data=None, n_steps, learning_rate)`
  - fits an in-memory variational posterior from scratch
- `predict(num_samples, data=None)`
  - posterior predictive sampling (after `train_svi()`)
- `infer(strategy)`
  - runs a Frame-aware strategy against persisted observation history and retains its posterior
- `do_operation(interventions, num_samples)`
  - sampling under interventions

## Inputs and outputs

- Input data is a mapping keyed by variable name.
- Predictive calls return tensors by sample-site name.
- `train_svi()` returns a loss curve (one value per step).

`num_samples` controls draw count. `n_steps` controls SVI optimization depth.

## Usage notes

- A single Frame can be run many times with different data and run settings.
- Keep run settings with saved results so runs are reproducible.
- Treat Runner instances as short-lived per analysis task.
- Construct `Runner(model, frame=frame)` when calling `infer()`. For the exact,
  resumable Beta–Bernoulli strategy, see [Observations and Posterior State](observations.md).

## About interventions

`do_operation()` is useful for intervention-style experiments.

Interpretation still depends on model quality: intervention results are only causal if
your Frame's structure is a valid causal model.
