# Frames

A **Frame** is the model file for your domain.

It defines:

- variables (what you care about),
- distributions (how each variable behaves),
- dependencies (which variables influence others).

Frames are meant to be durable and reviewable. In practice, YAML is your source of truth
for the model structure and its **initial priors**. Runtime observations and posterior
results are separate state; see [Observations and Posterior State](observations.md).

## Frame history

It's worth noting that the frame yaml definition is hashed into the graph to key its identity.
Thus, if you make edits to a frame definition, you will spawn a fresh history for this new definition.

## Typical flow

1. Write or load a Frame YAML file.
2. Validate and compile it with `FrameRepository`.
3. Run it with `Engine` and `Runner`.

```python
from grail.engine import Engine
from grail.frame import FrameRepository
from grail.runner import Runner

frame = FrameRepository().load("health_model.yaml")
runner = Runner(Engine(frame).get_model())
samples = runner.simulate(num_samples=1_000)
```

## Example YAML

```yaml
version: 1
name: health-model
variables:
  - name: Exercise
    distribution: bernoulli
    params:
      theta: 0.5
  - name: Health
    distribution: normal
    params:
      loc:
        $ref: Exercise
      scale: 0.1
dependencies:
  - source: Exercise
    target: Health
```

## Rules that matter most

- Variable names must be unique.
- Distribution codes are lowercase (for example `normal`, `bernoulli`, `constant`).
- Params must match the selected distribution's accepted fields.
- Dependencies must form a DAG (no cycles).
- If a param uses `$ref`, there must be a matching dependency edge.

## Distribution quick reference

- `normal`: `loc`, `scale`
- `bernoulli`: `theta`
- `uniform`: `low`, `high`
- `exponential`: `rate`
- `gamma`: `concentration`, `rate`
- `lognormal`: `loc`, `scale`
- `binomial`: `n`, `theta`
- `beta`: `alpha`, `beta`
- `constant`: `value`

## Practical notes

- Keep large or changing datasets outside Frame YAML; pass them at run time.
- Keep observations and posterior outputs outside Frame YAML; a Frame should stay a
  model definition. Use the Frame state store for durable evidence history.
- Runtime node IDs are internal. Prefer variable names in app-facing code.
