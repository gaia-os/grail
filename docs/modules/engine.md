# Engine

The **Engine** turns a validated Frame into an executable model.

If Frame is the model definition, Engine is the step that prepares that definition for execution.

## Where it fits

1. Frame defines variables and dependencies.
2. Engine builds a runnable model from that Frame.
3. Runner executes the model.

This separation keeps one model reusable across many runs.

## What `Engine.get_model()` does

The returned callable model:

1. walks variables in dependency order,
2. resolves `$ref` parent values,
3. builds each variable's distribution,
4. creates sample sites by variable name.

Current backend: Pyro.

## Observed data

You can pass observed data as a mapping keyed by variable name.

- When observations are passed at run time, those values are used.
- Otherwise, values are sampled from prior distributions.

Small fixed observations can live in YAML, but changing datasets are usually better passed at run time.

## Limits

- Only supported built-in distribution codes can run.
- Graph must be acyclic (DAG).
- Engine runs in-process with the current Pyro runtime state.

