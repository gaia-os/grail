# Elixir

**Elixir** is GRAIL's language-model-assisted statistical operation builder. A Frame
defines *what the world model is*--variables, distributions, dependencies. Elixir
proposes narrowly scoped Python operations that say *what to do with that model*: the
actual experiments and investigations run against a Frame's data.

It does not replace the Frame graph or infer hidden structure from code. The
dependency graph stays declarative and primary; generated operations are derived
artifacts that must respect it.

## What Elixir is for

Most of the value is in answering *specific* questions about a Frame: turning
prior/posterior samples into an exceedance probability, a conditional subgroup
estimate, a sensitivity check, or an interventional/counterfactual comparison. Each of
these depends on what a particular request actually asks--which variables, which
conditioning, which summary--so it's naturally something to compose per-question
rather than hard-code once.

Elixir is also useful for inference on submodels that don't fit the standard
conjugate catalog (`grail/inference/`) or the generic SVI/MCMC fallback--e.g. models
with correlated, skewed, or discrete-latent structure that want a custom guide or
enumeration strategy. Well-tested generated routines like this are good candidates to
later graduate into that hard-coded catalog, so the static library grows from what
Elixir discovers instead of regenerating the same solved routine every run.

## Role in the system

The intended flow is:

1. An application or future Interpreter identifies a Frame, variables of interest,
   and the requested operation.
2. That request is converted into an explicit function contract: purpose, argument
   names and types, return values, and relevant Frame/graph context.
3. Elixir asks an actor LLM to produce Python for that contract.
4. A structural validator checks the returned code.
5. A critic LLM reviews the proposal, returns targeted feedback, and may approve it.
6. If necessary, Elixir feeds the last proposal and critique into another actor
   attempt until an iteration or time budget is exhausted.
7. A later operation registry should attach the approved artifact, its contract,
   provenance, and revision to the Frame that it was generated for.

This division lets GRAIL use generated code without treating it as the source of a
world model. A Frame can be recompiled, inspected, or manually edited independently
of any generated operation.

## Current implementation surfaces

The current implementation is a foundation for this workflow, rather than the final
Frame-aware operation system.

- `Elixir.generate_function()` requests a one-off implementation from the actor LLM
  using an `ElixirValidator` subclass as the function contract.
- `Elixir.critic_loop()` runs actor/critic iterations subject to retry, iteration,
  and time budgets. It returns the approved code only when both internal validation
  and critic approval succeed, along with attempt history and timing information.
- `ElixirCritic` owns the task, generates an initial high-level directive, evaluates
  attempts, and retains review history.
- `ElixirValidator` uses Pydantic and Python AST inspection to require a single
  function with the requested name and arguments, constrain imports, and reject a
  small set of unsafe calls.
- `load_code()` is an experimental loader for a generated function. It is not yet a
  persistence, versioning, or deployment mechanism.

There is not yet a stable API that accepts a `Frame` and creates, stores, or invokes
its operations. The current classes should be understood as the actor-critic and
contract-validation substrate for that future API.

## Validation, review, and execution are distinct

A generated proposal can be structurally valid without being statistically correct.
Conversely, a critic can approve code that later proves invalid under GRAIL's own
contract. Elixir therefore keeps these decisions separate:

- **Validation** asks whether code satisfies the declared structural constraints.
- **Critique** asks whether the implementation seems to meet the task correctly.
- **Testing** should ask whether it behaves correctly on deterministic fixtures and
  representative Frame scenarios.
- **Execution** runs an approved operation in a controlled application environment.

Only the first two are partly implemented today. Automated statistical tests,
operation provenance, reproducible environments, and artifact persistence are
important next pieces before generated functions are treated as durable application
assets.

## Relationship to Frames

Frame definitions should remain portable YAML. Generated operations should instead
be versioned artifacts that reference:

- the Frame's name, YAML schema version, and content hash;
- selected variables and graph revision;
- the declared operation contract and run parameters;
- the generator/critic configuration and approval evidence; and
- test results and the approved source artifact.

This makes operation invalidation tractable: changing a Frame can mark dependent
operations stale without losing the Frame itself. It also lets an application choose
whether an operation is ephemeral for one analysis or promoted to a reusable
Frame-associated asset.
