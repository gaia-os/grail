# Elixir

**Elixir** is GRAIL's proposed language-model-assisted statistical operation builder. A
Frame says *what the world model is*: its variables, distributions, metadata, and
explicit graph. Elixir is responsible for proposing narrowly scoped Python operations
that say *what to do with that model*.

It is not intended to replace the Frame graph or infer hidden graph structure from
code. The dependency graph remains declarative and primary; generated operations are
derived artifacts that must respect it.

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

## Operations

The initial target operation set is:

- **Prior predictive**: draw from a Frame before conditioning on data, to inspect
  whether domain assumptions produce plausible outcomes.
- **Posterior inference**: update beliefs using observations or other supplied data.
- **Posterior predictive**: draw future/held-out outcomes from beliefs after
  inference.

All three should expose explicit integer run controls such as `n_samples` and
`n_steps`; the Engine and Runner use these when executing an operation.

The broader catalog remains a design target, not a promise of current functionality:
likelihood evaluation, sensitivity analysis, interventional simulation, and
counterfactual analysis. Those operations require more than code generation: they
need clear model semantics, diagnostics, test fixtures, and—in the causal cases—a
justified causal graph.

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

## Safety boundary

Elixir-generated Python is untrusted input. AST checks and an import allowlist are
useful guardrails, but they are not a security sandbox. In particular, the current
experimental code-loading path executes Python in-process.

Applications should not execute generated code merely because an LLM or critic
approved it. The eventual design should add isolated execution, resource limits,
allowlisted dependencies, test gates, and a human/application approval policy before
an operation can affect durable state or external systems.

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

