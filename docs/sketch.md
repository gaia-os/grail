# Sketching Ideas

## Questions

1. Granularity. Do we want to guide granularity in construction or be more hands-off? I think the answer will become more apparent as we define model structures
2. Do we want an agentic-orchestrator type setup? Maybe not for first iteration. Let's start with first focusing on the tools. An agentic system would depend on this anyway.

## Glossary and Terms

### Frame 

A **Frame** is a collection of models staged to be used for a simulation.
It will likely contain a mixture of data types and structures.
World models, graph-relational or causal information, results, sim params...

- Question: Should subgraphs be a distinct mechanism from Frames? Or should these be mutually-dependent? (Each frame has a subgraph, each subgraph belongs to a frame)

### Variable

Every Frame has at least one Variable towards which to perform inference on.
Observed data values will be associated with such variables.

### Causal Graph

Causal graphs are like more robust world models, depicting direct causal relationships.
They allow for studying _interventions_ and _counterfactuals_.
For instance, world models don't inherently distinguish between correlation and causation, unlike Causal.
They are also more robust to distributional shifts.

#### Causal Illustration

Suppose you have three variables: Exercise (E), Health (H), and Age (A).

- World Model: Might learn that E and H are correlated, and use this to predict health outcomes.
- Causal Graph: Might show that Age affects both Exercise and Health, revealing that the E-H correlation is confounded by Age. Intervening on Exercise (do(E)) can then be properly analyzed for its causal effect on Health.

### Graph Methods

We also want to provide the necessary toolset for performing various graph methods.
This aspect is...probably "adjacent" to the other WM/statistical aspects..

### Initializer / Init

As a toolkit, we will probably want some sort of initialization process.
Maybe should be bundled into Engine

### Stats dir

The stats directory houses various statistical definitions and tools for GRAIL to use.

### The Interpreter

Language model with special prompting that interprets the query.
It is to be seen how much of the runs will be parsed from natural language vs exposed tooling, but anyway.

- The interpreter establishes which nodes are involved, what attributes to use, and prepares a final, sanitized simulation request to be passed along.
- It should probably also spec. the creation of new models/components for the simulation, and potentially whether such pieces are ephemeral or not.

This final prompt is also passed into the Engine, where it will guide the simulation construction.

### The Runner

"Where" simulations are run. Intakes the loaded components and run config parameters.

### The Engine

In between the Builder and the Runner.
The Engine **initializes** and strings together the different pieces for the simulation run,
and sends the prepared package into the Runner.

### Frames

Users(/LLM) should initialize a Frame, and then sequentially add elements to the Frame.
This is the world model construction/initialization.

### The Builder

- This is where new models are built
- It also **composes** such models from stored models in the library

#### GRAIL YAML config spec

Consider if we want to implement a GRAIL YAML spec, to interpret and construct models.

#### Saving, Loading, etc.

The Builder should have some means of saving and loading different products.
This is to allow an external persistence that is not managed by GRAIL, and thus not required to be stored in RAM all the time.
These can be packaged as **Frames**. 

#### Thinking

The Builder has to do be capable of some careful thinking when it comes to constructing these statistical models.

- It should be allowed to perform actor-critic and research style loops to evaluate the quality of its proposed builds. 
  - This is especially relevant as builds become more advanced.

### The Library / Repo

_We will postpone this in full for now, since we are first developing this in a more toolkit fashion._

Where constructed models are stored for retrieval.

- What should be ephemeral and what should be stored? 
  - Well, things that are 'static', models of tech nodes should persist
  - These sorts of models can be constructed during document upload/AI research phases, and queued as side processes.
    - e.g. "X enabling tech produces between a-b TWh"
    - or as an llm interprets a tech, the TRL is surmised and recorded as part of its world model as well (assigned as attribute).

### Attributes

Data attached to a node

### Elixir

**Elixir** is a language-model-driven code generation system for constructing statistical operation functions.

Given a Frame's variable structure and dependency graph, Elixir uses an actor-critic loop to generate Python implementations of various statistical operations. The Interpreter specifies which variables are of interest and what operations are needed; Elixir then generates, validates (via critic feedback), and compiles the functions that Frames will use during inference and simulation.

#### Standard Statistical Operations

Elixir can generate implementations for the following operations:

1. **Prior Predictive** - Sample from the prior to validate prior assumptions before observing data
2. **Likelihood** - Evaluate how well observed data fits under current parameters
3. **Posterior Inference** - Update beliefs given data (Bayesian inference, etc.)
4. **Posterior Predictive** - Predict future observations using updated beliefs
5. **Sensitivity Analysis** - Assess robustness of inferences to perturbations
6. **Interventional** - Compute causal effects via interventions (do-calculus)
7. **Counterfactual** - Analyze "what would have happened" under alternative scenarios

#### Initial Scope

For the initial implementation, Elixir will focus on three core operations:

- **Prior Predictive**: Validation of domain assumptions
- **Posterior Inference**: Learning from observed data
- **Posterior Predictive**: Running simulations and forecasts

Each generated operation accepts integer parameters (e.g., `n_samples`, `n_steps`) to control simulation granularity, which the Engine and Runner use during execution.

---

## Development Map

1. Building world models models with Elixir
   - Upgrade our existing foundation so the LM can construct viable world models
2. Define a Frame, and how it can manage WMs
3. Work on the Runner and Engine through a toy example
4. Construct the Builder to CRUD frames
5. ....
