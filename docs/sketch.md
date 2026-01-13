# Sketching Ideas

## Questions

1. Granularity. Do we want to guide granularity in construction or be more hands-off? I think the answer will become more apparent as we define model structures
2. How much do we want the simulation tools to be exposed to the user and how much do we want the system to be run via natural language?

## Glossary and Terms

### Frame 

A **Frame** is a collection of models staged to be used for a simulation.
It will likely contain a mixture of data types and structures.
World models, graph-relational or causal information, results, sim params...

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

### The Builder

- This is where new models are built
- It also **composes** such models from stored models in the library

#### Frames

Frames (defined above) are build in the Builder.
Users(/LLM) should initialize a Frame, and then sequentially add elements to the Frame.
This is the world model construction/initialization.
The frame should also 

#### Saving, Loading, etc.

The Builder should have some means of saving and loading different products.
This is to allow an external persistence that is not managed by GRAIL, and thus not required to be stored in RAM all the time.
These can be packaged as **Frames**. 

#### Thinking

The Builder has to do be capable of some careful thinking when it comes to constructing these statistical models.

- It should be allowed to perform actor-critic and research style loops to evaluate the quality of its proposed builds. 
  - This is especially relevant as builds become more advanced.

### The Library / Repo

Where constructed models are stored for retrieval.

- What should be ephemeral and what should be stored? 
  - Well, things that are 'static', models of tech nodes should persist
  - These sorts of models can be constructed during document upload/AI research phases, and queued as side processes.
    - e.g. "X enabling tech produces between a-b TWh"
    - or as an llm interprets a tech, the TRL is surmised and recorded as part of its world model as well (assigned as attribute).

### Attributes

Data attached to a node

