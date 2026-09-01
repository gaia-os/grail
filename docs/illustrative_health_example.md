# Example Walkthrough: Health and Exercise

Let's say you're an analyst evaluating a community walking program. You
want to know whether exercising is associated with better health outcomes in your
participant group, and later, whether that relationship holds up across age groups.

## 1. Describing the model to the Interpreter

You don't start by writing YAML. You describe the program in plain terms: some
participants exercise regularly and some don't; you're not sure of the exact rate; and
you believe exercising affects a downstream health score.

The Interpreter's role is to convert that description into an explicit structure: a
latent exercise-rate variable, a per-participant exercise outcome that depends on it,
and a health-score outcome that depends on exercise. It resolves ambiguity (for
example, how confident you are in your initial rate guess) before handing the
structure off.

## 2. Drafting the Frame

**Elixir** uses that structure to draft a Frame: three named variables, a distribution and
initial prior for each, plain-language descriptions, and explicit dependency edges
(`ExerciseRate → Exercise → Health`). This maps directly onto the standard Frame YAML
format described in [Frames](modules/frames.md).

You review the draft, edit a description, and accept it.

## 3. Prior predictive check

Before any real data is involved, you run a prior predictive simulation--drawing an
imaginary population directly from the Frame's priors. This is a sanity check on the
model, not on the data: do the simulated health scores fall in a plausible range, and
does the assumed exercise rate look appropriately uncertain?

If the output looked implausible (e.g., health scores outside a valid range), the
right response would be to revise the Frame, not to proceed. This is a step worth
repeating any time the Frame's structure or priors change materially.

## 4. Recording observations

You upload a batch of real survey data: eight participants, seven of whom report
exercising regularly. This data is stored as an observation batch, and can be called upon
for whatever future analysis.

See [Observations](modules/observations.md) for details.

## 5. Updating beliefs

Through the Interpreter or your own code, you run inference to update the exercise-rate belief 
using the recorded observations. The **Engine** and **Runner** modules come into play here.

Because `Beta` prior + `Bernoulli` likelihood is a standard conjugate pair, this
update has a single, exact closed-form answer--computed instantly.
The result is a sharper posterior belief about the exercise rate, and a record of exactly which observation batches 
were already accounted for.
Observation batches are handled accordingly so that subsequent analysis can correctly work from the previous
FrameState.

Note: For simpler "neat-and-tidy" inference operations like this (with known conjugate priors etc.),
**Elixir** may not be necessary to generate custom scripting, and some in-house GRAIL statistical
tools may be sufficient.

See [Engine](modules/engine.md)  
See [Runner](modules/runner.md)

## 6. Standard predictive and intervention queries

With updated beliefs in place, you run two queries using GRAIL's existing,
Frame-agnostic tools--no code-gen at this stage, with a straightforward model:

- **Posterior predictive** — simulate a new cohort under the *updated* beliefs, to
  see what outcomes look plausible going forward.
- **Intervention** — simulate a version of the cohort where exercise is fixed to
  "yes" for everyone, and compare simulated health scores against the un-intervened
  case.

Both are generic capabilities: they work the same way for any valid Frame, since they
depend only on graph structure, not on anything specific to exercise and health.

## 7. Turning up the heat: A question with no fixed answer

You then ask something more specific: does the exercise-health relationship hold up
separately for younger and older participants, and--given that your participant pool
is small and self-selected rather than randomly assigned--is there a defensible
estimate of the effect of encouraging more exercise in the least-active older
participants?

This does not reduce to a known formula, and it isn't served by the generic
predictive/intervention tools above either. It bundles several specific choices: how
to split by age, what a fair comparison looks like given a small non-random sample,
and what the answer should be reported as. No pre-built tool matches this
particular combination of asks.

## 8. Generating a bespoke analysis with Elixir

This is the case **Elixir** is for. The Interpreter turns your request into an explicit
function contract: given the Frame, its recorded evidence, and an age cutoff, produce
a comparison of the exercise-health relationship above and below that cutoff, with an
explicit caveat about sample size and selection bias.

The actor-critic loop proceeds as described in [Elixir](modules/elixir.md):

1. The actor LLM drafts an implementation against the contract.
2. A structural validator checks function formatting and safety.
3. A critic model reviews it for statistical soundness--correct age conditioning, an
   honest treatment of the small-sample/selection-bias caveat, and no unwarranted leap
   from association to causation.
4. The actor revises based on critic feedback; this repeats until an attempt is
   approved or the iteration/time budget runs out. If no attempt is approved, that is
   reported explicitly rather than returning an unvetted result.

With the new analysis scripts in hand, the **Engine** and **Runner** execute the analysis,
pulling in the latest FrameState and/or observation data as needed.

## 9. Findings

The approved analysis reports a smaller exercise-health association among older
participants than younger ones, with an explicit caveat that the small, self-selected
older subgroup limits how strongly that should be interpreted.
