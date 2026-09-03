"""Executes the documented walkthrough in docs/illustrative_health_example.md.

Each test corresponds to a numbered step so that the narrative and the shipped
example data cannot drift apart unnoticed.
"""


from pathlib import Path

import pytest
import torch

from grail.engine import Engine
from grail.frame import Frame, FrameRepository
from grail.inference import BetaBernoulliInference
from grail.runner import Runner

OBSERVATION_FILE = Path("data") / "observations" / "examples" / "health_model.json"


@pytest.fixture
def health_frame(project_root: Path, tmp_path: Path) -> Frame:
    """The committed example Frame, with runtime state kept in a temp database."""
    repository = FrameRepository(
        project_root / "data" / "frames",
        state_database_path=tmp_path / "state.sqlite3",
    )
    return repository.load("examples/health_model.yaml")


def test_step_2_the_example_frame_declares_the_documented_structure(health_frame: Frame):
    assert health_frame.name == "health-model"
    assert {variable.name for variable in health_frame.get_variables()} == {
        "ExerciseRate",
        "Exercise",
        "Health",
    }
    # ExerciseRate -> Exercise -> Health
    order = [health_frame.get_variable(node_id).name for node_id in health_frame.graph.topological_sort()]
    assert order == ["ExerciseRate", "Exercise", "Health"]


def test_step_3_prior_predictive_sampling_stays_in_a_plausible_range(health_frame: Frame):
    samples = Runner(Engine(health_frame).get_model()).simulate(num_samples=500)

    rate = samples["ExerciseRate"]
    assert ((rate >= 0.0) & (rate <= 1.0)).all()
    # A Beta(1,1) prior is uniform, so the rate should be genuinely uncertain.
    assert float(rate.mean()) == pytest.approx(0.5, abs=0.1)
    # Health ~ Normal(Exercise, 0.1), so it clusters near 0 and 1.
    assert float(samples["Health"].min()) > -1.0
    assert float(samples["Health"].max()) < 2.0


def test_step_4_recording_observations_creates_one_durable_batch(
    health_frame: Frame, project_root: Path
):
    batch_ids = health_frame.load_observations(project_root / OBSERVATION_FILE)

    assert batch_ids == ["health-model-exercise-001"]
    batch = health_frame.get_observation_batches("Exercise")[0]
    assert batch.values == [1, 1, 1, 1, 1, 1, 1, 0]
    assert batch.source == "example:health-model"


def test_step_5_the_conjugate_update_sharpens_the_exercise_rate(
    health_frame: Frame, project_root: Path
):
    health_frame.load_observations(project_root / OBSERVATION_FILE)
    runner = Runner(Engine(health_frame).get_model(), frame=health_frame)

    posterior = runner.infer(BetaBernoulliInference())["ExerciseRate"]

    # Beta(1,1) + seven successes and one failure = Beta(8,2), a mean of 0.8.
    assert posterior.params == {"alpha": 8.0, "beta": 2.0}
    assert posterior.params["alpha"] / sum(posterior.params.values()) == pytest.approx(0.8)
    assert posterior.metadata["processed_batch_ids"] == ["health-model-exercise-001"]


def test_step_5_the_update_is_recorded_on_the_frame_state(
    health_frame: Frame, project_root: Path
):
    health_frame.load_observations(project_root / OBSERVATION_FILE)
    Runner(Engine(health_frame).get_model(), frame=health_frame).infer(BetaBernoulliInference())

    state = health_frame.inspect_state()

    assert state.variables["ExerciseRate"].posterior.params == {"alpha": 8.0, "beta": 2.0}
    assert "posterior: beta(alpha=8.0, beta=2.0)" in state.format()
    assert state.variables["Exercise"].posterior is None


def test_step_6_intervening_on_exercise_raises_simulated_health(health_frame: Frame):
    """do(Exercise=1) should lift Health relative to the un-intervened cohort."""
    runner = Runner(Engine(health_frame).get_model(), frame=health_frame)

    observational = runner.simulate(num_samples=400)
    intervened = runner.do_operation({"Exercise": 1.0}, num_samples=400)

    assert torch.allclose(intervened["Exercise"], torch.ones(400))
    assert float(intervened["Health"].mean()) == pytest.approx(1.0, abs=0.05)
    assert float(intervened["Health"].mean()) > float(observational["Health"].mean())


def test_the_documented_posterior_is_reproducible_from_a_reloaded_frame(
    project_root: Path, tmp_path: Path
):
    """Steps 4-5 must give the same answer after the Frame is rebuilt from YAML."""
    repository = FrameRepository(
        project_root / "data" / "frames", state_database_path=tmp_path / "state.sqlite3"
    )
    first = repository.load("examples/health_model.yaml")
    first.load_observations(project_root / OBSERVATION_FILE)
    Runner(Engine(first).get_model(), frame=first).infer(BetaBernoulliInference())

    reloaded = repository.load("examples/health_model.yaml")

    assert reloaded.get_posterior("ExerciseRate").params == {"alpha": 8.0, "beta": 2.0}
