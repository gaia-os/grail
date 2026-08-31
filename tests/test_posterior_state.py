import json
from pathlib import Path

import pytest

from grail.engine import Engine
from grail.frame import FrameRepository
from grail.inference import BetaBernoulliInference
from grail.runner import Runner


def _write_beta_bernoulli_spec(path: Path, *, alpha: float = 1.0, beta: float = 1.0) -> None:
    path.write_text(
        f"""
version: 1
name: coin-model
variables:
  - name: Theta
    distribution: beta
    params:
      alpha: {alpha}
      beta: {beta}
  - name: Toss
    distribution: bernoulli
    params:
      theta:
        $ref: Theta
dependencies:
  - source: Theta
    target: Toss
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _make_frame(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_beta_bernoulli_spec(frames_dir / "coin-model.yaml")
    repository = FrameRepository(frames_dir)
    return repository, repository.load("coin-model.yaml")


def test_exact_beta_bernoulli_posterior_persists_and_resumes(tmp_path: Path):
    repository, frame = _make_frame(tmp_path)
    runner = Runner(Engine(frame).get_model(), frame=frame)

    first_batch_id = frame.record_observations("Toss", [1, 1, 1, 0], batch_id="toss-001")
    assert first_batch_id == "toss-001"
    first_result = runner.infer(BetaBernoulliInference())
    assert first_result["Theta"].params == {"alpha": 4.0, "beta": 2.0}
    assert first_result["Theta"].metadata["observation_count"] == 4

    # The exact same upload is an idempotent retry, not a duplicate observation.
    assert frame.record_observations("Toss", [1, 1, 1, 0], batch_id="toss-001") == "toss-001"
    assert runner.infer(BetaBernoulliInference())["Theta"].params == {
        "alpha": 4.0,
        "beta": 2.0,
    }

    # Rebuild the Frame from YAML to verify that only state, not the in-memory
    # Variable object, enables the next incremental update.
    resumed_frame = repository.load("coin-model.yaml")
    resumed_runner = Runner(Engine(resumed_frame).get_model(), frame=resumed_frame)
    resumed_frame.record_observations("Toss", [1, 0], batch_id="toss-002", source="script")
    posterior = resumed_runner.infer(BetaBernoulliInference())["Theta"]

    assert posterior.params == {"alpha": 5.0, "beta": 3.0}
    assert posterior.metadata["successes"] == 4
    assert posterior.metadata["failures"] == 2
    assert posterior.metadata["processed_batch_ids"] == ["toss-001", "toss-002"]

    state = resumed_frame.inspect_state()
    assert [batch.id for batch in state.variables["Toss"].observation_batches] == [
        "toss-001",
        "toss-002",
    ]
    assert state.variables["Theta"].posterior == posterior
    assert "posterior: beta(alpha=5.0, beta=3.0)" in state.format()
    assert json.loads(state.to_json())["variables"]["Theta"]["posterior"]["params"] == {
        "alpha": 5.0,
        "beta": 3.0,
    }


def test_observation_json_uploads_to_sqlite_ledger(tmp_path: Path):
    _, frame = _make_frame(tmp_path)
    observation_file = tmp_path / "observations.json"
    observation_file.write_text(
        json.dumps(
            {
                "frame": "coin-model",
                "batches": [
                    {
                        "id": "file-001",
                        "variable": "Toss",
                        "values": [1, 1, 1, 1, 1, 1, 1, 0],
                        "source": "file:test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert frame.load_observations(observation_file) == ["file-001"]
    posterior = Runner(Engine(frame).get_model(), frame=frame).infer(BetaBernoulliInference())["Theta"]

    assert posterior.params == {"alpha": 8.0, "beta": 2.0}
    assert frame.get_observation_batches("Toss")[0].values == [1, 1, 1, 1, 1, 1, 1, 0]


def test_health_example_observations_produce_documented_posterior(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    repository = FrameRepository(
        project_root / "data" / "frames",
        state_database_path=tmp_path / "state.sqlite3",
    )
    frame = repository.load("examples/health_model.yaml")

    assert frame.load_observations(
        project_root / "data" / "observations" / "examples" / "health_model.json"
    ) == ["health-model-exercise-001"]
    posterior = Runner(Engine(frame).get_model(), frame=frame).infer(BetaBernoulliInference())["ExerciseRate"]

    assert posterior.params == {"alpha": 8.0, "beta": 2.0}
    assert posterior.params["alpha"] / sum(posterior.params.values()) == pytest.approx(0.8)


def test_exact_updater_requires_compatible_likelihood_and_runner_frame(tmp_path: Path):
    _, frame = _make_frame(tmp_path)

    with pytest.raises(ValueError, match="requires a Frame"):
        Runner(Engine(frame).get_model()).infer(BetaBernoulliInference())

    frame.record_observations("Toss", [1], batch_id="invalid-001")
    frame.get_variable("Toss").set_distribution("normal", {"loc": 0.0, "scale": 1.0})

    assert Runner(Engine(frame).get_model(), frame=frame).infer(BetaBernoulliInference()) == {}
