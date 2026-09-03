"""Persisted observations and resumable posterior state across Frame reloads."""


import json
from pathlib import Path

from grail.engine import Engine
from grail.frame import Frame, FrameRepository
from grail.inference import BetaBernoulliInference
from grail.runner import Runner


def test_a_posterior_survives_rebuilding_the_frame_from_yaml(
    repository: FrameRepository, coin_frame: Frame
):
    runner = Runner(Engine(coin_frame).get_model(), frame=coin_frame)
    assert coin_frame.record_observations("Toss", [1, 1, 1, 0], batch_id="toss-001") == "toss-001"

    first = runner.infer(BetaBernoulliInference())
    assert first["Theta"].params == {"alpha": 4.0, "beta": 2.0}

    # Re-uploading identical content is a retry, not new evidence.
    coin_frame.record_observations("Toss", [1, 1, 1, 0], batch_id="toss-001")
    assert runner.infer(BetaBernoulliInference())["Theta"].params == {"alpha": 4.0, "beta": 2.0}

    # Rebuild from YAML so only persisted state, not the in-memory Variable,
    # can carry the previous update forward.
    resumed = repository.load("coin-model.yaml")
    resumed_runner = Runner(Engine(resumed).get_model(), frame=resumed)
    resumed.record_observations("Toss", [1, 0], batch_id="toss-002", source="script")
    posterior = resumed_runner.infer(BetaBernoulliInference())["Theta"]

    assert posterior.params == {"alpha": 5.0, "beta": 3.0}
    assert posterior.metadata["successes"] == 4
    assert posterior.metadata["failures"] == 2
    assert posterior.metadata["processed_batch_ids"] == ["toss-001", "toss-002"]


def test_inspect_state_renders_the_full_diagnostic_snapshot(coin_frame: Frame):
    coin_frame.record_observations("Toss", [1, 1, 1, 0], batch_id="toss-001")
    coin_frame.record_observations("Toss", [1, 0], batch_id="toss-002")
    Runner(Engine(coin_frame).get_model(), frame=coin_frame).infer(BetaBernoulliInference())

    state = coin_frame.inspect_state()

    assert [batch.id for batch in state.variables["Toss"].observation_batches] == [
        "toss-001",
        "toss-002",
    ]
    assert "posterior: beta(alpha=5.0, beta=3.0)" in state.format()
    payload = json.loads(state.to_json())
    assert payload["variables"]["Theta"]["posterior"]["params"] == {"alpha": 5.0, "beta": 3.0}
    assert payload["frame"] == "coin-model"
    assert payload["definition_hash"] == coin_frame.definition_hash


def test_observations_upload_from_a_json_file(coin_frame: Frame, tmp_path: Path):
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

    assert coin_frame.load_observations(observation_file) == ["file-001"]
    posterior = Runner(Engine(coin_frame).get_model(), frame=coin_frame).infer(
        BetaBernoulliInference()
    )["Theta"]

    assert posterior.params == {"alpha": 8.0, "beta": 2.0}
    assert coin_frame.get_observation_batches("Toss")[0].source == "file:test"


def test_editing_the_frame_definition_isolates_the_earlier_posterior(
    repository: FrameRepository, coin_frame: Frame
):
    """A changed prior must not silently inherit the previous model's posterior."""
    coin_frame.record_observations("Toss", [1, 1, 1, 0], batch_id="toss-001")
    Runner(Engine(coin_frame).get_model(), frame=coin_frame).infer(BetaBernoulliInference())

    edited = repository.load("coin-model.yaml")
    edited.get_variable("Theta").set_distribution("beta", {"alpha": 5.0, "beta": 5.0})

    assert edited.definition_hash != coin_frame.definition_hash
    assert edited.get_posterior("Theta") is None
    assert edited.get_observation_batches("Toss") == []


def test_an_incompatible_likelihood_produces_no_posterior(coin_frame: Frame):
    coin_frame.record_observations("Toss", [1], batch_id="invalid-001")
    coin_frame.get_variable("Toss").set_distribution("normal", {"loc": 0.0, "scale": 1.0})

    runner = Runner(Engine(coin_frame).get_model(), frame=coin_frame)

    assert runner.infer(BetaBernoulliInference()) == {}
