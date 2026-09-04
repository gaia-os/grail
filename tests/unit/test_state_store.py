"""Append-only observation and posterior persistence in SQLite."""


from pathlib import Path

import pytest
import torch

from grail.frame.state import FrameStateStore

FRAME = "coin-model"
HASH = "definition-hash-a"
OTHER_HASH = "definition-hash-b"


@pytest.fixture
def store(tmp_path: Path) -> FrameStateStore:
    return FrameStateStore(tmp_path / "state.sqlite3")


def test_appended_values_are_returned_in_submission_order(store: FrameStateStore):
    store.append_observations(FRAME, HASH, "Toss", [1, 0], batch_id="first")
    store.append_observations(FRAME, HASH, "Toss", [1], batch_id="second")

    batches = store.get_observation_batches(FRAME, HASH)

    assert [batch.id for batch in batches] == ["first", "second"]
    assert [batch.values for batch in batches] == [[1, 0], [1]]


def test_values_keep_their_ordinal_order_within_a_batch(store: FrameStateStore):
    values = [3, 1, 2, 5, 4]
    store.append_observations(FRAME, HASH, "Score", values, batch_id="ordered")

    assert store.get_observation_batches(FRAME, HASH)[0].values == values


def test_replaying_an_identical_batch_is_idempotent(store: FrameStateStore):
    first = store.append_observations(FRAME, HASH, "Toss", [1, 0], batch_id="retry")
    second = store.append_observations(FRAME, HASH, "Toss", [1, 0], batch_id="retry")

    assert first.id == second.id
    assert len(store.get_observation_batches(FRAME, HASH)) == 1


def test_reusing_a_batch_id_with_different_content_is_rejected(store: FrameStateStore):
    store.append_observations(FRAME, HASH, "Toss", [1, 0], batch_id="conflict")

    with pytest.raises(ValueError, match="already exists with different content"):
        store.append_observations(FRAME, HASH, "Toss", [0, 0], batch_id="conflict")


def test_batches_can_be_filtered_by_variable(store: FrameStateStore):
    store.append_observations(FRAME, HASH, "Toss", [1], batch_id="toss")
    store.append_observations(FRAME, HASH, "Height", [1.8], batch_id="height")

    filtered = store.get_observation_batches(FRAME, HASH, variable_name="Height")

    assert [batch.id for batch in filtered] == ["height"]


def test_a_changed_definition_hash_gets_its_own_namespace(store: FrameStateStore):
    """Editing a Frame's YAML must not silently reuse the old model's evidence."""
    store.append_observations(FRAME, HASH, "Toss", [1, 1], batch_id="original")

    assert store.get_observation_batches(FRAME, OTHER_HASH) == []
    assert len(store.get_observation_batches(FRAME, HASH)) == 1


def test_tensor_values_are_normalized_into_plain_json(store: FrameStateStore):
    batch = store.append_observations(
        FRAME, HASH, "Toss", torch.tensor([1.0, 0.0]), batch_id="tensor"
    )

    assert batch.values == [1.0, 0.0]


def test_a_scalar_value_is_stored_as_a_single_element_batch(store: FrameStateStore):
    batch = store.append_observations(FRAME, HASH, "Toss", 1, batch_id="scalar")

    assert batch.values == [1]


def test_non_finite_and_unserializable_values_are_rejected(store: FrameStateStore):
    with pytest.raises(ValueError):
        store.append_observations(FRAME, HASH, "Toss", [float("nan")], batch_id="nan")

    with pytest.raises(TypeError, match="not JSON serializable"):
        store.append_observations(FRAME, HASH, "Toss", [object()], batch_id="object")


def test_variable_name_and_source_must_be_non_empty(store: FrameStateStore):
    with pytest.raises(ValueError, match="variable_name must be a non-empty string"):
        store.append_observations(FRAME, HASH, "", [1])

    with pytest.raises(ValueError, match="source must be a non-empty string"):
        store.append_observations(FRAME, HASH, "Toss", [1], source="  ")


def test_a_generated_batch_id_is_returned_when_none_is_supplied(store: FrameStateStore):
    batch = store.append_observations(FRAME, HASH, "Toss", [1])

    assert batch.id
    assert store.get_observation_batches(FRAME, HASH)[0].id == batch.id


def test_saving_a_posterior_twice_updates_the_same_snapshot(store: FrameStateStore):
    prior = {"distribution": "beta", "params": {"alpha": 1.0, "beta": 1.0}}
    store.save_posterior(
        FRAME, HASH, variable_name="Theta", strategy="exact", distribution="beta",
        prior=prior, params={"alpha": 2.0, "beta": 1.0}, metadata={"n": 1},
    )
    store.save_posterior(
        FRAME, HASH, variable_name="Theta", strategy="exact", distribution="beta",
        prior=prior, params={"alpha": 5.0, "beta": 3.0}, metadata={"n": 6},
    )

    posterior = store.get_posterior(FRAME, HASH, variable_name="Theta")

    assert posterior.params == {"alpha": 5.0, "beta": 3.0}
    assert posterior.metadata == {"n": 6}
    assert len(store.get_posteriors(FRAME, HASH)) == 1


def test_posteriors_are_retrievable_per_strategy(store: FrameStateStore):
    prior = {"distribution": "beta", "params": {}}
    for strategy, alpha in (("exact", 5.0), ("svi", 4.7)):
        store.save_posterior(
            FRAME, HASH, variable_name="Theta", strategy=strategy, distribution="beta",
            prior=prior, params={"alpha": alpha}, metadata={},
        )

    assert store.get_posterior(FRAME, HASH, variable_name="Theta", strategy="exact").params == {
        "alpha": 5.0
    }
    # Without a strategy the most recently updated snapshot wins.
    assert store.get_posterior(FRAME, HASH, variable_name="Theta").strategy == "svi"


def test_get_posteriors_returns_one_entry_per_variable(store: FrameStateStore):
    prior = {"distribution": "beta", "params": {}}
    for variable in ("Theta", "Phi"):
        store.save_posterior(
            FRAME, HASH, variable_name=variable, strategy="exact", distribution="beta",
            prior=prior, params={"alpha": 1.0}, metadata={},
        )

    assert set(store.get_posteriors(FRAME, HASH)) == {"Theta", "Phi"}


def test_a_missing_posterior_reads_back_as_none(store: FrameStateStore):
    assert store.get_posterior(FRAME, HASH, variable_name="Absent") is None
    assert store.get_posteriors(FRAME, HASH) == {}


def test_state_survives_reopening_the_database(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    FrameStateStore(database).append_observations(
        FRAME, HASH, "Toss", [1, 0], batch_id="persisted"
    )

    reopened = FrameStateStore(database)

    assert [b.id for b in reopened.get_observation_batches(FRAME, HASH)] == ["persisted"]


def test_run_records_can_be_created_and_completed(store: FrameStateStore):
    created = store.create_run_record(
        FRAME,
        HASH,
        strategy_id="beta-bernoulli-exact",
        observation_batch_ids=["batch-001"],
        parameters={"n_steps": 10},
        versions={"runner": 1},
    )

    completed = store.mark_run_succeeded(
        created["id"],
        diagnostics={"posterior_count": 1},
        artifact_paths={"metadata": "data/runs/abc/metadata.json"},
    )

    assert completed["status"] == "succeeded"
    assert completed["diagnostics"]["posterior_count"] == 1
    assert completed["artifact_paths"]["metadata"].endswith("metadata.json")


def test_run_record_failures_are_persisted(store: FrameStateStore):
    created = store.create_run_record(FRAME, HASH, strategy_id="bad-strategy")

    failed = store.mark_run_failed(
        created["id"],
        error_type="RuntimeError",
        error_message="explode",
        error_traceback="trace",
        diagnostics={"posterior_count": 0},
    )

    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "explode"


def test_run_records_are_scoped_by_definition_hash(store: FrameStateStore):
    store.create_run_record(FRAME, HASH, strategy_id="exact", run_id="run-a")

    assert store.get_run_record(FRAME, OTHER_HASH, "run-a") is None
    assert len(store.list_run_records(FRAME, HASH)) == 1
    assert store.list_run_records(FRAME, OTHER_HASH) == []


def test_run_records_survive_reopening_the_database(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    first = FrameStateStore(database)
    created = first.create_run_record(FRAME, HASH, strategy_id="exact")
    first.mark_run_succeeded(created["id"], diagnostics={"ok": True})

    reopened = FrameStateStore(database)
    records = reopened.list_run_records(FRAME, HASH)

    assert len(records) == 1
    assert records[0]["status"] == "succeeded"
