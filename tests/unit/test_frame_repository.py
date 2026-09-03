"""YAML persistence, path safety, and state-store wiring for Frames."""


from pathlib import Path

import pytest

from grail.frame import Frame, FrameRepository, FrameSpec


def test_a_saved_frame_reloads_as_an_equivalent_spec(repository: FrameRepository):
    frame = Frame("round-trip")
    cause_id = frame.add_variable("Cause", "bernoulli", {"theta": 0.5})
    frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 0.1})
    frame.add_dependency("Cause", "Effect")

    path = repository.save(frame)

    assert path.name == "round-trip.yaml"
    assert repository.load_spec("round-trip.yaml") == frame.to_spec()


def test_declarative_references_survive_a_save_and_load(repository: FrameRepository):
    frame = Frame("references")
    cause_id = frame.add_variable("Cause", "bernoulli", {"theta": 0.5})
    frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 0.1})
    frame.add_dependency("Cause", "Effect")
    repository.save(frame)

    reloaded = repository.load("references.yaml")

    # The reference is portable as a name in YAML, but a node ID at runtime.
    reloaded_cause_id = reloaded.get_variable_id("Cause")
    assert reloaded.get_variable("Effect").get_distribution_params()["loc"] == reloaded_cause_id


def test_nested_directories_are_created_on_save(repository: FrameRepository):
    frame = Frame("health-model")
    frame.add_variable("Cause", "bernoulli", {"theta": 0.5})

    path = repository.save(frame, "examples/health_model.yaml")

    assert path == repository.root / "examples" / "health_model.yaml"
    assert repository.load_spec("examples/health_model.yaml").name == "health-model"


@pytest.mark.parametrize("filename", ["plain", "plain.yaml", "plain.yml"])
def test_a_missing_or_alternate_suffix_resolves_to_yaml(
    repository: FrameRepository, filename: str
):
    expected = "plain.yml" if filename.endswith(".yml") else "plain.yaml"

    assert repository.path_for("plain").name == "plain.yaml"
    assert repository._resolve_path(filename).name == expected


def test_paths_outside_the_repository_root_are_rejected(repository: FrameRepository):
    with pytest.raises(ValueError, match="must be within"):
        repository.load_spec("../escape.yaml")


def test_malformed_yaml_is_reported_with_its_path(repository: FrameRepository):
    (repository.root / "broken.yaml").write_text("name: [unclosed", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Frame YAML"):
        repository.load_spec("broken.yaml")


def test_an_empty_document_is_rejected(repository: FrameRepository):
    (repository.root / "empty.yaml").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        repository.load_spec("empty.yaml")


def test_a_non_mapping_document_is_rejected(repository: FrameRepository):
    (repository.root / "list.yaml").write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(TypeError, match="was not loaded as dict object"):
        repository.load_spec("list.yaml")


def test_loading_attaches_a_state_store(repository: FrameRepository, coin_frame: Frame):
    # A Frame without a store cannot record evidence at all.
    detached = Frame.from_spec(coin_frame.to_spec())
    with pytest.raises(RuntimeError, match="has no state store"):
        detached.record_observations("Toss", [1])

    assert coin_frame.record_observations("Toss", [1], batch_id="b1") == "b1"


def test_spec_observations_are_copied_into_the_ledger_once(repository: FrameRepository):
    spec = FrameSpec.model_validate(
        {
            "version": 1,
            "name": "seeded",
            "variables": [
                {
                    "name": "Toss",
                    "distribution": "bernoulli",
                    "params": {"theta": 0.5},
                    "observations": [1, 0, 1],
                }
            ],
        }
    )
    repository.save(spec)

    first = repository.load("seeded.yaml")
    second = repository.load("seeded.yaml")

    batches = second.get_observation_batches("Toss")
    assert len(batches) == 1
    assert batches[0].values == [1, 0, 1]
    assert batches[0].source == "frame-spec"
    assert first.definition_hash == second.definition_hash


def test_saving_is_atomic_and_leaves_no_temporary_file(repository: FrameRepository):
    frame = Frame("atomic")
    frame.add_variable("Cause", "bernoulli", {"theta": 0.5})

    repository.save(frame)

    assert [path.name for path in repository.root.iterdir()] == ["atomic.yaml"]


def test_a_repository_creates_its_root_directory(tmp_path: Path):
    root = tmp_path / "does" / "not" / "exist"

    repository = FrameRepository(root, state_database_path=tmp_path / "state.sqlite3")

    assert repository.root.is_dir()
