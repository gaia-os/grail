"""Workspace-backed Frame construction and pickle persistence."""


from pathlib import Path

import pytest

from grail.frame.builder import Builder


@pytest.fixture
def builder(tmp_path: Path) -> Builder:
    return Builder(tmp_path / "workspace")


def test_a_new_frame_starts_empty_and_becomes_current(builder: Builder):
    frame = builder.new_frame("TestModel")

    assert frame.name == "TestModel"
    assert frame.graph.graph.number_of_nodes() == 0
    assert builder.current_frame is frame


def test_the_workspace_directory_is_created(tmp_path: Path):
    workspace = tmp_path / "does" / "not" / "exist"

    Builder(workspace)

    assert workspace.is_dir()


def test_a_frame_round_trips_through_the_workspace(builder: Builder):
    frame = builder.new_frame("Persisted")
    cause_id = frame.add_variable("Cause", "bernoulli", {"theta": 0.5})
    frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 0.1})
    frame.add_dependency("Cause", "Effect")

    builder.save_frame(frame)
    loaded = builder.load_frame("Persisted.grail")

    assert loaded.name == "Persisted"
    assert loaded.to_spec() == frame.to_spec()
    assert builder.current_frame is loaded


def test_an_explicit_filename_is_honoured(builder: Builder):
    frame = builder.new_frame("Named")
    frame.add_variable("Cause", "bernoulli", {"theta": 0.5})

    builder.save_frame(frame, "custom.pkl")

    assert (builder.workspace_path / "custom.pkl").exists()
    assert builder.load_frame("custom.pkl").name == "Named"


def test_loading_a_missing_frame_raises(builder: Builder):
    with pytest.raises(FileNotFoundError):
        builder.load_frame("absent.grail")
