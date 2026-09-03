"""SQLite indexing of canonical Frame YAML artifacts."""


from pathlib import Path

import pytest

from grail.frame import Frame, FrameRegistry, FrameRepository


@pytest.fixture
def registry(tmp_path: Path) -> FrameRegistry:
    return FrameRegistry(tmp_path / "registry.sqlite3")


def _saved_frame(repository: FrameRepository, name: str) -> Path:
    frame = Frame(name)
    frame.add_variable("Latent", "normal", {"loc": 0, "scale": 1})
    return repository.save(frame)


def test_registering_the_same_path_twice_updates_one_record(
    registry: FrameRegistry, repository: FrameRepository
):
    path = _saved_frame(repository, "stable-frame")
    spec = repository.load_spec(path.name)

    first = registry.register(spec, path)
    second = registry.register(spec, path)

    assert first.id == second.id
    assert len(registry.list_frames()) == 1


def test_a_changed_spec_updates_the_stored_hash(
    registry: FrameRegistry, repository: FrameRepository
):
    path = _saved_frame(repository, "changing-frame")
    original = registry.register(repository.load_spec(path.name), path)
    original_hash = original.spec_hash

    frame = repository.load(path.name)
    frame.add_variable("Extra", "normal", {"loc": 1, "scale": 1})
    repository.save(frame, path.name)
    updated = registry.register(repository.load_spec(path.name), path)

    assert updated.id == original.id
    assert updated.spec_hash != original_hash


def test_the_registry_stores_a_pointer_rather_than_the_definition(
    registry: FrameRegistry, repository: FrameRepository
):
    path = _saved_frame(repository, "pointer-frame")
    spec = repository.load_spec(path.name)

    record = registry.register(spec, path)

    assert record.spec_path == str(path)
    assert record.spec_version == spec.version
    assert not hasattr(record, "variables")


def test_lookup_by_name_returns_none_when_absent(registry: FrameRegistry):
    assert registry.get_by_name("never-registered") is None
    assert registry.list_frames() == []


def test_metadata_description_is_indexed(
    registry: FrameRegistry, repository: FrameRepository
):
    frame = Frame("described-frame")
    frame.metadata = {"description": "A documented Frame."}
    frame.add_variable("Latent", "normal", {"loc": 0, "scale": 1})
    path = repository.save(frame)

    record = registry.register(repository.load_spec(path.name), path)

    assert record.description == "A documented Frame."


def test_several_frames_are_listed_together(
    registry: FrameRegistry, repository: FrameRepository
):
    for name in ("frame-a", "frame-b"):
        path = _saved_frame(repository, name)
        registry.register(repository.load_spec(path.name), path)

    assert {record.name for record in registry.list_frames()} == {"frame-a", "frame-b"}
