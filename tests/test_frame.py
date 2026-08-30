from pathlib import Path

from pydantic import ValidationError
import pytest

from grail.engine import Engine
from grail.frame import Frame, FrameRegistry, FrameRepository, FrameSpec
from grail.runner import Runner


def test_yaml_spec_compiles_with_declarations_in_any_order(tmp_path: Path):
    repository = FrameRepository(tmp_path)
    spec = FrameSpec.model_validate(
        {
            "version": 1,
            "name": "ordered-later",
            "variables": [
                {
                    "name": "Outcome",
                    "distribution": "Normal",
                    "params": {"loc": {"$ref": "Cause"}, "scale": 0.1},
                },
                {
                    "name": "Cause",
                    "distribution": "Bernoulli",
                    "params": {"probs": 0.5},
                },
            ],
            "dependencies": [{"source": "Cause", "target": "Outcome"}],
        }
    )

    path = repository.save(spec)
    frame = repository.load(path.name)

    cause_id = frame.get_variable_id("Cause")
    outcome_id = frame.get_variable_id("Outcome")
    assert frame.graph.topological_sort() == [
        cause_id,
        outcome_id,
    ]
    assert frame.get_variable(outcome_id).get_distribution_params()["loc"] == cause_id
    assert frame.to_spec() == spec

    samples = Runner(Engine(frame).get_model()).simulate(num_samples=8)
    assert set(samples) == {"Cause", "Outcome"}
    assert samples["Outcome"].shape == (8,)


def test_frame_spec_rejects_missing_dependencies_and_cycles():
    base = {
        "version": 1,
        "name": "invalid-frame",
        "variables": [
            {"name": "A", "distribution": "Normal", "params": {"loc": 0, "scale": 1}},
            {
                "name": "B",
                "distribution": "Normal",
                "params": {"loc": {"$ref": "A"}, "scale": 1},
            },
        ],
    }

    with pytest.raises(ValidationError, match="without matching dependencies"):
        FrameSpec.model_validate(base)

    with pytest.raises(ValidationError, match="cycle detected"):
        FrameSpec.model_validate(
            {
                **base,
                "dependencies": [
                    {"source": "A", "target": "B"},
                    {"source": "B", "target": "A"},
                ],
            }
        )


def test_runtime_frame_protects_graph_integrity():
    frame = Frame("runtime-frame")
    cause_id = frame.add_variable("Cause", "Normal", {"loc": 0, "scale": 1})
    effect_id = frame.add_variable("Effect", "Normal", {"loc": cause_id, "scale": 1})
    cause_variable = frame.get_variable(cause_id)
    effect_variable = frame.get_variable(effect_id)
    assert frame.get_variable("Cause") is cause_variable
    assert frame.get_variable("Effect") is effect_variable

    all_variables = frame.get_variables()
    assert len(all_variables) == 2
    assert {variable.name for variable in all_variables} == {"Cause", "Effect"}

    with pytest.raises(KeyError, match="no variable with id 'Missing'"):
        frame.add_dependency("Missing", effect_id)

    frame.add_dependency(cause_variable, effect_variable)
    with pytest.raises(ValueError, match="acyclic"):
        frame.add_dependency(effect_variable, cause_variable)

    assert frame.graph.graph.number_of_edges() == 1
    assert frame.graph.graph.has_edge(cause_id, effect_id)


def test_sqlite_registry_indexes_yaml_without_storing_its_definition(tmp_path: Path):
    repository = FrameRepository(tmp_path / "frames")
    frame = Frame("registry-frame")
    frame.add_variable("Latent", "Normal", {"loc": 0, "scale": 1})
    frame_path = repository.save(frame)
    spec = repository.load_spec(frame_path.name)
    registry = FrameRegistry(tmp_path / "grail.sqlite3")

    first = registry.register(spec, frame_path)
    second = registry.register(spec, frame_path)

    assert first.id == second.id
    assert registry.get_by_name("registry-frame").spec_path == str(frame_path)
    assert [record.name for record in registry.list_frames()] == ["registry-frame"]


def test_runtime_frame_validates_variable_names_and_uniqueness():
    frame = Frame("name-validation")

    frame.add_variable("Valid_1", "Normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="already contains variable"):
        frame.add_variable("Valid_1", "Normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="must be non-empty"):
        frame.add_variable("   ", "Normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="must start with a letter"):
        frame.add_variable("1bad", "Normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="must start with a letter"):
        frame.add_variable("bad-name", "Normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="<= 50 chars"):
        frame.add_variable("A" * 51, "Normal", {"loc": 0, "scale": 1})
