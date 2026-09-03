"""Runtime Frame construction, graph integrity, and spec round-tripping."""


from pathlib import Path

import pytest

from grail.frame import Frame, FrameRepository, FrameSpec


def test_yaml_spec_compiles_with_declarations_in_any_order(repository: FrameRepository):
    """A variable may be declared before the parent its parameters reference."""
    spec = FrameSpec.model_validate(
        {
            "version": 1,
            "name": "ordered-later",
            "variables": [
                {
                    "name": "Outcome",
                    "distribution": "normal",
                    "params": {"loc": {"$ref": "Cause"}, "scale": 0.1},
                },
                {"name": "Cause", "distribution": "bernoulli", "params": {"theta": 0.5}},
            ],
            "dependencies": [{"source": "Cause", "target": "Outcome"}],
        }
    )

    path = repository.save(spec)
    frame = repository.load(path.name)

    cause_id = frame.get_variable_id("Cause")
    outcome_id = frame.get_variable_id("Outcome")
    assert frame.graph.topological_sort() == [cause_id, outcome_id]
    assert frame.get_variable(outcome_id).get_distribution_params()["loc"] == cause_id
    assert frame.to_spec() == spec


def test_runtime_frame_protects_graph_integrity():
    frame = Frame("runtime-frame")
    cause_id = frame.add_variable("Cause", "normal", {"loc": 0, "scale": 1})
    effect_id = frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 1})
    cause_variable = frame.get_variable(cause_id)
    effect_variable = frame.get_variable(effect_id)
    assert frame.get_variable("Cause") is cause_variable

    all_variables = frame.get_variables()
    assert {variable.name for variable in all_variables} == {"Cause", "Effect"}

    with pytest.raises(KeyError, match="no variable named or id 'Missing'"):
        frame.add_dependency("Missing", effect_id)

    frame.add_dependency(cause_variable, effect_variable)
    with pytest.raises(ValueError, match="acyclic"):
        frame.add_dependency(effect_variable, cause_variable)

    assert frame.graph.graph.number_of_edges() == 1
    assert frame.graph.graph.has_edge(cause_id, effect_id)


def test_dependencies_accept_names_ids_and_variable_objects():
    frame = Frame("endpoint-styles")
    frame.add_variable("A", "normal", {"loc": 0, "scale": 1})
    b_id = frame.add_variable("B", "normal", {"loc": 0, "scale": 1})
    frame.add_variable("C", "normal", {"loc": 0, "scale": 1})

    frame.add_dependency("A", b_id)
    frame.add_dependency(frame.get_variable("B"), "C")

    assert frame.graph.graph.number_of_edges() == 2


def test_an_unbound_variable_cannot_anchor_a_dependency():
    from grail.frame.variable import Variable

    frame = Frame("unbound")
    frame.add_variable("Real", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(KeyError, match="not bound to a node"):
        frame.add_dependency(Variable(name="Detached"), "Real")


def test_a_variable_from_another_frame_is_rejected():
    first = Frame("first")
    first.add_variable("Shared", "normal", {"loc": 0, "scale": 1})
    second = Frame("second")
    second.add_variable("Shared", "normal", {"loc": 0, "scale": 1})
    second.add_variable("Target", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(KeyError, match="does not match runtime node"):
        second.add_dependency(first.get_variable("Shared"), "Target")


def test_runtime_frame_validates_variable_names_and_uniqueness():
    frame = Frame("name-validation")
    frame.add_variable("Valid_1", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="already contains variable"):
        frame.add_variable("Valid_1", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="must be non-empty"):
        frame.add_variable("   ", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="must start with a letter"):
        frame.add_variable("1bad", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="must start with a letter"):
        frame.add_variable("bad-name", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(ValueError, match="<= 50 chars"):
        frame.add_variable("A" * 51, "normal", {"loc": 0, "scale": 1})

    with pytest.raises(TypeError, match="must be a string"):
        frame.add_variable(42, "normal", {"loc": 0, "scale": 1})


def test_lookup_of_an_unknown_variable_reports_the_name():
    frame = Frame("lookup")
    frame.add_variable("Known", "normal", {"loc": 0, "scale": 1})

    with pytest.raises(KeyError, match="no variable named or id 'Unknown'"):
        frame.get_variable("Unknown")

    with pytest.raises(KeyError, match="no variable named 'Unknown'"):
        frame.get_variable_id("Unknown")


def test_definition_hash_tracks_the_declarative_definition():
    def _frame(scale: float) -> Frame:
        frame = Frame("hashed")
        frame.add_variable("Outcome", "normal", {"loc": 0.0, "scale": scale})
        return frame

    assert _frame(1.0).definition_hash == _frame(1.0).definition_hash
    assert _frame(1.0).definition_hash != _frame(2.0).definition_hash


def test_spec_round_trip_preserves_metadata_and_attributes():
    frame = Frame("rich-frame")
    frame.metadata = {"description": "Documented.", "tags": ["demo"]}
    frame.add_variable(
        "Outcome",
        "normal",
        {"loc": 0.0, "scale": 1.0},
        description="An outcome.",
        attributes={"unit": "score"},
    )

    rebuilt = Frame.from_spec(frame.to_spec())

    assert rebuilt.to_spec() == frame.to_spec()
    assert rebuilt.get_variable("Outcome").attributes == {"unit": "score"}
    assert rebuilt.get_variable("Outcome").description == "An outcome."


def test_state_operations_require_an_attached_store():
    frame = Frame("stateless")
    frame.add_variable("Toss", "bernoulli", {"theta": 0.5})

    with pytest.raises(RuntimeError, match="has no state store"):
        frame.inspect_state()


def test_observation_files_are_validated_before_import(coin_frame: Frame, tmp_path: Path):
    path = tmp_path / "observations.json"

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid observation JSON"):
        coin_frame.load_observations(path)

    path.write_text('{"frame": "other-model", "batches": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="is for Frame 'other-model'"):
        coin_frame.load_observations(path)

    path.write_text('{"frame": "coin-model", "batches": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'batches' list"):
        coin_frame.load_observations(path)

    path.write_text(
        '{"batches": [{"variable": "Toss", "values": [1], "oops": 1}]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"unsupported fields: \['oops'\]"):
        coin_frame.load_observations(path)

    path.write_text('{"batches": [{"variable": "Toss"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="requires 'variable' and 'values'"):
        coin_frame.load_observations(path)


def test_inspect_state_reports_priors_evidence_and_posteriors(coin_frame: Frame):
    coin_frame.record_observations("Toss", [1, 0], batch_id="b1")

    state = coin_frame.inspect_state()

    assert state.frame_name == "coin-model"
    assert state.variables["Theta"].prior == {
        "distribution": "beta",
        "params": {"alpha": 1.0, "beta": 1.0},
    }
    # Runtime node IDs are restored to portable $ref form for inspection.
    assert state.variables["Toss"].prior["params"]["theta"] == {"$ref": "Theta"}
    assert state.variables["Toss"].observation_batches[0].values == [1, 0]
    assert state.variables["Theta"].posterior is None
    assert "observations=2" in state.format()
    assert "posterior: unavailable" in state.format()
