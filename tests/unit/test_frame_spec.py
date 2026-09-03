"""Validation rules enforced on declarative Frame YAML before compilation."""


from typing import Any

from pydantic import ValidationError
import pytest

from grail.frame import FrameSpec, VariableSpec


def _spec_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 1,
        "name": "valid-frame",
        "variables": [
            {"name": "A", "distribution": "normal", "params": {"loc": 0, "scale": 1}},
            {
                "name": "B",
                "distribution": "normal",
                "params": {"loc": {"$ref": "A"}, "scale": 1},
            },
        ],
        "dependencies": [{"source": "A", "target": "B"}],
    }
    payload.update(overrides)
    return payload


def test_a_well_formed_spec_validates():
    spec = FrameSpec.model_validate(_spec_payload())

    assert spec.name == "valid-frame"
    assert [variable.name for variable in spec.variables] == ["A", "B"]
    assert spec.dependencies[0].source == "A"


def test_unsupported_versions_are_rejected():
    with pytest.raises(ValidationError, match="unsupported Frame spec version 2"):
        FrameSpec.model_validate(_spec_payload(version=2))


def test_variable_names_must_be_unique():
    payload = _spec_payload(
        variables=[
            {"name": "A", "distribution": "normal", "params": {}},
            {"name": "A", "distribution": "normal", "params": {}},
        ],
        dependencies=[],
    )

    with pytest.raises(ValidationError, match=r"duplicates: \['A'\]"):
        FrameSpec.model_validate(payload)


def test_dependencies_must_reference_declared_variables():
    payload = _spec_payload(
        dependencies=[{"source": "A", "target": "B"}, {"source": "A", "target": "Ghost"}]
    )

    with pytest.raises(ValidationError, match="must reference declared variables"):
        FrameSpec.model_validate(payload)


def test_duplicate_dependencies_are_rejected():
    payload = _spec_payload(
        dependencies=[{"source": "A", "target": "B"}, {"source": "A", "target": "B"}]
    )

    with pytest.raises(ValidationError, match="dependencies must be unique"):
        FrameSpec.model_validate(payload)


def test_self_dependencies_are_rejected():
    payload = _spec_payload(dependencies=[{"source": "A", "target": "A"}])

    with pytest.raises(ValidationError, match="cannot target the same variable"):
        FrameSpec.model_validate(payload)


def test_references_must_name_a_declared_variable():
    payload = _spec_payload(
        variables=[
            {"name": "A", "distribution": "normal", "params": {"loc": {"$ref": "Ghost"}}},
        ],
        dependencies=[],
    )

    with pytest.raises(ValidationError, match=r"references unknown variables: \['Ghost'\]"):
        FrameSpec.model_validate(payload)


def test_every_reference_needs_a_matching_dependency_edge():
    with pytest.raises(ValidationError, match="without matching dependencies"):
        FrameSpec.model_validate(_spec_payload(dependencies=[]))


def test_cycles_are_reported_with_the_offending_path():
    payload = _spec_payload(
        dependencies=[{"source": "A", "target": "B"}, {"source": "B", "target": "A"}]
    )

    with pytest.raises(ValidationError, match="cycle detected"):
        FrameSpec.model_validate(payload)


@pytest.mark.parametrize("name", ["1leading", "has-hyphen", "has space", "", "A" * 51])
def test_variable_names_follow_the_identifier_pattern(name: str):
    with pytest.raises(ValidationError):
        VariableSpec.model_validate({"name": name, "distribution": "normal"})


def test_unknown_fields_are_forbidden():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FrameSpec.model_validate(_spec_payload(unexpected="value"))


def test_at_least_one_variable_is_required():
    with pytest.raises(ValidationError):
        FrameSpec.model_validate(_spec_payload(variables=[], dependencies=[]))


def test_surrounding_whitespace_is_stripped_from_identifiers():
    spec = FrameSpec.model_validate(
        {
            "version": 1,
            "name": "  spaced-frame  ",
            "variables": [{"name": " A ", "distribution": " normal "}],
        }
    )

    assert spec.name == "spaced-frame"
    assert spec.variables[0].name == "A"
    assert spec.variables[0].distribution == "normal"


def test_referenced_variables_are_found_at_any_nesting_depth():
    variable = VariableSpec.model_validate(
        {
            "name": "Deep",
            "distribution": "normal",
            "params": {
                "loc": [{"$ref": "A"}, {"nested": {"$ref": "B"}}],
                "scale": 1.0,
            },
        }
    )

    assert variable.referenced_variables() == {"A", "B"}


def test_reference_values_must_be_non_empty_strings():
    with pytest.raises(ValidationError, match="non-empty variable names"):
        FrameSpec.model_validate(
            _spec_payload(
                variables=[
                    {"name": "A", "distribution": "normal", "params": {"loc": {"$ref": 5}}}
                ],
                dependencies=[],
            )
        )
