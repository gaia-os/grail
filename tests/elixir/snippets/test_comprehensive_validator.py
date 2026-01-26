import pytest
from pydantic import ValidationError
from grail.elixir import ElixirValidator, ElixirException
from grail.elixir.validator import ElixirInputVar, ElixirOutputVar, GREEN_IMPORTS, INSECURE_METHODS

# --- Scenarios for Base Code Validator ---

def test_allowed_imports_comprehensive():
    """
    Test that all 'green' imports are actually allowed.
    """
    for lib in GREEN_IMPORTS:
        code = f"""
import {lib}
import {lib} as alias
from {lib} import something
"""
        # Should not raise
        ElixirValidator.base_code_validator(code)

def test_disallowed_imports_comprehensive():
    """
    Test a variety of disallowed imports.
    """
    bad_imports = [
        "os", "sys", "subprocess", "shutil", "ctypes", "socket",
        "requests", "http.client", "urllib", "pathlib"
    ]

    for lib in bad_imports:
        code = f"import {lib}"
        with pytest.raises(ElixirException, match=f"Import '{lib}' is not allowed"):
            ElixirValidator.base_code_validator(code)

        code_alias = f"import {lib} as foo"
        with pytest.raises(ElixirException, match=f"Import '{lib}' is not allowed"):
            ElixirValidator.base_code_validator(code_alias)

def test_insecure_methods_comprehensive():
    """
    Test usage of insecure methods.
    """
    # bad_calls = [
    #     ("eval", 'eval("1+1")'),
    #     ("exec", 'exec("val = 1")'),
    #     ("open", 'open("file", "w")'),
    #     ("globals", 'globals()'),
    #     ("locals", 'locals()'),
    #     ("input", 'input("Enter something: ")'),
    #     ("getattr", 'getattr(obj, "name")'),
    #     ("setattr", 'setattr(obj, "name", "val")'),
    #     ("delattr", 'delattr(obj, "name")')
    # ]

    for func_name in INSECURE_METHODS:
        code = f"""
def outer():
    {func_name}()
"""
        with pytest.raises(ElixirException, match=f"Usage of '{func_name}' is not allowed"):
            ElixirValidator.base_code_validator(code)


# --- Scenarios for Function Meta Validator ---

class MockFunctionValidator(ElixirValidator):
    """
    A concrete validator for testing function limitations.
    """
    is_function = True
    function_name = "calculate_metric"
    required_args = (
        ElixirInputVar(arg="data", description="Data points", datatype=list),
        ElixirInputVar(arg="threshold", description="Cutoff", datatype=float),
    )
    returned_data = (
        ElixirOutputVar(name="score", description="Resulting score", datatype=float),
    )


def test_valid_function_structure():
    code = """
import numpy as np

def calculate_metric(data, threshold):
    return np.mean(data) * threshold
"""
    # This should pass validation
    validator = MockFunctionValidator(code=code)
    assert validator.code == code


def test_invalid_function_name():
    code = """
def invalid_name(data, threshold):
    return 0.0
"""
    # Pydantic validation error wraps the inner exception
    with pytest.raises((ValidationError, ElixirException)) as excinfo:
        MockFunctionValidator(code=code)

    # We check string representation of error for key phrases
    assert "Function name must be 'calculate_metric'" in str(excinfo.value)


def test_invalid_arguments_names():
    code = """
def calculate_metric(wrong_arg, threshold):
    return 0.0
"""
    with pytest.raises((ValidationError, ElixirException)) as excinfo:
        MockFunctionValidator(code=code)

    assert "Expected arguments" in str(excinfo.value)
    assert "wrong_arg" in str(excinfo.value)


def test_multiple_functions_defined():
    code = """
def calculate_metric(data, threshold):
    return 0.0

def other_func():
    pass
"""
    with pytest.raises((ValidationError, ElixirException)) as excinfo:
        MockFunctionValidator(code=code)

    assert "There must be exactly one function definition" in str(excinfo.value)


def test_return_values_count_single_expected():
    """
    MockFunctionValidator expects 1 return value.
    """
    # 1. No return value (None)
    code_none = """
def calculate_metric(data, threshold):
    return
"""
    with pytest.raises((ValidationError, ElixirException)) as excinfo:
        MockFunctionValidator(code=code_none)
    assert "Explicit 'return' with no value is not allowed" in str(excinfo.value)

    # 2. Too many return values (Tuple)
    code_tuple = """
def calculate_metric(data, threshold):
    return 1.0, 2.0
"""
    with pytest.raises((ValidationError, ElixirException)) as excinfo:
        MockFunctionValidator(code=code_tuple)
    assert "Expected 1 return values, got 2" in str(excinfo.value)


class MockMultiReturnValidator(ElixirValidator):
    is_function = True
    function_name = "split_data"
    required_args = (
        ElixirInputVar(arg="data", description="Data", datatype=list),
    )
    returned_data = (
        ElixirOutputVar(name="train", description="Train set", datatype=list),
        ElixirOutputVar(name="test", description="Test set", datatype=list),
    )

def test_return_values_count_multi_expected():
    """
    MockMultiReturnValidator expects 2 return values.
    """
    # 1. Returning single value
    code_single = """
def split_data(data):
    return [1, 2, 3]
"""
    with pytest.raises((ValidationError, ElixirException)) as excinfo:
        MockMultiReturnValidator(code=code_single)
    assert "Expected 2 return values, got 1" in str(excinfo.value)

    # 2. Returning correct tuple size
    code_correct = """
def split_data(data):
    return data[:2], data[2:]
"""
    MockMultiReturnValidator(code=code_correct)

    # 3. Returning wrong tuple size
    code_wrong_tuple = """
def split_data(data):
    return 1, 2, 3
"""
    with pytest.raises((ValidationError, ElixirException)) as excinfo:
        MockMultiReturnValidator(code=code_wrong_tuple)
    assert "Expected 2 return values, got 3" in str(excinfo.value)

