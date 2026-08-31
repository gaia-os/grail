"""
Validators and formatting models for Elixir
"""
import ast
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, Field, field_validator


class ElixirException(Exception):
    """
    Custom error for Elixir
    """

    def __init__(self, message: str, exception: Exception | None = None, code: str | None = None):
        # Format the message
        # Optionally include the code and exception
        if exception:
            message += f"\n\t{type(exception)}: {exception}"
        super().__init__(message)
        self.exception = exception
        self.code = code
        # self.add_note(self.code)


INSECURE_METHODS = {
    "exec", "eval", "compile", "globals", "locals",
    "open", "input", "file", "os", "sys", "subprocess",
    "shutil", "ctypes", "getattr", "setattr", "delattr",
    "vars", "socket", "requests", "urllib",
    "pathlib"
}

GREEN_IMPORTS = {
    # Imports that we trust
    "grail", "elixir",
    # Built-ins
    "functools", "itertools", "math", "copy", "datetime", "random", "json",
    # Science
    "numpy", "pandas", "scipy", "sklearn", "pymc", "xarray", "jax",
}


class ElixirInputVar(BaseModel):
    """
    Defines the formatting of an elixir function input argument.
    Helps the LLM understand what data is incoming.
    """
    arg: str
    description: str
    datatype: Any


class ElixirOutputVar(BaseModel):
    """
    Defines the formatting of an elixir function output argument.
    Helps the LLM understand what data to return.
    """
    name: str
    description: str
    datatype: Any


def load_code(code: str, func_name) -> Callable:
    """
    Load code into AST and add to the namespace. Return the function.

    # TODO -- Exfiltrate this functionality to an upgraded version of code save/load
    """
    tree = ast.parse(code)
    # Check for green imports and add them to the namespace
    # Also provide some standards
    namespace = {}
    # The following is to try and import green libs on the fly,
    # but it seems to be a bit buggy and the internal function fails to use them
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[0] in GREEN_IMPORTS:
                    # Add the import to the namespace
                    namespace[alias.name.split(".")[0]] = __import__(alias.name)
                else:
                    raise ElixirException(f"Import '{alias.name}' is not allowed in elixir")

    # Create local namespace for this function
    # exec(code, {}, namespace)
    exec(code, namespace)
    # Load the imports in from the namespace (those that are func_name)
    return namespace[func_name]


class ElixirValidator(BaseModel):
    """
    Base validator for LLM-generated code
    """
    code: str = Field(
        ..., title="Code",
        description="Python code for execution."
    )
    # Self-referencing other methods in multimethod scenarios
    # Disable this for now
    # sibling: 'Optional[ElixirValidator]' = None

    # ---
    # Private attributes that avoid regular pydantic validation stream
    is_function: ClassVar[bool | None] = None  # Must be set to specified to declare if returned code is a function or
    # not
    function_name: ClassVar[str | None] = None
    required_args: ClassVar[tuple[ElixirInputVar] | None] = None
    returned_data: ClassVar[tuple[ElixirOutputVar] | None] = None
    # Optional description to pass into LLM for elixir. We default to the validator docstring if this is null.
    # The args and returned data objects will automatically be included in the prompt.
    prompt_description: ClassVar[str | None] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.is_function is None:
            raise ValueError(f"{self} must set 'is_function' to True or False.")

    @field_validator("code", mode="after")
    @classmethod
    def base_code_validator(cls, code: str):
        """
        A base code validator that covers some desirable defaults
        """
        tree = ast.parse(code)

        # Check imports if any
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in GREEN_IMPORTS:
                        msg = f"Import '{alias.name}' is not allowed."
                        raise ElixirException(msg, ValueError(msg), code)
            elif isinstance(node, ast.ImportFrom):
                if not node.module or node.module.split(".")[0] not in GREEN_IMPORTS:
                    # If module is None (relative import) or not in green list
                    module_name = node.module if node.module else "relative import"
                    msg = f"Import from '{module_name}' is not allowed."
                    raise ElixirException(msg, ValueError(msg), code)

        # Prevent unsafe operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in INSECURE_METHODS:
                    msg = f"Usage of '{node.func.id}' is not allowed."
                    raise ElixirException(msg, ValueError(msg), code)

        return code

    @field_validator("code", mode="after")
    @classmethod
    def function_meta_validator(cls, code: str):
        """
        Uses AST to validate:
        - Correct function name
        - Correct function signature
        """
        if not cls.is_function:
            return code

        tree = ast.parse(code)

        # Ensure there's only one function definition
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            raise ElixirException("", ValueError("There must be exactly one function definition."), code)

        func = functions[0]

        # Validate function name
        if func.name != cls.function_name:
            raise ElixirException("", ValueError(f"Function name must be '{cls.function_name}'."), code)

        # Validate required argument names
        actual_args = tuple([arg.arg for arg in func.args.args])

        arg_names = tuple([arg.arg for arg in cls.required_args])
        if actual_args != arg_names:
            raise ElixirException("", ValueError(f"Expected arguments {arg_names}, got {actual_args}."), code)

        # Much less clear how to validate output data, but we can at least check the number of returns
        # TODO -- Evaluating return datatype? Might have to be through descendant classes
        for stmt in func.body:
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    raise ElixirException("", ValueError("Explicit 'return' with no value is not allowed."), code)

                elif isinstance(stmt.value, ast.Tuple):
                    if len(stmt.value.elts) != len(cls.returned_data):
                        raise ElixirException(
                            "", ValueError(
                                f"Expected {len(cls.returned_data)} return values, got {len(stmt.value.elts)}."
                            ), code
                        )

                elif len(cls.returned_data) != 1:
                    raise ElixirException(
                        "", ValueError(
                            f"Expected {len(cls.returned_data)} return values, got 1."
                        ), code
                    )

        return code
