import pytest

from grail.elixir import ElixirValidator, ElixirException

good_import = '''
import numpy as np
def square(x: int) -> np.int64:
    return np.square(x)
'''

bad_import = '''
import os
def hateyou():
    print("hate you")
'''

bad_method = '''
def eval_add():
    return eval("3+2")
'''

# Base code validator
def test_base_code_val_imports():
    # Good import
    _ = ElixirValidator.base_code_validator(good_import)

    # Bad import
    with pytest.raises(ElixirException, match=r".*Import 'os' is not allowed.*") as einfo:
        _ = ElixirValidator.base_code_validator(bad_import)


def test_base_code_val_methods():
    with pytest.raises(ElixirException, match=r".*Usage of 'eval' is not allowed.*") as einfo:
        _ = ElixirValidator.base_code_validator(bad_method)

    return
