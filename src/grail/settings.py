import os
import time
from pathlib import Path


def env_to_bool(value: str | bool | None) -> str| bool | None:
    """Try to parse bool; Return value if unable"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized == "":
        return None

    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}

    if normalized in truthy:
        return True
    if normalized in falsy:
        return False

    # Just return the value...
    return value


# Climb up the paths until we find the pyproject toml
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / 'pyproject.toml').exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:  # We've reached the root of the filesystem
        raise FileNotFoundError("pyproject.toml not found in any parent directories.")
    PROJECT_ROOT = PROJECT_ROOT.parent

TESTS_OUT = os.path.join(PROJECT_ROOT, "tests", "tmp")
DATA_DIR = PROJECT_ROOT / "data"
FRAME_SPECS_DIR = DATA_DIR / "frames"
FRAME_REGISTRY_DB_PATH = DATA_DIR / "grail.sqlite3"
RUNS_DIR = DATA_DIR / "runs"

# Log filename will be based on the current time
_default_name = f"{time.strftime('%Y-%m-%d-%H%M%S')}.log"
# Note the path may be modified in grail.logger.py so this is not a reliably imported variable
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", _default_name)

# Can help redirect operations depending on user/system context
SYSTEM_USER = os.getenv("USER", "unknown")

DEBUGGING = env_to_bool(os.environ.get("GRAIL_DEBUG", False))

# <><><  Versions  ><><>
FRAME_VERSION = 1
FRAME_SPEC_VERSION = 1
BUILDER_VERSION = 1
ENGINE_VERSION = 1
RUNNER_VERSION = 1
