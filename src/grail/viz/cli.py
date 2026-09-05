"""CLI entry point for launching the GRAIL Streamlit dashboard."""

import os
import sys

from grail.settings import PROJECT_ROOT

APP_PATH = os.path.join(PROJECT_ROOT, "webapp", "dashboard", "app.py")


def main() -> None:
    """Launch the dashboard via `streamlit run`, forwarding any extra CLI args."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", APP_PATH, *sys.argv[1:]]
    sys.exit(stcli.main())
