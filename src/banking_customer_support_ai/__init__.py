import sys
from pathlib import Path


def main() -> None:
    """Start the app: Streamlit UI with the agent backend running in the same process."""
    from streamlit.web import cli as stcli

    app = Path(__file__).parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(stcli.main())
