import sys
from pathlib import Path


def main() -> None:
    """Entry point for the msl command."""
    print("Hello from wqf7006-msl!")


def run_streamlit() -> None:
    """Run the Streamlit app."""
    import subprocess
    
    # Get the project root (parent of src)
    project_root = Path(__file__).parent.parent.parent
    streamlit_app = project_root / "streamlit_app.py"
    
    # Run streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_app)])
