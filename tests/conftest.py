from pathlib import Path
import sys

# Ensure the project's `src` directory is on sys.path so tests can import modules
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
