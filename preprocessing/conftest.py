"""Make the in-tree package importable when running pytest from the repo root."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
