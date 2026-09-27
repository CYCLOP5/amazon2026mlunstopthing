'repository entry point: uv run main.py --help'
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "business_entity_resolution/scripts/run_stack.py"), run_name="__main__")
