import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import get_default_config_path

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
TARGET = THIS_DIR / "evaluate.py"

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", get_default_config_path("eval_safeeraser")])
    runpy.run_path(str(TARGET), run_name="__main__")
