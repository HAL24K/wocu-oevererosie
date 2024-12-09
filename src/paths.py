"""Various relevant paths"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
SCRIPT_DIR = ROOT_DIR / "scripts"
TEST_DIR = ROOT_DIR / "tests"
