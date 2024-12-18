from pathlib import Path

from tinygrad.helpers import getenv

BASE_PATH = Path(getenv("BASE_PATH", "./base/"))
