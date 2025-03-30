from pathlib import Path

from tinygrad.helpers import getenv

SYSTEM_PATH = Path(getenv("SYSTEM_PATH", "./sys/"))
