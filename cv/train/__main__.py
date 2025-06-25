import sys, traceback

from tinygrad.helpers import getenv

from ..system.core.logging import logger
from ..system.core.supervisor import Supervisor, SupervisedProcess
from ..system.core.helpers import unblock_stdout
from ..common.dataloader import DATAPROC

MODEL = getenv("MODEL", "autoaim")

PROCS = [
  SupervisedProcess("train", f"cv.{MODEL}.train")
]
PROCS += [SupervisedProcess(f"dl_{i}", f"cv.{MODEL}.data") for i in range(DATAPROC)]

if __name__ == "__main__":
  unblock_stdout()

  try:
    Supervisor(PROCS).run()
  except KeyboardInterrupt:
    logger.warning("caught keyboard interrupt, exiting...")
  except Exception:
    logger.error("supervisor exception while starting")
    logger.error(traceback.format_exc())
    raise

  sys.exit(0)
