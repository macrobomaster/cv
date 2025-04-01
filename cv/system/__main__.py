import sys, traceback

from tinygrad.helpers import getenv

from .core.logging import logger
from .core.supervisor import Supervisor, SupervisedProcess
from .core.helpers import unblock_stdout

def not_pc(_):
  return getenv("PC", 0) == 0

PROCS = [
  SupervisedProcess("commsd", "cv.system.commsd.commsd", not_pc),
  SupervisedProcess("camerad", "cv.system.camerad.camerad"),
  SupervisedProcess("autoaimd", "cv.system.autoaimd.autoaimd"),
  SupervisedProcess("plated", "cv.system.plated.plated"),
  SupervisedProcess("decisiond", "cv.system.decisiond.decisiond"),
]

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
