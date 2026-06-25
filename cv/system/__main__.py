import sys, traceback, platform

from tinygrad.helpers import getenv

from .core.logging import logger
from .core.supervisor import Supervisor, SupervisedProcess
from .core.helpers import unblock_stdout

def on_orin(_=None):
  return platform.machine() == "aarch64" and getenv("PC", 0) == 0

ON_ORIN = on_orin()

PROCS = [
  SupervisedProcess("commsd", "cv.system.commsd.commsd", on_orin, watchdog_dt=5),
  SupervisedProcess("camerad", "cv.system.camerad.camerad", watchdog_dt=5,
                    cpu_affinity=2 if ON_ORIN else None, rt_priority=80 if ON_ORIN else None),
  SupervisedProcess("autoaimd", "cv.system.autoaimd.autoaimd",
                    cpu_affinity=3 if ON_ORIN else None, rt_priority=80 if ON_ORIN else None),
  SupervisedProcess("plated", "cv.system.plated.plated"),
  # SupervisedProcess("frontd", "cv.system.frontd.frontd"),
  # SupervisedProcess("slamd", "cv.system.slamd.slamd"),
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
