import os, sys, traceback, logging, signal, fcntl, errno

from tinygrad.helpers import getenv

from .logging import logger
from .supervisor import Supervisor, SupervisedProcess

PROCS = [
  SupervisedProcess("commsd", "cv.system.commsd.commsd"),
  SupervisedProcess("autoaimd", "cv.system.autoaimd.autoaimd"),
]

# from https://github.com/commaai/openpilot/blob/e674bc1355a4c85c807b3494b26090f7b7d4c99e/system/manager/helpers.py#L15
def unblock_stdout() -> None:
  # get a non-blocking stdout
  child_pid, child_pty = os.forkpty()
  if child_pid != 0:  # parent

    # child is in its own process group, manually pass kill signals
    signal.signal(signal.SIGINT, lambda signum, frame: os.kill(child_pid, signal.SIGINT))
    signal.signal(signal.SIGTERM, lambda signum, frame: os.kill(child_pid, signal.SIGTERM))

    fcntl.fcntl(sys.stdout, fcntl.F_SETFL, fcntl.fcntl(sys.stdout, fcntl.F_GETFL) | os.O_NONBLOCK)

    while True:
      try:
        dat = os.read(child_pty, 4096)
      except OSError as e:
        if e.errno == errno.EIO:
          break
        continue

      if not dat:
        break

      try:
        sys.stdout.write(dat.decode('utf8'))
      except (OSError, UnicodeDecodeError):
        pass

    # os.wait() returns a tuple with the pid and a 16 bit value
    # whose low byte is the signal number and whose high byte is the exit status
    exit_status = os.wait()[1] >> 8
    os._exit(exit_status)

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG if getenv("DEBUG") else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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
