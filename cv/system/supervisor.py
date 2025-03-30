import os, sys, signal, time, traceback, importlib
from multiprocessing import Process
from typing import Callable

from setproctitle import setproctitle
from tinygrad.helpers import colored

from .logging import logger

class SupervisedProcess:
  name: str
  module: str
  should_run: Callable[[], bool]

  proc: Process | None = None
  shutting_down: bool = False

  def __init__(self, name:str, module:str, should_run:Callable[[], bool]=lambda: True):
    self.name = name
    self.module = module
    self.should_run = should_run

  @staticmethod
  def _start(name:str, module:str):
    try:
      mod = importlib.import_module(module)

      setproctitle(name)

      logger.unbind()
      logger.bind(name)

      mod.run()
    except KeyboardInterrupt:
      pass
    except Exception:
      logger.error(f"exception in {name}")
      logger.error(traceback.format_exc())
      raise
    finally:
      logger.info(f"{name} exiting")

  def start(self):
    if self.shutting_down:
      self.stop()

    if self.proc is not None:
      return

    logger.info(f"starting {self.module} as {self.name}")
    self.proc = Process(name=self.name, target=self._start, args=(self.name, self.module))
    self.proc.start()
    self.shutting_down = False

  def stop(self, force:bool=False, block:bool=True):
    if self.proc is None:
      return

    if self.proc.is_alive():
      if not self.shutting_down:
        logger.info(f"stopping {self.name}")
        self.signal(signal.SIGKILL if force else signal.SIGINT)
        self.shutting_down = True

      if not block:
        return

      t = time.monotonic()
      while self.proc.is_alive() and time.monotonic() - t < 5:
        time.sleep(0.001)

      if self.proc.is_alive():
        logger.error(f"failed to stop {self.name}, killing")
        self.signal(signal.SIGKILL)
        self.proc.join()

    if not self.proc.is_alive():
      self.shutting_down = False
      self.proc = None

  def restart(self):
    self.stop(force=True)
    self.start()

  def signal(self, sig:int):
    if self.proc is None:
      return

    if not self.proc.is_alive():
      return

    if self.proc.pid is None:
      return

    os.kill(self.proc.pid, sig)

class Supervisor:
  sprocs: dict[str, SupervisedProcess]

  def __init__(self, procs:list[SupervisedProcess]):
    self.sprocs = {p.name: p for p in procs}

  def run(self):
    logger.bind("supervisor")
    setproctitle("supervisor")

    try:
      self.ensure_running()

      while True:
        time.sleep(1)

        self.ensure_running()

        running = " ".join(colored(p.name, "green" if p.proc.is_alive() else "red") for p in self.sprocs.values() if p.proc is not None)
        logger.debug(f"{running}")
    except Exception:
      logger.error("supervisor exception")
      logger.error(traceback.format_exc())
    finally:
      for p in self.sprocs.values():
        p.stop(block=False)

      for p in self.sprocs.values():
        p.stop()

  def ensure_running(self):
    for p in self.sprocs.values():
      if p.should_run():
        p.start()
      else:
        p.stop(block=False)
