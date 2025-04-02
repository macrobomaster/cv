import os, sys, signal, time, traceback, importlib
from multiprocessing import Process
from typing import Callable

from setproctitle import setproctitle
from tinygrad.helpers import colored

from .logging import logger
from .keyvalue import kv_get, kv_getall, kv_clear
from . import messaging

class SupervisedProcess:
  name: str
  module: str
  should_run: Callable[[dict], bool]

  proc: Process | None = None
  shutting_down: bool = False

  def __init__(self, name:str, module:str, should_run:Callable[[dict], bool]=lambda _: True, watchdog_dt:float=-1):
    self.name = name
    self.module = module
    self.should_run = should_run
    self.watchdog_dt = watchdog_dt

  @staticmethod
  def _start(name:str, module:str):
    try:
      logger.unbind()
      logger.bind(name)

      messaging.reset_context()

      setproctitle(name)

      mod = importlib.import_module(module)

      mod.run()

      logger.info(f"{name} exiting")
    except KeyboardInterrupt:
      pass
    except Exception:
      logger.error(f"exception in {name}")
      raise

  def start(self):
    if self.shutting_down:
      self.stop()

    if self.proc is not None:
      return

    logger.info(f"starting {self.module} as {self.name}")
    self.proc = Process(name="MainProcess", target=self._start, args=(self.name, self.module))
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

  def watchdog(self):
    if self.watchdog_dt <= 0:
      return

    if self.proc is None:
      return

    watchdog_time = kv_get("watchdog", self.name)
    if watchdog_time is None:
      return

    if time.monotonic() - watchdog_time > self.watchdog_dt:
      logger.error(f"{self.name} watchdog timeout")
      self.restart()

class Supervisor:
  sprocs: dict[str, SupervisedProcess]

  def __init__(self, procs:list[SupervisedProcess]):
    self.sprocs = {p.name: p for p in procs}

    kv_clear("global_rt")
    kv_clear("watchdog")

  def run(self):
    logger.bind("supervisor")
    setproctitle("supervisor")

    try:
      while True:
        kv = kv_getall("global_rt")
        self.ensure_running(kv)

        running = " ".join(colored(p.name, "green" if p.proc.is_alive() else "red") for p in self.sprocs.values() if p.proc is not None)
        logger.debug(f"{running}")

        if kv.get("do_shutdown", False):
          break

        time.sleep(1)
    except Exception:
      logger.error("supervisor exception")
      logger.error(traceback.format_exc())
    finally:
      for p in self.sprocs.values():
        p.stop(block=False)

      for p in self.sprocs.values():
        p.stop()

      logger.info("supervisor exiting")

  def ensure_running(self, kv:dict):
    for p in self.sprocs.values():
      if p.should_run(kv):
        p.start()
      else:
        p.stop(block=False)

      p.watchdog()
