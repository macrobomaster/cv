import logging

from tinygrad.helpers import colored, getenv

class Formatter(logging.Formatter):
  COLOR = {
    logging.DEBUG: "white",
    logging.INFO: "green",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "magenta",
  }
  def format(self, record):
    fmt = f"%(asctime)s {colored("[%(levelname)s]", self.COLOR.get(record.levelno))} %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    return formatter.format(record)

class Logger:
  def __init__(self):
    self.logger = logging.getLogger()

    self.logger.setLevel(logging.DEBUG if getenv("DEBUG") else logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(Formatter())

    self.logger.addHandler(handler)

  def bind(self, name:str):
    self.logger = self.logger.getChild(name)
  def unbind(self):
    if self.logger.parent is not None:
      self.logger = self.logger.parent

  def debug(self, *args, **kwargs): self.logger.debug(*args, **kwargs)
  def info(self, *args, **kwargs): self.logger.info(*args, **kwargs)
  def warning(self, *args, **kwargs): self.logger.warning(*args, **kwargs)
  def error(self, *args, **kwargs): self.logger.error(*args, **kwargs)
  def critical(self, *args, **kwargs): self.logger.critical(*args, **kwargs)
  def exception(self, *args, **kwargs): self.logger.exception(*args, **kwargs)
  def log(self, *args, **kwargs): self.logger.log(*args, **kwargs)

logger = Logger()
