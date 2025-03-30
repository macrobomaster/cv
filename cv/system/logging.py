import logging

class Logger:
  def __init__(self):
    self.logger = logging.getLogger("system")

  def bind(self, name:str):
    self.logger = self.logger.getChild(name)

  def debug(self, *args, **kwargs): self.logger.debug(*args, **kwargs)
  def info(self, *args, **kwargs): self.logger.info(*args, **kwargs)
  def warning(self, *args, **kwargs): self.logger.warning(*args, **kwargs)
  def error(self, *args, **kwargs): self.logger.error(*args, **kwargs)
  def critical(self, *args, **kwargs): self.logger.critical(*args, **kwargs)
  def exception(self, *args, **kwargs): self.logger.exception(*args, **kwargs)
  def log(self, *args, **kwargs): self.logger.log(*args, **kwargs)

logger = Logger()
