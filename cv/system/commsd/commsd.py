import time

import serial

from ..logging import logger
from .protocol import Protocol, Command

def run():
  port = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
  protocol = Protocol(port)

  while True:
    time.sleep(1)
    try:
      logger.info("checking game running state")
      state = protocol.msg(Command.CHECK_STATE, 0x0)
      logger.info(f"game running: {state}")
    except TimeoutError:
      logger.error("timeout checking state")

