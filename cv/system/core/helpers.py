import os, signal, sys, fcntl, errno, time

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

class FrequencyKeeper:
  def __init__(self, freq:int):
    self.freq = freq
    self.dt = 1 / freq
    self.last_time = 0

  def step(self):
    now = time.monotonic()
    dt = now - self.last_time
    if dt < self.dt:
      time.sleep(self.dt - dt)
    self.last_time = now

class Debounce:
  def __init__(self, dt:float=0.1):
    self.dt = dt
    self.stable_value = False
    self.last_time = 0
    self.last_value = False

  def debounce(self, value:bool) -> bool:
    now = time.monotonic()

    if self.last_time == 0:
      self.last_time = now

    if self.last_value != value:
      self.last_value = value
      self.last_time = now
    elif self.stable_value != value and now - self.last_time > self.dt:
      self.stable_value = value

    return self.stable_value
