import time
from typing import Any
from collections import deque

import zmq
import cbor2
import xxhash

context = zmq.Context()
def reset_context():
  global context
  context = zmq.Context()

# deterministically generate a port number from the service name
def get_port(service:str):
  return 19000 + xxhash.xxh32(service.encode()).intdigest() % (65535 - 19000)

class Pub:
  def __init__(self, services:list[str]):
    self.socks = {}
    for service in services:
      sock = context.socket(zmq.PUB)
      sock.set(zmq.CONFLATE, 1)
      sock.set(zmq.LINGER, 0)
      sock.bind(f"tcp://*:{get_port(service)}")
      self.socks[service] = sock

  def send(self, service:str, data:Any|None):
    self.socks[service].send(cbor2.dumps(data))

class AliveChecker:
  def __init__(self, count:int=100):
    self.count = count
    self.lt = 0
    self.avg_dt = 0
    self.dts = deque(maxlen=count)
    self.missed_dt = 0

  def update(self, t:float):
    dt = t - self.lt
    self.lt = t

    # add new dt to moving average
    if len(self.dts) == self.count:
      self.avg_dt -= self.dts.popleft() / self.count
    self.avg_dt += dt / self.count
    self.dts.append(dt)

  def alive(self, t:float) -> bool:
    dt = t - self.lt

    # check if dt is too large compared to the average
    if len(self.dts) == self.count:
      if dt > 10 * self.avg_dt:
        return False
    return True

class Sub:
  def __init__(self, services:list[str], poll:str|None=None, addr:str="127.0.0.1"):
    self.services = set(services)
    self.polled_services = set([poll]) if poll else self.services
    self.non_polled_services = self.services - self.polled_services

    self.socks = {}
    for service in services:
      sock = context.socket(zmq.SUB)
      sock.set(zmq.CONFLATE, 1)
      sock.set(zmq.LINGER, 0)
      sock.connect(f"tcp://{addr}:{get_port(service)}")
      sock.subscribe(b"")
      self.socks[service] = sock

    self.poller = zmq.Poller()
    for service in self.polled_services:
      self.poller.register(self.socks[service], zmq.POLLIN)

    self.data = {service: None for service in self.services}
    self.valid = {service: False for service in self.services}
    self.updated = {service: False for service in self.services}
    self.alive_checker = {service: AliveChecker() for service in self.services}
    self.alive = {service: False for service in self.services}

  def __getitem__(self, service:str):
    return self.data[service]

  def _read_update(self, service:str):
    try: data = self.socks[service].recv(flags=zmq.NOBLOCK)
    except zmq.error.Again: return
    data = cbor2.loads(data)
    if data is None:
      self.valid[service] = False
    else:
      self.valid[service] = True
      self.data[service] = data
    self.updated[service] = True

  def update(self, timeout:int=100):
    self.updated = {service: False for service in self.services}

    # check for polled services
    socks = dict(self.poller.poll(timeout))
    for service in self.polled_services:
      if socks.get(self.socks[service]) == zmq.POLLIN:
        self._read_update(service)

    # check for non-polled services
    for service in self.non_polled_services:
      self._read_update(service)

    # check if alive services are still alive
    t = time.monotonic()
    for service in self.services:
      if self.updated[service]:
        self.alive_checker[service].update(t)
      if self.data[service] is not None:
        self.alive[service] = self.alive_checker[service].alive(t)
