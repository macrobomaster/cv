from typing import Any

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

  def send(self, service:str, data:Any):
    self.socks[service].send(cbor2.dumps(data))

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
    self.updated = {service: False for service in self.services}

  def __getitem__(self, service:str):
    return self.data[service]

  def _read_update(self, service:str):
    try: data = self.socks[service].recv(flags=zmq.NOBLOCK)
    except zmq.error.Again: return
    self.data[service] = cbor2.loads(data)
    self.updated[service] = True

  def update(self, timeout:int=100):
    self.updated = {service: False for service in self.services}

    socks = dict(self.poller.poll(timeout))
    for service in self.polled_services:
      if socks.get(self.socks[service]) == zmq.POLLIN:
        self._read_update(service)

    for service in self.non_polled_services:
      self._read_update(service)
