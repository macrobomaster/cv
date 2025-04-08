import random, os, pickle, time
from multiprocessing import shared_memory
from typing import Callable
from dataclasses import dataclass

from tinygrad.helpers import prod, getenv, Context, trange
from tinygrad.tensor import Tensor
from tinygrad.dtype import DType

from ..system.core import messaging
from ..system.core.logging import logger
from ..system.core.keyvalue import kv_clear, kv_get, kv_put

DATAPROC = getenv("DATAPROC", 1)

@dataclass
class BatchDesc:
  shape: tuple
  dtype: DType

def shuffled_indices(n:int):
  indices = {}
  for i in range(n-1, -1, -1):
    j = random.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

class Dataloader:
  def __init__(self, descs:dict[str, BatchDesc], bs:int, files_fn:Callable):
    self.descs = descs
    self.bs = bs
    self.files_fn = files_fn

    self.loading = False

    kv_clear("dataloader")

    self.push_pull = messaging.PushPull("dataloader")
    sync = messaging.Pub(["dataloader_sync"])

    # wait for dataloader processes to start
    started = 0
    while started < DATAPROC:
      self.push_pull.push(True)
      time.sleep(0.1)
      if self.push_pull.pull(False) == True:
        started += 1
        logger.info(f"dataloader {started} connected")
      time.sleep(0.1)
    logger.info("all dataloaders connected")

    szs = {name: (bs, *desc.shape) for name, desc in descs.items()}

    inflight_shms = []
    for i in range(8):
      shms = {}
      for name, desc in descs.items():
        if os.path.exists(f"/dev/shm/dataloader_{name}_{i}"): os.unlink(f"/dev/shm/dataloader_{name}_{i}")
        shms[name] = shared_memory.SharedMemory(name=f"dataloader_{name}_{i}", create=True, size=prod(szs[name]) * desc.dtype.itemsize)
      inflight_shms.append(shms)

    self.inflight = []
    for i in range(8):
      tensors = {}
      for name, desc in descs.items():
        tensors[name] = Tensor.empty(*szs[name], dtype=desc.dtype, device=f"disk:/dev/shm/dataloader_{name}_{i}")
      self.inflight.append(tensors)

    kv_put("dataloader", "inflight", pickle.dumps(self.inflight))

    # start
    for _ in range(5):
      time.sleep(0.1)
      sync.send("dataloader_sync", True)

  def _epoch(self):
    files = self.files_fn()

    gen = shuffled_indices(len(files))
    def enqueue_batch(num):
      for idx in range(self.bs):
        file = files[next(gen)]
        self.push_pull.push((num, idx, file))

    class Cookie:
      def __init__(self, num): self.num = num
      def __del__(self):
        try: enqueue_batch(self.num)
        except StopIteration: pass

    gottten = [0]*32
    def receive_batch():
      while True:
        num = self.push_pull.pull()
        gottten[num] += 1
        if gottten[num] == self.bs: break
      gottten[num] = 0
      return self.inflight[num], Cookie(num)

    for bn in range(8):
      enqueue_batch(bn)

    for _ in trange(len(files)//self.bs):
      yield receive_batch()

    self.loading = False

  def load(self):
    if self.loading: return
    self.iter = iter(self._epoch())
    self.loading = True

  def next(self, device:str):
    d, c = next(self.iter)
    ret = []
    for _, t in d.items():
      ret.append(t.to(device))
    return *ret, c

class DataloaderProc:
  def __init__(self, load_single_fn:Callable):
    self.load_single_fn = load_single_fn

  def start(self):
    pull_push = messaging.PullPush("dataloader")
    sync = messaging.Sub(["dataloader_sync"])

    # wait for something to be pushed then push something back to sync
    pull_push.pull()
    pull_push.push(True)

    # global sync for all dataloaders
    sync.update(None)

    # grab the shared memory
    inflight = pickle.loads(kv_get("dataloader", "inflight"))

    logger.info("dataloader online")

    with Context(DEBUG=0):
      while (recv := pull_push.pull()):
        try: num, idx, file = recv
        except: continue
        data = self.load_single_fn(file)

        # write to shared memory
        for name, t in inflight[num].items():
          t[idx].contiguous().realize().lazydata.base.realized.ensure_allocated().as_buffer(force_zero_copy=True)[:] = data[name]

        pull_push.push(num)
