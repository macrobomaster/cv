import random, signal, os
from multiprocessing import Queue, Process, shared_memory
from typing import Callable
from dataclasses import dataclass

from tinygrad.helpers import Context, getenv, prod
from tinygrad.tensor import Tensor
from tinygrad.dtype import DType

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

def loader_process(q_in:Queue, q_out:Queue, load_single_fn:Callable, tensors:dict[str, Tensor]):
  signal.signal(signal.SIGINT, lambda *_: exit(0))
  with Context(DEBUG=0):
    while (recv := q_in.get()):
      idx, file = recv
      data = load_single_fn(file)

      # write to shared memory
      for name, t in tensors.items():
        try:
          t[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = data[name]
        except Exception as e:
          print(f"error writing {name} to shared memory: {e}")
          raise e

      q_out.put(idx)
    q_out.put(None)

def batch_load(descs: dict[str, BatchDesc], load_single_fn, files_fn, bs:int=32, shuffle=True):
  files = files_fn()
  if len(files) == 0: raise ValueError("no files found")
  BATCH_COUNT = min(32, len(files) // bs)

  gen = shuffled_indices(len(files)) if shuffle else iter(range(len(files)))
  def enqueue_batch(num):
    for idx in range(num*bs, (num+1)*bs):
      file = files[next(gen)]
      q_in.put((idx, file))

  running = True
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if running:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while True:
      num = q_out.get()//bs
      gotten[num] += 1
      if gotten[num] == bs: break
    gotten[num] = 0
    return {name: tensors[name][num*bs:(num+1)*bs] for name in descs}, Cookie(num)

  q_in, q_out = Queue(), Queue()

  # get sizes
  szs = {name: (BATCH_COUNT*bs, *desc.shape) for name, desc in descs.items()}

  shms = {}
  for name, desc in descs.items():
    if os.path.exists(f"/dev/shm/dataloader_{name}"): os.unlink(f"/dev/shm/dataloader_{name}")
    shms[name] = shared_memory.SharedMemory(name=f"dataloader_{name}", create=True, size=prod(szs[name]) * desc.dtype.itemsize)

  procs = []
  try:
    tensors = {name: Tensor.empty(*szs[name], dtype=desc.dtype, device=f"disk:/dev/shm/dataloader_{name}") for name, desc in descs.items()}

    for _ in range(getenv("DATAPROC", 1)):
      p = Process(target=loader_process, args=(q_in, q_out, load_single_fn, tensors))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    for _ in range(0, len(files)//bs): yield receive_batch()
  finally:
    running = False
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    for p in procs: p.terminate()
    for p in procs: p.join()
    for shm in shms.values(): shm.close()
    for shm in shms.values():
      try: shm.unlink()
      except FileNotFoundError: pass
