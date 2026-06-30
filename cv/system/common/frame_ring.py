"""Shared-memory frame ring for the camera hot path.

camerad writes each undistorted frame straight into a /dev/shm slot and publishes
only the slot index on `camera_feed`, so the local consumers (autoaimd, tagd) read
the frame with ZERO serialization copies — no tobytes / cbor / TCP for the 864KB
frame (cbor would serialize it every frame even with no subscriber). autoaimd wraps
the slot as a CPU Tensor via from_blob (zero-copy); the in-JIT .to(device) is the
only copy left. See project_autoaim_frame_input_copies.

Lock-free single-writer / multi-reader. Synchronization is the ring depth: the writer
round-robins N_SLOTS, so a slot isn't reused for N frames (~N*5ms) — far longer than
any reader holds one (autoaimd's H2D ~0.1ms, tagd's cvtColor ~2ms). No locks.

shm is host-local: REMOTE tools can't attach (view/tensor return None). The bridge
daemon `framed` re-publishes full frames on `camera_feed_full` for them. frame_view /
frame_tensor also fall back to an inline `frame` bytes field, so a replayed bag (or
the bridge topic) still works.
"""
import ctypes
from multiprocessing import shared_memory

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import mv_address

CAMERA_RING = "camera_feed"   # ring id; shared between writer and readers
N_SLOTS = 8

class FrameRing:
  def __init__(self, name:str, shape:tuple[int, ...], n_slots:int=N_SLOTS, create:bool=False):
    self.name, self.shape, self.n_slots = name, tuple(shape), n_slots
    self.nbytes = int(np.prod(shape))   # uint8 frame
    self._shm:  list = [None] * n_slots
    self._view: list = [None] * n_slots
    self._tens: list = [None] * n_slots
    self._w = 0
    if create:
      for i in range(n_slots): self._open(i, create=True)
      # NOTE: no atexit unlink — leaving the slots lets a camerad restart reattach the
      # same inode so readers stay valid (a fresh inode would feed them stale frames).

  def _slot_name(self, i:int) -> str: return f"camerad_{self.name}_{i}"

  def _open(self, i:int, create:bool) -> bool:
    if self._shm[i] is not None: return True
    try:
      shm = shared_memory.SharedMemory(name=self._slot_name(i), create=False)
      if shm.size < self.nbytes:        # stale wrong-size leftover (e.g. IMG dims changed)
        shm.close()
        if create: shm.unlink()
        raise FileNotFoundError
    except FileNotFoundError:
      if not create: return False       # reader, writer not up yet / remote host
      shm = shared_memory.SharedMemory(name=self._slot_name(i), create=True, size=self.nbytes)
    self._shm[i] = shm
    self._view[i] = np.ndarray(self.shape, dtype=np.uint8, buffer=shm.buf)
    return True

  def next_view(self) -> tuple[int, np.ndarray]:
    """writer: (slot, numpy view) to write the next frame into; advances round-robin."""
    i, self._w = self._w, (self._w + 1) % self.n_slots
    return i, self._view[i]

  def view(self, i:int):
    """reader: numpy uint8 view over slot i (for cv2/np), or None if not locally attachable."""
    if self._view[i] is None and not self._open(i, create=False): return None
    return self._view[i]

  def tensor(self, i:int):
    """reader: persistent zero-copy CPU Tensor over slot i (autoaimd's JIT input), or None."""
    if self._tens[i] is None:
      if not self._open(i, create=False): return None
      self._tens[i] = Tensor.from_blob(mv_address(self._shm[i].buf), (self.nbytes,), dtype=dtypes.uint8, device="CPU")
    return self._tens[i]

  def close(self):
    for shm in self._shm:
      if shm is not None: shm.close()
    self._shm = [None] * self.n_slots

def frame_view(ring:FrameRing, msg):
  """RGB numpy view for a camera_feed msg: shm slot (local), else inline bytes (bridge/replay), else None."""
  if msg is None: return None
  if (slot := msg.get("slot")) is not None and (v := ring.view(slot)) is not None: return v
  if (b := msg.get("frame")) is not None: return np.frombuffer(b, np.uint8).reshape(ring.shape)
  return None

def frame_tensor(ring:FrameRing, msg):
  """zero-copy CPU Tensor over the frame for autoaimd's JIT: shm slot (local), else inline bytes, else None."""
  if msg is None: return None
  if (slot := msg.get("slot")) is not None and (t := ring.tensor(slot)) is not None: return t
  if (b := msg.get("frame")) is not None:
    return Tensor.from_blob(ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value, (ring.nbytes,), dtype=dtypes.uint8, device="CPU")
  return None
