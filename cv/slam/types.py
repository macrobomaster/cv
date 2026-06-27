"""Container dataclasses for the SLAM stack.

Filter state (numpy arrays) lives in `msckf.MsckfState`. Everything here is
plain-Python / numpy data passed between the front-end and the filter.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

@dataclass
class Frame:
  """A camera frame fed into the front-end."""
  t: float                           # capture time (s, monotonic)
  img: np.ndarray                    # (H, W, 3) uint8 RGB
  gray: Optional[np.ndarray] = None  # (H, W) uint8, lazily filled

@dataclass
class Track:
  """A 2D feature tracked across frames by KLT.

  `frame_ids` and `uv` are kept in lockstep: `uv[k]` is the pixel location in
  the frame with `frame_ids[k]`. A track is `alive` until KLT loses it or it
  fails the outlier rejection step; once `alive=False` and `len(uv) >= 2` the
  back-end triangulates it and consumes it as an MSCKF measurement.
  """
  id: int
  frame_ids: list[int] = field(default_factory=list)
  uv: list[np.ndarray] = field(default_factory=list)   # each (2,) float32 px
  alive: bool = True

  def append(self, frame_id:int, uv:np.ndarray) -> None:
    self.frame_ids.append(frame_id)
    self.uv.append(uv.astype(np.float32))

  def __len__(self) -> int: return len(self.uv)
