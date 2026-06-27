"""Sparse feature front-end: detect, track, reject outliers, hand off to MSCKF.

v1 uses classical Shi-Tomasi corners + KLT optical flow (`cv2.goodFeaturesToTrack`
and `cv2.calcOpticalFlowPyrLK`). The `FeatureFrontend` class is the public
surface; swapping in a neural detector in v2 means replacing `_detect` with a
network call and leaving the rest of the loop untouched.

Per camera frame the front-end:
  1. Converts incoming RGB to gray.
  2. Tracks live features forward by KLT.
  3. Rejects geometric outliers (and most dynamic-scene features) with
     RANSAC essential-matrix between previous and current frame.
  4. Detects new corners to top up the target track count.
  5. Returns the list of tracks that were terminated this frame so the
     MSCKF can triangulate and consume them.
"""
import cv2
import numpy as np

from . import common
from .types import Frame, Track

TARGET_TRACK_COUNT = 200
KLT_WIN = (21, 21)
KLT_PYR = 3
KLT_MAX_ITERS = 30
KLT_EPS = 0.01
MIN_FEATURE_DISTANCE = 8       # px, between detected corners
QUALITY_LEVEL = 0.01           # Shi-Tomasi quality threshold
MIN_TRACK_AGE_FOR_UPDATE = common.MIN_FEATURE_OBS  # don't hand over tracks the filter would reject
MAX_TRACK_LEN = common.N_CLONES # don't keep tracks longer than the clone window
# Force-terminate length is staggered per track over [TRACK_LEN_MIN, MAX_TRACK_LEN]
# so a bulk-seeded cohort doesn't all retire on the same frame — that synchronized
# batch dumps ~150 feature updates into one frame (periodic position snaps +
# pipeline stalls). Spreading it keeps a few features retiring every frame.
TRACK_LEN_MIN = max(MIN_TRACK_AGE_FOR_UPDATE, MAX_TRACK_LEN // 2)

class FeatureFrontend:
  def __init__(self):
    self.next_track_id = 0
    self.live:dict[int, Track] = {}     # track_id -> Track (only currently-alive)
    self.prev_gray:np.ndarray|None = None
    self.prev_frame_id = -1
    self.frame_counter = 0

  # -------------------------------------------------------------------------
  def process(self, frame:Frame) -> tuple[list[Track], int]:
    """Run one frame through the front-end.

    Args:
      frame: incoming camera frame. `frame.gray` is filled in if absent.

    Returns:
      (terminated_tracks, frame_id)
        terminated_tracks: tracks finalized this frame (lost or maxed out).
                          The caller (MSCKF) triangulates and consumes them.
        frame_id: monotonic integer ID this front-end assigned to the frame.
    """
    if frame.gray is None:
      frame.gray = cv2.cvtColor(frame.img, cv2.COLOR_RGB2GRAY)
    frame_id = self.frame_counter
    self.frame_counter += 1

    terminated:list[Track] = []

    if self.prev_gray is None or not self.live:
      # First frame, or all tracks already lost — detect and seed.
      self.prev_gray = frame.gray
      self.prev_frame_id = frame_id
      self._seed_new(frame.gray, frame_id, set())
      return [], frame_id

    # Pack live tracks' last-seen pixel positions in a stable order
    track_ids = list(self.live.keys())
    pts_prev = np.array([self.live[tid].uv[-1] for tid in track_ids], dtype=np.float32).reshape(-1, 1, 2)

    pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
      self.prev_gray, frame.gray, pts_prev, None,
      winSize=KLT_WIN, maxLevel=KLT_PYR,
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, KLT_MAX_ITERS, KLT_EPS),
    )
    status = status.flatten().astype(bool)

    # Drop tracks that fell off the image edge as well
    if pts_next is not None:
      pn = pts_next.reshape(-1, 2)
      in_bounds = ((pn[:, 0] >= 0) & (pn[:, 0] < common.IMG_W) &
                   (pn[:, 1] >= 0) & (pn[:, 1] < common.IMG_H))
      status &= in_bounds

    # RANSAC essential-matrix outlier rejection on the surviving correspondences.
    # Looser than before — strict 1 px cv2.RANSAC was killing nearly every track
    # under realistic motion + rough calibration, which is why trails never
    # accumulated. USAC_MAGSAC with 3 px is permissive enough to keep
    # well-behaved features while still cutting clear dynamic-scene outliers.
    if status.sum() >= 8:
      prev_ok = pts_prev[status].reshape(-1, 2).astype(np.float64)
      next_ok = pn[status].astype(np.float64)
      _, mask = cv2.findEssentialMat(prev_ok, next_ok, common.K,
                                     method=cv2.USAC_MAGSAC, prob=0.999, threshold=3.0)
      if mask is not None:
        mask = mask.flatten().astype(bool)
        ok_idx = np.where(status)[0]
        keep_mask = np.zeros(len(status), dtype=bool)
        keep_mask[ok_idx[mask]] = True
        for k, tid in enumerate(track_ids):
          if status[k] and not keep_mask[k]:
            t = self.live.pop(tid)
            t.alive = False
            terminated.append(t)
        status = keep_mask

    # Append the current observation to each surviving track; terminate the rest
    for k, tid in enumerate(track_ids):
      if not status[k]:
        if tid in self.live:
          t = self.live.pop(tid)
          t.alive = False
          if len(t) >= MIN_TRACK_AGE_FOR_UPDATE:
            terminated.append(t)
        continue
      t = self.live[tid]
      t.append(frame_id, pn[k])
      # Cap track length (staggered per track) so triangulation has a finite
      # window AND cohorts don't all terminate on the same frame.
      if len(t) >= t.max_len:
        self.live.pop(tid)
        t.alive = False
        terminated.append(t)

    # Top up new features in the regions not already covered
    occupied = {tid for tid in self.live}
    self._seed_new(frame.gray, frame_id, occupied)
    self.prev_gray = frame.gray
    self.prev_frame_id = frame_id
    return terminated, frame_id

  # -------------------------------------------------------------------------
  def _seed_new(self, gray:np.ndarray, frame_id:int, _occupied:set[int]) -> None:
    """Detect new corners up to the target count, avoiding existing tracks."""
    need = TARGET_TRACK_COUNT - len(self.live)
    if need <= 0: return
    # Mask out existing tracks so we don't re-detect them
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    for t in self.live.values():
      u, v = t.uv[-1]
      iu, iv = int(round(u)), int(round(v))
      cv2.circle(mask, (iu, iv), MIN_FEATURE_DISTANCE, 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=need, qualityLevel=QUALITY_LEVEL,
                                  minDistance=MIN_FEATURE_DISTANCE, mask=mask)
    if pts is None: return
    span = MAX_TRACK_LEN - TRACK_LEN_MIN + 1
    for p in pts.reshape(-1, 2):
      t = Track(id=self.next_track_id)
      t.max_len = TRACK_LEN_MIN + (self.next_track_id % span)   # staggered by id
      t.append(frame_id, p)
      self.live[self.next_track_id] = t
      self.next_track_id += 1
