"""SLAM daemon — MSCKF VIO with AprilTag absolute-pose correction.

Front-end (KLT + AprilTag detection) lives in cv/system/frontd. slamd
consumes its output and runs the filter:
  - IMU samples propagate the state (predict)
  - terminated KLT tracks → triangulate → MSCKF feature update (local VIO)
  - AprilTag detections → PnP + known field map → absolute 6DoF pose update
    (drift-free global correction; replaces loop closure)

Frame convention: the MSCKF body frame is the gimbal IMU frame (full gimbal,
yaw+pitch), so `predict` consumes the gimbal IMU directly with no transform.
The SLAM camera is on the yaw-only stage, so the IMU<-camera extrinsic varies
with gimbal pitch (calib.cam_from_imu(pitch)). MSCKF clones store the CAMERA
pose (composed at augment from the IMU pose + that extrinsic), so feature
updates are native; AprilTag camera-pose fixes are converted back to the IMU
body frame before the absolute update.

Subs:
  raw_imu:        {t, accel:[3], gyro:[3]}    (gimbal IMU; RAW_ACCEL via commsd)
  gimbal_state:   {pitch_gi, yaw_gi, ...}     (gimbal angles, for the extrinsic)
  feature_tracks: {ct, frame_id, terminated, live_uvs, live_ids, live_ages}
  apriltags:      {ct, frame_id, detections:[{id, corners:[[u,v]*4]}]}

Pubs:
  slam_pose:  {t, p_w, v_w, q_wb(from gimbal), cov_pos(3x3), n_tracks, n_clones, n_tags}
  slam_debug: {t, clone_p, recent_points, seen_tag_p}
"""
import gc, time
from collections import deque

import cv2
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ...slam import calib
from ...slam.msckf import MsckfState
from ...slam.triangulate import triangulate_feature

# Keep cv2 (solvePnP/Rodrigues) from spawning an all-core thread pool — with
# camerad + frontd also running, oversubscription starves camerad and trips its
# watchdog. One thread keeps the daemons cooperative.
OPENCV_THREADS = getenv("OPENCV_THREADS", 1)

# Tag object points in the marker's own frame (cv2.aruco corner order:
# TL, TR, BR, BL; marker frame +x right, +y up, +z out of the tag).
_S = calib.TAG_SIZE / 2.0
_TAG_OBJ_PTS = np.array([[-_S,  _S, 0], [_S,  _S, 0],
                         [ _S, -_S, 0], [-_S, -_S, 0]], dtype=np.float32)
_DIST_ZERO = np.zeros((1, 5), dtype=np.float32)  # frames are pre-undistorted

def _R_to_quat_np(M:np.ndarray) -> np.ndarray:
  from scipy.spatial.transform import Rotation as R
  q = R.from_matrix(M).as_quat()
  return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

def _gimbal_R_wb(yaw:float, pitch:float) -> np.ndarray:
  """world<-IMU(body) orientation from the gimbal's absolute angles (roll≈0).
  Gravity-referenced, so the world frame is z-up automatically.
  TODO: confirm yaw/pitch axis order + signs against the gimbal firmware."""
  cy, sy = np.cos(yaw), np.sin(yaw)
  cp, sp = np.cos(pitch), np.sin(pitch)
  Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], np.float32)   # yaw about world +z
  Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], np.float32)   # pitch about +y
  return (Rz @ Ry).astype(np.float32)

def _tag_body_position(tag_id:int, corners:np.ndarray, p_offset:np.ndarray):
  """PnP a single tag + field map → camera position in world, then the IMU
  body position p_wb = p_wc - p_offset (p_offset = R_wb·t_ic, known from the
  gimbal). Orientation is NOT taken from the tag — the gimbal owns it.
  Returns p_wb (3,) or None."""
  entry = calib.TAG_FIELD_MAP.get(tag_id)
  if entry is None: return None
  R_wt, t_wt = entry
  ok, rvec, tvec = cv2.solvePnP(_TAG_OBJ_PTS, corners.astype(np.float32),
                                calib.K, _DIST_ZERO, flags=cv2.SOLVEPNP_IPPE_SQUARE)
  if not ok: return None
  t_ct = tvec.reshape(3)
  if float(np.linalg.norm(t_ct)) > calib.TAG_MAX_RANGE: return None
  R_ct = cv2.Rodrigues(rvec)[0]                        # camera <- tag
  R_wc = R_wt @ R_ct.T                                  # world <- camera
  p_wc = t_wt - R_wc @ t_ct                            # camera origin in world
  return (p_wc - p_offset).astype(np.float32)

def _triangulate_msg_track(tr_msg:dict, R_cl:list, p_cl:np.ndarray,
                           fid_to_slot:dict[int, int]):
  obs = []
  for fid, uv in zip(tr_msg["frame_ids"], tr_msg["uvs"]):
    slot = fid_to_slot.get(int(fid))
    if slot is not None: obs.append((slot, np.asarray(uv, dtype=np.float32)))
  if len(obs) < 2: return None, [], False
  Rs = [R_cl[s] for s, _ in obs]                       # known world<-camera per clone
  ts = [p_cl[s] for s, _ in obs]
  uvs = np.array([uv for _, uv in obs], dtype=np.float32)
  pw, ok = triangulate_feature(uvs, Rs, ts)
  return pw, obs, ok

# ---------------------------------------------------------------------------
def run():
  gc.disable()
  cv2.setNumThreads(OPENCV_THREADS)
  Tensor.training = False
  pub = messaging.Pub(["slam_pose", "slam_debug"])
  # Latest-only topics drive the per-frame iteration.
  sub = messaging.Sub(["gimbal_state", "feature_tracks", "apriltags"],
                      poll="feature_tracks")
  # raw_imu is 200 Hz but we iterate at feature-track rate (~15 Hz), so it must
  # NOT conflate — queue every sample and drain them all each frame so the
  # filter integrates the full IMU stream instead of one stale sample.
  imu_sub = messaging.Sub(["raw_imu"], conflate=False)

  st = MsckfState.init()
  last_imu_t:float|None = None
  recent_points:deque = deque(maxlen=200)
  n_tags_total = 0
  gimbal_yaw = 0.0
  gimbal_pitch = 0.0          # latest gimbal angles for the body/camera orientation
  last_wd = 0.0
  # flow diagnostics (logged ~0.5 Hz)
  diag_t = time.monotonic()
  d_imu = d_feat_used = d_tags_seen = d_tags_matched = d_frames = 0

  kv_put("watchdog", "slamd", time.monotonic())

  while True:
    # Block on feature_tracks (don't busy-spin) — a 0-timeout poll spins at
    # ~10k Hz, flooding the sqlite KV with watchdog writes and starving camerad
    # (its watchdog then times out). Wake on a frame, or every 200 ms.
    sub.update(timeout=200)
    now = time.monotonic()
    if now - last_wd > 1.0:
      kv_put("watchdog", "slamd", now)
      last_wd = now

    gs = sub["gimbal_state"]
    if gs is not None and sub.updated["gimbal_state"]:
      gimbal_yaw = float(gs["yaw_gi"]); gimbal_pitch = float(gs["pitch_gi"])

    ft = sub["feature_tracks"]
    if ft is None or not sub.updated["feature_tracks"]: continue
    ct = float(ft["ct"])
    frame_id = int(ft["frame_id"])
    d_frames += 1

    # Orientation is known from the gimbal (body = IMU, camera = yaw-stage).
    R_wb = _gimbal_R_wb(gimbal_yaw, gimbal_pitch)        # world<-IMU
    R_ic, t_ic = calib.cam_from_imu(gimbal_pitch)        # IMU<-camera
    R_wc = (R_wb @ R_ic).astype(np.float32)              # world<-camera
    p_offset = (R_wb @ t_ic).astype(np.float32)          # camera-pos offset from IMU

    # Drain every queued IMU sample (non-conflate), oldest first.
    imu_samples = imu_sub.drain("raw_imu")
    d_imu += len(imu_samples)

    # --- Predict through all IMU samples since the last frame --------------
    for s in imu_samples:
      t_imu = float(s["t"])
      dt = t_imu - last_imu_t if last_imu_t is not None else 0.005
      last_imu_t = t_imu
      if dt <= 0 or dt > 0.5: continue
      st.predict(np.array(s["accel"], dtype=np.float32), dt, R_wb)

    # --- Augment clone (camera position; orientation R_wc is known) --------
    st.augment(t=ct, frame_id=frame_id, R_wc=R_wc, p_offset=p_offset)

    # --- Local VIO: feature update from terminated tracks -----------------
    fid_to_slot = {fid: i for i, fid in enumerate(st.fid_cl) if fid >= 0}
    p_cl_np = st.p_cl.numpy()
    points, obs = [], []
    for tr_msg in ft["terminated"]:
      pw, obs_list, ok = _triangulate_msg_track(tr_msg, st.R_cl, p_cl_np, fid_to_slot)
      if not ok or len(obs_list) < 2: continue
      points.append(pw); obs.append(obs_list)
    if points:
      st.update_with_features(points, obs)
      d_feat_used += len(points)
      for pw in points: recent_points.append(pw.tolist())

    # --- Absolute correction: AprilTag position fixes ---------------------
    seen_tag_p = []
    tags = sub["apriltags"]
    if tags is not None and sub.updated["apriltags"]:
      d_tags_seen += len(tags["detections"])
      for det in tags["detections"]:
        p_wb = _tag_body_position(int(det["id"]), np.array(det["corners"], dtype=np.float32),
                                  p_offset)
        if p_wb is None: continue
        st.update_with_position(p_wb)
        n_tags_total += 1
        d_tags_matched += 1
        seen_tag_p.append(p_wb.tolist())

    # --- Flow diagnostics --------------------------------------------------
    if now - diag_t > 2.0:
      dt = now - diag_t
      logger.info(
        f"slamd flow: frames {d_frames/dt:.0f}/s  imu {d_imu/dt:.0f}/s  "
        f"feats_used {d_feat_used/dt:.0f}/s  tags_seen {d_tags_seen/dt:.0f}/s  "
        f"tags_matched {d_tags_matched/dt:.0f}/s  p_w={st.p_w.numpy().round(3).tolist()}")
      diag_t = now
      d_imu = d_feat_used = d_tags_seen = d_tags_matched = d_frames = 0

    # --- Publish ----------------------------------------------------------
    # Orientation is the gimbal's (known); the filter estimates position.
    cov_pos = st.P.numpy()[0:3, 0:3]                     # 3x3 position covariance

    pub.send("slam_pose", {
      "t": ct,
      "p_w": st.p_w.numpy().tolist(),
      "v_w": st.v_w.numpy().tolist(),
      "q_wb": _R_to_quat_np(R_wb).tolist(),
      "cov_pos": cov_pos.flatten().tolist(),
      "n_tracks": len(ft["live_ids"]),
      "n_clones": sum(1 for fid in st.fid_cl if fid >= 0),
      "n_tags": n_tags_total,
    })

    p_cl_full = st.p_cl.numpy()
    active_idx = [i for i, fid in enumerate(st.fid_cl) if fid >= 0]
    pub.send("slam_debug", {
      "t": ct,
      "clone_p": [p_cl_full[i].tolist() for i in active_idx],
      "recent_points": list(recent_points),
      "seen_tag_p": seen_tag_p,
    })
