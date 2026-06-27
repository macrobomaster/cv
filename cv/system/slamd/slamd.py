"""SLAM daemon — numpy MSCKF VIO with AprilTag absolute correction.

Single process: front-end (KLT + AprilTag detection) and the filter
(`cv/slam/msckf.py`, pure numpy) run in one loop, so feature observations and
filter clones share the same frame ids natively (no cross-process plumbing).

Per camera frame:
  - drain accel samples → predict position/velocity (orientation is input)
  - KLT track → terminated tracks → triangulate → MSCKF feature update (VIO)
  - augment a clone (camera position; orientation is the known gimbal pose)
  - AprilTag detect (throttled) → PnP + known field map → absolute POSITION
    fix + YAW-bias correction (drift-free global anchor; replaces loop closure)

Orientation is a known input from the gimbal (body = IMU = full gimbal,
yaw+pitch; pitch gravity-referenced, yaw a drift-bias δψ anchored by tags).
The SLAM camera is on the yaw-only stage, so the IMU<-camera extrinsic varies
with gimbal pitch (common.cam_from_imu).

Subs:
  camera_feed:  {ct, st, frame}              (512x256 RGB, canonical pinhole)
  raw_imu:      {t, accel:[3]}               (gimbal accelerometer; non-conflate)
  gimbal_state: {yaw_gi, pitch_gi, ...}      (non-conflate; interpolated to ct → orientation)

Pubs:
  slam_pose:  {t, p_w, v_w, q_wb(gimbal), cov_pos(3x3), n_tracks, n_clones, n_tags}
  slam_debug: {t, clone_p, recent_points, seen_tag_p,    # 3D
               live_uvs, live_ids, live_ages, tag_dets}  # 2D overlays for the viewer
"""
import gc, time
from collections import deque

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ..common.geometry import rotz, roty, wrap_pi
from ..common.gimbal import GimbalBuffer
from ...slam import common
from ...slam.msckf import MsckfState
from ...slam.frontend import FeatureFrontend
from ...slam.triangulate import triangulate_feature
from ...slam.types import Frame

# Cap OpenCV's thread pool: each cv2 call (KLT, goodFeatures, findEssentialMat,
# ArUco, solvePnP) otherwise spawns an all-core pool, which oversubscribes the
# cores against camerad and trips its watchdog. One thread keeps them cooperative.
OPENCV_THREADS = getenv("OPENCV_THREADS", 1)
# ArUco detection is the heaviest per-frame op and spikes on cluttered scenes;
# tags are intermittent and only need occasional absolute fixes, so throttle it.
TAG_DETECT_HZ = 10.0

# Tag object points in the marker's own frame (cv2.aruco order: TL,TR,BR,BL).
_S = common.TAG_SIZE / 2.0
_TAG_OBJ_PTS = np.array([[-_S,  _S, 0], [_S,  _S, 0],
                         [ _S, -_S, 0], [-_S, -_S, 0]], dtype=np.float32)
_DIST_ZERO = np.zeros((1, 5), dtype=np.float32)  # frames are pre-undistorted

def _make_tag_detector():
  d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, common.APRILTAG_DICT))
  return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

def _R_to_quat_np(M:np.ndarray) -> np.ndarray:
  from scipy.spatial.transform import Rotation as R
  q = R.from_matrix(M).as_quat()
  return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

def _gimbal_R_wb(yaw:float, pitch:float) -> np.ndarray:
  """world<-IMU(body) orientation from the gimbal's absolute angles (roll≈0).
  Composed as world-yaw ∘ pitch ∘ level-camera-base, so at (0,0) the camera
  looks horizontally forward (not straight up) in the z-up world.
  TODO: confirm yaw/pitch axis order + signs against the gimbal firmware."""
  return (rotz(yaw) @ roty(pitch) @ common.CAM_BASE_R).astype(np.float32)

def _tag_body_pose(tag_id:int, corners:np.ndarray, p_offset:np.ndarray, R_ic:np.ndarray):
  """PnP a single tag + field map → (p_wb, yaw_tag) or None. p_wb is the IMU
  body position in world; yaw_tag the absolute body yaw (for δψ correction).
  Pitch/roll are NOT taken from the tag — the gimbal owns them."""
  entry = common.TAG_FIELD_MAP.get(tag_id)
  if entry is None: return None
  R_wt, t_wt = entry
  ok, rvec, tvec = cv2.solvePnP(_TAG_OBJ_PTS, corners.astype(np.float32),
                                common.K, _DIST_ZERO, flags=cv2.SOLVEPNP_IPPE_SQUARE)
  if not ok: return None
  t_ct = tvec.reshape(3)
  if float(np.linalg.norm(t_ct)) > common.TAG_MAX_RANGE: return None
  R_wc = R_wt @ cv2.Rodrigues(rvec)[0].T               # world <- camera
  p_wc = t_wt - R_wc @ t_ct                            # camera origin in world
  R_wb_tag = R_wc @ R_ic.T                             # world <- IMU body (from tag)
  yaw_tag = float(np.arctan2(R_wb_tag[1, 0], R_wb_tag[0, 0]))  # Rz(yaw)Ry(pitch) → yaw
  return (p_wc - p_offset).astype(np.float32), yaw_tag

def _triangulate_track(tr, R_cl:list, p_cl:np.ndarray, fid_to_slot:dict):
  obs = [(fid_to_slot[int(f)], np.asarray(uv, np.float32))
         for f, uv in zip(tr.frame_ids, tr.uv) if int(f) in fid_to_slot]
  if len(obs) < 2: return None, [], False
  Rs = [R_cl[s] for s, _ in obs]
  ts = [p_cl[s] for s, _ in obs]
  pw, ok = triangulate_feature(np.array([uv for _, uv in obs], np.float32), Rs, ts)
  return pw, obs, ok

# ---------------------------------------------------------------------------
def run():
  gc.disable()
  cv2.setNumThreads(OPENCV_THREADS)
  pub = messaging.Pub(["slam_pose", "slam_debug"])
  sub = messaging.Sub(["camera_feed"], poll="camera_feed")
  # gimbal_state + raw_imu are high-rate; non-conflate + drain so we keep every
  # sample — gimbal for capture-time interpolation, accel for the predict.
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()
  imu_sub = messaging.Sub(["raw_imu"], conflate=False)

  fe = FeatureFrontend()
  tag_detector = _make_tag_detector()
  st = MsckfState.init()

  last_imu_t = None
  last_tag_t = last_wd = 0.0
  recent_points = deque(maxlen=200)
  n_tags_total = 0
  gimbal_yaw = gimbal_pitch = yaw_offset = 0.0
  diag_t = time.monotonic()
  d_imu = d_feat = d_tags = d_frames = 0; d_acc_mag = d_aw_z = 0.0; d_acc_n = 0

  kv_put("watchdog", "slamd", time.monotonic())

  while True:
    sub.update(timeout=200)          # block on camera; wake every 200ms for watchdog
    now = time.monotonic()
    if now - last_wd > 1.0:
      kv_put("watchdog", "slamd", now); last_wd = now

    for m in gimbal_sub.drain("gimbal_state"): gimbal_buf.push(m)

    cam = sub["camera_feed"]
    if cam is None or not sub.updated["camera_feed"]: continue
    ct = float(cam["ct"]); d_frames += 1

    # Gimbal pose interpolated to the capture instant (not the latest sample), so the
    # clone orientation + tag PnP use where the camera actually looked at ct.
    gp = gimbal_buf.interpolate(ct)
    if gp is not None: gimbal_yaw, gimbal_pitch, _ = gp

    # Orientation is the known gimbal pose; yaw_offset is the δψ drift correction.
    R_wb = _gimbal_R_wb(gimbal_yaw + yaw_offset, gimbal_pitch)
    R_ic, t_ic = common.cam_from_imu(gimbal_pitch)
    R_wc = (R_wb @ R_ic).astype(np.float32)
    p_offset = (R_wb @ t_ic).astype(np.float32)

    # --- Predict over all queued accel samples (one batched covariance step) --
    accels, dts = [], []
    for s in imu_sub.drain("raw_imu"):
      t_imu = float(s["t"])
      dt = t_imu - last_imu_t if last_imu_t is not None else 0.005
      last_imu_t = t_imu
      if dt <= 0 or dt > 0.5: continue
      a = np.array(s["accel"], dtype=np.float32)
      accels.append(a); dts.append(dt)
      d_acc_mag += float(np.linalg.norm(a)); d_acc_n += 1
      d_aw_z += float((R_wb @ a + (common.GRAVITY if common.ACCEL_INCLUDES_GRAVITY else 0.0))[2])
    d_imu += len(accels)
    if accels:
      st.predict_batch(np.array(accels, np.float32), np.array(dts, np.float32), R_wb)

    # --- Front-end: KLT track, then augment this frame's clone ---------------
    img = np.frombuffer(cam["frame"], dtype=np.uint8).reshape(common.IMG_H, common.IMG_W, 3)
    frame = Frame(t=ct, img=img)
    terminated, frame_id = fe.process(frame)
    st.augment(t=ct, frame_id=frame_id, R_wc=R_wc, p_offset=p_offset)

    # --- VIO: feature update from terminated tracks --------------------------
    fid_to_slot = {fid: i for i, fid in enumerate(st.fid_cl) if fid >= 0}
    points, obs = [], []
    for tr in terminated:
      pw, obs_list, ok = _triangulate_track(tr, st.R_cl, st.p_cl, fid_to_slot)
      if ok: points.append(pw); obs.append(obs_list)
    if points:
      yaw_offset += st.update_with_features(points, obs)
      d_feat += len(points)
      for pw in points: recent_points.append(pw.tolist())

    # --- AprilTag detection (throttled) → absolute position + yaw fixes ------
    # tag_dets stays None on frames where detection didn't run so the viewer
    # leaves the overlay alone (drawing [] would blink it off at frame rate).
    seen_tag_p, tag_dets = [], None
    if now - last_tag_t >= 1.0 / TAG_DETECT_HZ:
      last_tag_t = now
      tag_dets = []
      corners, ids, _ = tag_detector.detectMarkers(frame.gray)
      if ids is not None:
        for tag_id, c in zip(ids.flatten(), corners):
          c4 = c.reshape(4, 2)
          tag_dets.append({"id": int(tag_id), "corners": c4.astype(float).tolist()})
          res = _tag_body_pose(int(tag_id), c4, p_offset, R_ic)
          if res is None: continue
          p_wb, yaw_tag = res
          yaw_offset += st.update_with_position(p_wb)
          err = wrap_pi(yaw_tag - (gimbal_yaw + yaw_offset))
          yaw_offset += st.update_with_yaw(err)
          n_tags_total += 1; d_tags += 1
          seen_tag_p.append(p_wb.tolist())

    # --- Diagnostics --------------------------------------------------------
    if now - diag_t > 2.0:
      span = now - diag_t
      logger.info(
        f"slamd flow: frames {d_frames/span:.0f}/s  imu {d_imu/span:.0f}/s  "
        f"feats_used {d_feat/span:.0f}/s  tags {d_tags/span:.0f}/s  "
        f"|accel|~{d_acc_mag/max(d_acc_n,1):.2f}  a_w.z~{d_aw_z/max(d_acc_n,1):+.2f}  "
        f"p_w={st.p_w.round(3).tolist()}")
      diag_t = now
      d_imu = d_feat = d_tags = d_frames = 0; d_acc_mag = d_aw_z = 0.0; d_acc_n = 0

    # --- Publish ------------------------------------------------------------
    live_uvs, live_ids, live_ages = [], [], []
    for tid, tr in fe.live.items():
      live_uvs.append([float(tr.uv[-1][0]), float(tr.uv[-1][1])])
      live_ids.append(int(tid)); live_ages.append(int(len(tr)))
    active = [i for i, fid in enumerate(st.fid_cl) if fid >= 0]

    pub.send("slam_pose", {
      "t": ct,
      "p_w": st.p_w.tolist(), "v_w": st.v_w.tolist(),
      "q_wb": _R_to_quat_np(R_wb).tolist(),
      "cov_pos": st.P[0:3, 0:3].flatten().tolist(),
      "n_tracks": len(fe.live), "n_clones": len(active), "n_tags": n_tags_total,
    })
    pub.send("slam_debug", {
      "t": ct,
      "clone_p": [st.p_cl[i].tolist() for i in active],
      "recent_points": list(recent_points),
      "seen_tag_p": seen_tag_p,
      "live_uvs": live_uvs, "live_ids": live_ids, "live_ages": live_ages,
      "tag_dets": tag_dets,
    })
