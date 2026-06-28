"""Localization daemon — wheel + IMU dead-reckoning anchored to a KNOWN field.

The field is surveyed (AprilTags at known locations), so this is localization to
a map, not SLAM — no feature tracking, no map building. A plain 10-state EKF
(`cv/slam/filter.py`) fuses:
  - IMU accel        → predict (short-term, high-rate, rides out wheel slip/skid)
  - wheel odometry   → velocity update (observes velocity directly) + planar vz=0
  - AprilTags        → absolute position + yaw fix (drift-free anchor; bounded
                       drift between sightings)

Orientation is a known input from the gimbal (body = IMU = full gimbal, yaw+pitch;
pitch gravity-referenced, yaw a drift-bias δψ anchored by tags). The camera (on
the yaw-only stage) is used ONLY to detect tags here.

Subs:
  camera_feed:  {ct, st, frame}    (RGB, canonical pinhole — for AprilTag detection)
  raw_imu:      {t, accel:[3]}     (gimbal accelerometer; non-conflate)
  gimbal_state: {yaw_gi, ...}      (non-conflate; interpolated to ct → orientation)
  chassis_odom: {vx, vy}           (wheel velocity, m/s, gimbal-heading frame)

Pubs:
  slam_pose:  {t, p_w, v_w, q_wb(gimbal), cov_pos(3x3), n_tags}
  slam_debug: {t, seen_tag_p, tag_dets}
"""
import gc, time

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ..common.geometry import rotz, roty, wrap_pi
from ..common.gimbal import GimbalBuffer
from ...slam import common
from ...slam.filter import PoseEKF

# Cap OpenCV's thread pool: each cv2 call (ArUco, solvePnP) otherwise spawns an
# all-core pool, which oversubscribes the cores against camerad and trips its
# watchdog. One thread keeps them cooperative.
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
  """PnP a single tag + field map → (p_wb, yaw_tag, range) or None. p_wb is the
  IMU body position in world; yaw_tag the absolute body yaw (for δψ correction);
  range = tag distance (slamd scales the measurement noise by it). Pitch/roll are
  NOT taken from the tag — the gimbal owns them."""
  entry = common.TAG_FIELD_MAP.get(tag_id)
  if entry is None: return None
  R_wt, t_wt = entry
  ok, rvec, tvec = cv2.solvePnP(_TAG_OBJ_PTS, corners.astype(np.float32),
                                common.K, _DIST_ZERO, flags=cv2.SOLVEPNP_IPPE_SQUARE)
  if not ok: return None
  t_ct = tvec.reshape(3)
  rng = float(np.linalg.norm(t_ct))
  if rng > common.TAG_MAX_RANGE: return None
  R_wc = R_wt @ cv2.Rodrigues(rvec)[0].T               # world <- camera
  p_wc = t_wt - R_wc @ t_ct                            # camera origin in world
  R_wb_tag = R_wc @ R_ic.T                             # world <- IMU body (from tag)
  # Extract gimbal yaw the SAME way _gimbal_R_wb defines it. R_wb_tag =
  # Rz(yaw)·Ry(pitch)·CAM_BASE_R, so its raw first column is the camera-RIGHT
  # axis — 90° off from world-forward. Undo CAM_BASE_R first, else the yaw fix
  # injects a spurious 90° (the "frustum snaps 90° CW on a tag" bug).
  R_yaw = R_wb_tag @ common.CAM_BASE_R.T               # = Rz(yaw)·Ry(pitch)
  yaw_tag = float(np.arctan2(R_yaw[1, 0], R_yaw[0, 0]))
  return (p_wc - p_offset).astype(np.float32), yaw_tag, rng

# ---------------------------------------------------------------------------
def run():
  gc.disable()
  cv2.setNumThreads(OPENCV_THREADS)
  pub = messaging.Pub(["slam_pose", "slam_debug"])
  sub = messaging.Sub(["camera_feed", "chassis_odom"], poll="camera_feed")
  # gimbal_state + raw_imu are high-rate; non-conflate + drain so we keep every
  # sample — gimbal for capture-time interpolation, accel for the predict.
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()
  imu_sub = messaging.Sub(["raw_imu"], conflate=False)

  tag_detector = _make_tag_detector()
  st = PoseEKF.init()

  last_imu_t = None
  last_tag_t = last_wd = 0.0
  n_tags_total = 0
  gimbal_yaw = gimbal_pitch = yaw_offset = 0.0
  last_vx = last_vy = 0.0      # latest wheel-odom reading (for the flow log)
  diag_t = time.monotonic()
  d_imu = d_tags = d_frames = d_vel = 0; d_acc_mag = d_aw_z = 0.0; d_acc_n = 0

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

    # Gimbal pose interpolated to the capture instant (not the latest sample), so
    # orientation + tag PnP use where the camera actually looked at ct.
    gp = gimbal_buf.interpolate(ct)
    if gp is not None: gimbal_yaw, gimbal_pitch, _ = gp

    # Orientation is the known gimbal pose; yaw_offset is the δψ drift correction.
    R_wb = _gimbal_R_wb(gimbal_yaw + yaw_offset, gimbal_pitch)
    R_ic, t_ic = common.cam_from_imu(gimbal_pitch)
    p_offset = (R_wb @ t_ic).astype(np.float32)          # camera-pos offset from IMU body

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
    # Skip implausibly long integration windows — the startup raw_imu backlog
    # (non-conflate, piles up before the first frame) or a post-stall burst.
    if accels and sum(dts) < 0.5:
      st.predict_batch(np.array(accels, np.float32), np.array(dts, np.float32), R_wb)

    # --- Wheel-odometry velocity fusion (gimbal-heading 2-D horizontal, m/s) --
    # Observes v_w directly so the estimate doesn't coast; the planar vz=0 row is
    # inside the update. v=0 when stopped is the same update (no ZUPT branch).
    odom = sub["chassis_odom"]
    if odom is not None and sub.updated["chassis_odom"]:
      last_vx, last_vy = float(odom["vx"]), float(odom["vy"])
      yaw_offset += st.update_with_velocity(last_vx, last_vy, gimbal_yaw + yaw_offset)
      d_vel += 1

    # --- AprilTag detection (throttled) → absolute position + yaw fixes ------
    # tag_dets stays None on frames where detection didn't run so the viewer
    # leaves the overlay alone (drawing [] would blink it off at frame rate).
    seen_tag_p, tag_dets = [], None
    if now - last_tag_t >= 1.0 / TAG_DETECT_HZ:
      last_tag_t = now
      tag_dets = []
      gray = cv2.cvtColor(np.frombuffer(cam["frame"], dtype=np.uint8).reshape(
        common.IMG_H, common.IMG_W, 3), cv2.COLOR_RGB2GRAY)
      corners, ids, _ = tag_detector.detectMarkers(gray)
      if ids is not None:
        for tag_id, c in zip(ids.flatten(), corners):
          c4 = c.reshape(4, 2)
          tag_dets.append({"id": int(tag_id), "corners": c4.astype(float).tolist()})
          if not common.FUSE_APRILTAGS: continue   # detect for the viz; don't fuse until map+extrinsics are real
          res = _tag_body_pose(int(tag_id), c4, p_offset, R_ic)
          if res is None: continue
          p_wb, yaw_tag, rng = res
          # noise grows with tag range — far PnP is much noisier (see common.py)
          pos_std = common.TAG_POS_NOISE + common.TAG_POS_NOISE_PER_M * rng
          yaw_std = common.TAG_YAW_NOISE + common.TAG_YAW_NOISE_PER_M * rng
          yaw_offset += st.update_with_position(p_wb, pos_std)
          yaw_offset += st.update_with_yaw(wrap_pi(yaw_tag - (gimbal_yaw + yaw_offset)), yaw_std)
          n_tags_total += 1; d_tags += 1
          seen_tag_p.append(p_wb.tolist())

    # --- Diagnostics --------------------------------------------------------
    if now - diag_t > 2.0:
      span = now - diag_t
      logger.info(
        f"slamd flow: frames {d_frames/span:.0f}/s  imu {d_imu/span:.0f}/s  "
        f"tags {d_tags/span:.0f}/s  |accel|~{d_acc_mag/max(d_acc_n,1):.2f}  "
        f"a_w.z~{d_aw_z/max(d_acc_n,1):+.2f}  vel {d_vel/span:.0f}/s  "
        f"|v|={float(np.linalg.norm(st.v_w)):.2f}  gyaw={np.degrees(gimbal_yaw):+.0f}deg  "
        f"odom=[{last_vx:+.2f},{last_vy:+.2f}]  p_w={st.p_w.round(3).tolist()}")
      diag_t = now
      d_imu = d_tags = d_frames = d_vel = 0; d_acc_mag = d_aw_z = 0.0; d_acc_n = 0

    # --- Publish ------------------------------------------------------------
    pub.send("slam_pose", {
      "t": ct,
      "p_w": st.p_w.tolist(), "v_w": st.v_w.tolist(),
      "q_wb": _R_to_quat_np(R_wb).tolist(),
      "cov_pos": st.P[0:3, 0:3].flatten().tolist(),
      "n_tags": n_tags_total,
    })
    pub.send("slam_debug", {"t": ct, "seen_tag_p": seen_tag_p, "tag_dets": tag_dets})
