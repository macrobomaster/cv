"""rerun visualizer for the field-localization stack.

Run with:
  python -m cv.tools.visual_slam <addr>

Subscribes to camera_feed, apriltags, slam_pose, slam_debug. Shows:
  - 2D camera view with AprilTag detections (from tagd's apriltags topic)
  - 3D world: robot trajectory, current pose marker, known field AprilTags,
    and tags actually used for a fix this frame (slam_debug.seen_tag_p)
  - Scalars: number of tags fixed, position-stddev
"""
import sys, gc, time
from collections import deque

import numpy as np
import rerun as rr

from ..system.core import messaging
from ..system.core.helpers import FrequencyKeeper
from ..slam import common

# Visual params
TRAJ_LEN          = 2000
ROBOT_FRUSTUM_SCALE = 0.15
TRAJ_LOG_HZ       = 5        # re-logging the whole growing trajectory strip is
                            # the heaviest redundant payload; cap its rate

def _q_wxyz_to_scipy(q):
  return np.array([q[1], q[2], q[3], q[0]])

def main():
  if len(sys.argv) < 2:
    print("usage: python -m cv.tools.visual_slam <addr>")
    sys.exit(1)
  addr = sys.argv[1]

  gc.disable()   # avoid GC pauses that intermittently freeze the feed
  rr.init("slam", spawn=True)

  # World frame is +z up. Tell rerun.
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  # `camera` is a 2D image space — do NOT set ViewCoordinates on it; that
  # would tell rerun to treat the subtree as 3D and hide the 2D overlays.

  # Robot frustum (a tiny pinhole entity attached to the live pose).
  rr.log("world/robot",
         rr.Pinhole(resolution=(common.IMG_W, common.IMG_H),
                    image_from_camera=common.K,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=ROBOT_FRUSTUM_SCALE),
         static=True)

  # Static field tag map — the known AprilTag locations (the localization anchors).
  if common.TAG_FIELD_MAP:
    tag_pts = np.array([t for (_, t) in common.TAG_FIELD_MAP.values()], dtype=np.float32)
    tag_lbls = [str(i) for i in common.TAG_FIELD_MAP]
    rr.log("world/field_tags",
           rr.Points3D(tag_pts, colors=[(200, 200, 200)], radii=0.05, labels=tag_lbls),
           static=True)

  sub = messaging.Sub(["camera_feed", "slam_pose", "slam_debug", "apriltags"],
                      poll="camera_feed", addr=addr)
  fk = FrequencyKeeper(30)

  for s in sub.services:
    rr.log(f"alive/{s}", rr.SeriesLines(widths=[10]), static=True)

  traj = deque(maxlen=TRAJ_LEN)
  n_dbg_received = 0
  n_cam_received = 0
  last_traj_log_t = 0.0
  diag_t = time.monotonic()

  while True:
    sub.update(timeout=10)
    rr.set_time("time", duration=time.monotonic())

    # Periodic diagnostic on stdout so we can tell which topic is missing.
    now = time.monotonic()
    if now - diag_t > 2.0:
      _tg = sub["apriltags"]
      n_tag_det = len(_tg["detections"]) if _tg is not None else 0
      print(f"[viz] cam_msgs={n_cam_received} dbg_msgs={n_dbg_received} "
            f"tag_dets={n_tag_det} alive={dict(sub.alive)}")
      diag_t = now

    # --- Camera feed ------------------------------------------------------
    cam = sub["camera_feed"]
    if cam is not None and sub.updated["camera_feed"]:
      n_cam_received += 1
      img = np.frombuffer(cam["frame"], dtype=np.uint8).reshape(common.IMG_H, common.IMG_W, 3)
      rr.log("camera/feed", rr.Image(img))

    # --- Live pose --------------------------------------------------------
    pose = sub["slam_pose"]
    if pose is not None and sub.updated["slam_pose"]:
      p = pose["p_w"]; q = pose["q_wb"]
      traj.append(p)
      rr.log("world/robot", rr.Transform3D(translation=p, quaternion=_q_wxyz_to_scipy(q)))
      # Re-logging the whole growing strip every pose is the biggest redundant
      # payload to rerun — throttle it; the path doesn't need full rate.
      if time.monotonic() - last_traj_log_t >= 1.0 / TRAJ_LOG_HZ:
        rr.log("world/trajectory", rr.LineStrips3D([list(traj)], colors=[(60, 180, 255)], radii=[0.01]))
        last_traj_log_t = time.monotonic()
      pos_std = np.sqrt(np.maximum(np.diag(np.array(pose["cov_pos"], np.float32).reshape(3, 3)), 0))
      rr.log("scalars/pos_std_x", rr.Scalars(float(pos_std[0])))
      rr.log("scalars/pos_std_y", rr.Scalars(float(pos_std[1])))
      rr.log("scalars/pos_std_z", rr.Scalars(float(pos_std[2])))
      rr.log("scalars/n_tags", rr.Scalars(int(pose["n_tags"])))

    # --- AprilTag detections overlay (2D, straight from tagd) -------------
    tg = sub["apriltags"]
    if tg is not None and sub.updated["apriltags"]:
      dets = tg["detections"]
      if dets:
        outlines = [d["corners"] + [d["corners"][0]] for d in dets]  # close the quad
        centers = [list(np.mean(np.array(d["corners"], np.float32), axis=0)) for d in dets]
        labels  = [str(d["id"]) for d in dets]
        rr.log("camera/feed/apriltags", rr.LineStrips2D(outlines, radii=2.0, colors=[(0, 220, 255)]))
        rr.log("camera/feed/apriltag_ids",
               rr.Points2D(centers, radii=4.0, colors=[(0, 220, 255)], labels=labels))
      else:
        rr.log("camera/feed/apriltags", rr.Clear(recursive=True))
        rr.log("camera/feed/apriltag_ids", rr.Clear(recursive=True))

    # --- Debug topic: tags actually fused this frame (3D) -----------------
    dbg = sub["slam_debug"]
    if dbg is not None and sub.updated["slam_debug"]:
      n_dbg_received += 1
      seen = np.array(dbg["seen_tag_p"], dtype=np.float32) if dbg.get("seen_tag_p") else np.zeros((0, 3), np.float32)
      if len(seen):
        rr.log("world/seen_tags", rr.Points3D(seen, colors=[(80, 240, 120)], radii=0.06))

    for k, v in sub.alive.items():
      rr.log(f"alive/{k}", rr.Scalars(int(v)))
    fk.step()

if __name__ == "__main__":
  main()
