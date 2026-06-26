"""rerun visualizer for the SLAM stack.

Run with:
  python -m cv.tools.visual_slam <addr>

Subscribes to camera_feed, feature_tracks, slam_pose, slam_debug. Shows:
  - 2D camera view with live feature tracks (color = track age) — from frontd
  - 3D world: robot trajectory, current pose marker, recent clone positions,
    triangulated points (sparse 3D map), known field AprilTags + tags seen
  - Scalars: number of tracks/clones/tags, position-stddev
"""
import sys, gc, time
from collections import deque

import numpy as np
import rerun as rr

from ..system.core import messaging
from ..system.core.helpers import FrequencyKeeper
from ..slam import calib

# Visual params
TRAJ_LEN          = 2000
TRACK_TRAIL_LEN   = 8        # how many recent obs to draw per live track
RECENT_PT_LEN     = 200
KF_FRUSTUM_SCALE  = 0.05
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

  # World frame: SLAM is +z up. Tell rerun.
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  # `camera` is a 2D image space — do NOT set ViewCoordinates on it; that
  # would tell rerun to treat the subtree as 3D and hide the 2D overlays.

  # Set up a robot frustum (a tiny pinhole entity attached to the live pose)
  rr.log("world/robot",
         rr.Pinhole(resolution=(calib.IMG_W, calib.IMG_H),
                    image_from_camera=calib.K,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=ROBOT_FRUSTUM_SCALE),
         static=True)

  # Static field tag map — the known AprilTag locations (drift anchors).
  if calib.TAG_FIELD_MAP:
    tag_pts = np.array([t for (_, t) in calib.TAG_FIELD_MAP.values()], dtype=np.float32)
    tag_lbls = [str(i) for i in calib.TAG_FIELD_MAP]
    rr.log("world/field_tags",
           rr.Points3D(tag_pts, colors=[(200, 200, 200)], radii=0.05, labels=tag_lbls),
           static=True)

  sub = messaging.Sub(["camera_feed", "feature_tracks", "apriltags", "slam_pose", "slam_debug"],
                      poll="camera_feed", addr=addr)
  fk = FrequencyKeeper(30)

  for s in sub.services:
    rr.log(f"alive/{s}", rr.SeriesLines(widths=[10]), static=True)

  traj = deque(maxlen=TRAJ_LEN)
  trail_uv = {}  # track_id -> deque[(u, v)]
  last_logged_kf = -1
  last_img:np.ndarray|None = None
  n_dbg_received = 0
  n_cam_received = 0
  last_traj_log_t = 0.0
  loop_starts = time.monotonic()
  diag_t = loop_starts

  while True:
    sub.update(timeout=10)
    rr.set_time("time", duration=time.monotonic())

    # Periodic diagnostic on stdout so we can tell which topic is missing.
    now = time.monotonic()
    if now - diag_t > 2.0:
      _tg = sub["apriltags"]
      n_tag_det = len(_tg["detections"]) if _tg is not None else 0
      print(f"[viz] cam_msgs={n_cam_received} dbg_msgs={n_dbg_received} "
            f"tag_dets={n_tag_det} trails={len(trail_uv)} alive={dict(sub.alive)}")
      diag_t = now

    # --- Camera feed ------------------------------------------------------
    # Logged every loop when fresh (conflate already limits it to camera rate),
    # raw not JPEG: on localhost the bandwidth is free and skipping the encode
    # keeps the per-iteration cost low so the feed stays responsive.
    cam = sub["camera_feed"]
    if cam is not None and sub.updated["camera_feed"]:
      n_cam_received += 1
      last_img = np.frombuffer(cam["frame"], dtype=np.uint8).reshape(calib.IMG_H, calib.IMG_W, 3)
      rr.log("camera/feed", rr.Image(last_img))

    # --- Live pose --------------------------------------------------------
    pose = sub["slam_pose"]
    if pose is not None and sub.updated["slam_pose"]:
      p = pose["p_w"]; q = pose["q_wb"]
      traj.append(p)
      quat = _q_wxyz_to_scipy(q)
      rr.log("world/robot",
             rr.Transform3D(translation=p, quaternion=quat))
      # Re-logging the whole growing strip every pose is the biggest redundant
      # payload to rerun — throttle it; the path doesn't need full rate.
      if time.monotonic() - last_traj_log_t >= 1.0 / TRAJ_LOG_HZ:
        rr.log("world/trajectory", rr.LineStrips3D([list(traj)],
                                                    colors=[(60, 180, 255)],
                                                    radii=[0.01]))
        last_traj_log_t = time.monotonic()
      cov = np.array(pose["cov_pose"], dtype=np.float32).reshape(6, 6)
      pos_std = np.sqrt(np.maximum(np.diag(cov)[3:6], 0))
      rr.log("scalars/pos_std_x", rr.Scalars(float(pos_std[0])))
      rr.log("scalars/pos_std_y", rr.Scalars(float(pos_std[1])))
      rr.log("scalars/pos_std_z", rr.Scalars(float(pos_std[2])))
      rr.log("scalars/n_tracks", rr.Scalars(int(pose["n_tracks"])))
      rr.log("scalars/n_clones", rr.Scalars(int(pose["n_clones"])))
      rr.log("scalars/n_tags", rr.Scalars(int(pose["n_tags"])))

    # --- Feature overlay (from frontd) ------------------------------------
    ft = sub["feature_tracks"]
    if ft is not None and sub.updated["feature_tracks"]:
      uvs = np.array(ft["live_uvs"], dtype=np.float32) if ft["live_uvs"] else np.zeros((0, 2), np.float32)
      ages = np.array(ft["live_ages"], dtype=np.int32) if ft["live_ages"] else np.zeros(0, np.int32)
      if len(uvs):
        max_age = max(int(ages.max()), 1)
        colors = np.zeros((len(uvs), 3), dtype=np.uint8)
        a01 = ages.astype(np.float32) / max_age
        colors[:, 0] = ((1 - a01) * 255).astype(np.uint8)
        colors[:, 1] = (a01 * 255).astype(np.uint8)
        rr.log("camera/feed/features", rr.Points2D(uvs, colors=colors, radii=3.0))
        tids = ft["live_ids"]
        alive_tids = set(tids)
        for tid in list(trail_uv.keys()):
          if tid not in alive_tids: trail_uv.pop(tid, None)
        for uv, tid in zip(uvs, tids):
          d = trail_uv.setdefault(int(tid), deque(maxlen=TRACK_TRAIL_LEN))
          d.append((float(uv[0]), float(uv[1])))
        strips = [list(d) for d in trail_uv.values() if len(d) >= 2]
        if strips:
          rr.log("camera/feed/trails",
                 rr.LineStrips2D(strips, radii=1.0, colors=[(255, 220, 60)]))

    # --- AprilTag detections overlay (raw, from frontd) -------------------
    # Drawn straight from the detector, so it shows even when TAG_FIELD_MAP is
    # empty (i.e. before the field is surveyed). Green outline + id label.
    tags = sub["apriltags"]
    if tags is not None and sub.updated["apriltags"]:
      dets = tags["detections"]
      if dets:
        outlines = [d["corners"] + [d["corners"][0]] for d in dets]  # close the quad
        centers = [list(np.mean(np.array(d["corners"], np.float32), axis=0)) for d in dets]
        labels  = [str(d["id"]) for d in dets]
        rr.log("camera/feed/apriltags",
               rr.LineStrips2D(outlines, radii=2.0, colors=[(0, 220, 255)]))
        rr.log("camera/feed/apriltag_ids",
               rr.Points2D(centers, radii=4.0, colors=[(0, 220, 255)], labels=labels))
      else:
        rr.log("camera/feed/apriltags", rr.Clear(recursive=True))
        rr.log("camera/feed/apriltag_ids", rr.Clear(recursive=True))

    # --- Debug topic (3D world only now) ----------------------------------
    dbg = sub["slam_debug"]
    if dbg is not None and sub.updated["slam_debug"]:
      n_dbg_received += 1
      if n_dbg_received % 30 == 0:
        print(f"[viz] slam_debug #{n_dbg_received}: "
              f"clones={len(dbg.get('clone_p', []) or [])} "
              f"tags_seen={len(dbg.get('seen_tag_p', []) or [])} "
              f"pts={len(dbg.get('recent_points', []) or [])}")

      # 3D clone frustums (recent past poses) — render as small markers
      clone_p = np.array(dbg["clone_p"], dtype=np.float32) if dbg["clone_p"] else np.zeros((0, 3), np.float32)
      if len(clone_p):
        rr.log("world/clones", rr.Points3D(clone_p, colors=[(120, 160, 200)], radii=0.025))

      # Sparse 3D point cloud from triangulations
      pts = np.array(dbg["recent_points"], dtype=np.float32) if dbg["recent_points"] else np.zeros((0, 3), np.float32)
      if len(pts):
        rr.log("world/points", rr.Points3D(pts, colors=[(255, 200, 80)], radii=0.015))

      # AprilTags seen this frame (world positions the absolute-pose fix used)
      seen = np.array(dbg["seen_tag_p"], dtype=np.float32) if dbg.get("seen_tag_p") else np.zeros((0, 3), np.float32)
      if len(seen):
        rr.log("world/seen_tags", rr.Points3D(seen, colors=[(80, 240, 120)], radii=0.06))

    for k, v in sub.alive.items():
      rr.log(f"alive/{k}", rr.Scalars(int(v)))
    fk.step()

if __name__ == "__main__":
  main()
