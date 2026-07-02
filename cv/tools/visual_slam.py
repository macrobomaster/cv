"""rerun visualizer for the field-localization stack.

Run with:
  python -m cv.tools.visual_slam <addr>

Subscribes to camera_feed, apriltags, slam_pose, slam_debug, nav_debug,
cam_occupancy. Shows:
  - 2D camera view with AprilTag detections (from tagd's apriltags topic)
  - 3D world: robot trajectory, current pose marker, known field AprilTags,
    and tags actually used for a fix this frame (slam_debug.seen_tag_p)
  - Nav: the field boundary + walls (NAV_MAP), the active goal, planned path,
    and detected enemy-robot obstacle circles (navd's nav_debug)
  - Occupancy: occupancyd's camera-derived obstacle cells (cam_occupancy)
  - Scalars: number of tags fixed, position-stddev
"""
import sys, os, gc, json, time
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

  # SLAM-camera frustum (the real, yaw-only camera): it sits a lever arm off the
  # robot/yaw-axis centre and swings as the gimbal pans. Positioned each frame at
  # slam_debug.p_cam with slam_pose.q_wb (= the SLAM-cam orientation).
  rr.log("world/slam_cam",
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

  # Static nav map: the 12×8 field boundary + walls/named goals from NAV_MAP (what navd plans on).
  fx0, fy0, fx1, fy1 = common.FIELD_BOUNDS
  rr.log("world/field", rr.LineStrips3D([[[fx0, fy0, 0], [fx1, fy0, 0], [fx1, fy1, 0], [fx0, fy1, 0],
         [fx0, fy0, 0]]], colors=[(110, 110, 110)], radii=[0.012]), static=True)
  try:
    with open(os.environ["NAV_MAP"]) as f: nav_map = json.load(f)
  except (KeyError, OSError, json.JSONDecodeError): nav_map = {}
  rects = [w["rect"] for w in nav_map.get("walls", []) if "rect" in w]
  if rects:
    rr.log("world/walls", rr.Boxes3D(
      centers=[[(r[0] + r[2]) / 2, (r[1] + r[3]) / 2, 0.15] for r in rects],
      half_sizes=[[abs(r[2] - r[0]) / 2, abs(r[3] - r[1]) / 2, 0.15] for r in rects],
      colors=[(90, 90, 90)]), static=True)
  if nav_map.get("goals"):
    rr.log("world/named_goals", rr.Points3D([[g["x"], g["y"], 0] for g in nav_map["goals"]],
           colors=[(180, 120, 255)], radii=0.07, labels=[g.get("label", "") for g in nav_map["goals"]]),
           static=True)

  # Semantic zones (spawn/capture) from slam.common.FIELD_ZONES — outline + centre label on the ground.
  for z in common.FIELD_ZONES:
    corners = common.zone_corners(z)
    loop = [[x, y, 0.01] for x, y in corners] + [[corners[0][0], corners[0][1], 0.01]]
    ent = "world/zones/" + z["name"].replace(" ", "_")
    rr.log(ent, rr.LineStrips3D([loop], colors=[z["color"]], radii=[0.02]), static=True)
    cx = sum(x for x, _ in corners) / len(corners); cy = sum(y for _, y in corners) / len(corners)
    rr.log(ent + "_lbl", rr.Points3D([[cx, cy, 0.02]], colors=[z["color"]], radii=0.02,
           labels=[z["name"]]), static=True)

  sub = messaging.Sub(["camera_feed_full", "slam_pose", "slam_debug", "apriltags", "nav_debug",
                       "cam_occupancy", "cam_occupancy_debug"],
                      poll="camera_feed_full", addr=addr)   # full frames via the framed bridge (DEBUG>=1)
  fk = FrequencyKeeper(30)

  for s in sub.services:
    rr.log(f"alive/{s}", rr.SeriesLines(widths=[10]), static=True)

  traj = deque(maxlen=TRAJ_LEN)
  n_dbg_received = 0
  n_cam_received = 0
  last_traj_log_t = 0.0
  last_p_w = last_q = None      # carried from slam_pose into the slam_debug block (same tick)
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
    cam = sub["camera_feed_full"]
    if cam is not None and sub.updated["camera_feed_full"]:
      n_cam_received += 1
      img = np.frombuffer(cam["frame"], dtype=np.uint8).reshape(common.IMG_H, common.IMG_W, 3)
      rr.log("camera/feed", rr.Image(img).compress(70))

    # --- Live pose --------------------------------------------------------
    pose = sub["slam_pose"]
    if pose is not None and sub.updated["slam_pose"]:
      p = pose["p_w"]
      traj.append(p)
      last_p_w, last_q = p, pose["q_wb"]           # for the camera frustum + lever line below
      # Robot centre = the yaw axis (what the filter tracks); the camera frustum is
      # drawn separately at slam_debug.p_cam, a lever arm away.
      rr.log("world/robot", rr.Points3D([p], colors=[(60, 180, 255)], radii=0.05))
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

    # --- occupancyd's detected obstacle contact line (2D, localization-independent) ------
    cod = sub["cam_occupancy_debug"]
    if cod is not None and sub.updated["cam_occupancy_debug"]:
      pts = cod.get("contact") or []
      rr.log("camera/feed/cam_contact", rr.Points2D(pts, radii=2.0, colors=[(255, 180, 40)])
             if pts else rr.Clear(recursive=True))

    # --- Debug topic: tags actually fused this frame (3D) -----------------
    dbg = sub["slam_debug"]
    if dbg is not None and sub.updated["slam_debug"]:
      n_dbg_received += 1
      seen = np.array(dbg["seen_tag_p"], dtype=np.float32) if dbg.get("seen_tag_p") else np.zeros((0, 3), np.float32)
      if len(seen):
        rr.log("world/seen_tags", rr.Points3D(seen, colors=[(80, 240, 120)], radii=0.06))
      # Real SLAM camera (frustum at the optical centre) + the lever arm from the
      # yaw axis to it: when the gimbal pans in place this swings while the robot
      # centre stays put. Orientation from slam_pose.q_wb (= SLAM-cam frame).
      pc = dbg.get("p_cam")
      if pc is not None and last_q is not None:
        rr.log("world/slam_cam", rr.Transform3D(translation=pc, quaternion=_q_wxyz_to_scipy(last_q)))
        if last_p_w is not None:
          rr.log("world/lever_arm", rr.LineStrips3D([[last_p_w, pc]], colors=[(255, 160, 0)], radii=[0.012]))

    # --- Nav debug: active goal, planned path, dynamic obstacle circles ----
    nav = sub["nav_debug"]
    if nav is not None and sub.updated["nav_debug"]:
      g = nav.get("goal")
      rr.log("world/nav_goal", rr.Points3D([[g[0], g[1], 0.0]], colors=[(255, 0, 255)], radii=0.10)
             if g is not None else rr.Clear(recursive=True))
      path = nav.get("path") or []
      rr.log("world/nav_path", rr.LineStrips3D([[[x, y, 0.02] for x, y in path]], colors=[(0, 220, 0)],
             radii=[0.02]) if len(path) >= 2 else rr.Clear(recursive=True))
      obs = nav.get("obstacles") or []
      rr.log("world/nav_obstacles", rr.Points3D([[o[0], o[1], 0.1] for o in obs], colors=[(255, 60, 60)],
             radii=[o[2] for o in obs]) if obs else rr.Clear(recursive=True))

    # --- Camera occupancy: occupancyd's world obstacle grid (floor-vs-obstacle IPM) -
    co = sub["cam_occupancy"]
    if co is not None and sub.updated["cam_occupancy"]:
      occ = np.frombuffer(co["occ"], bool).reshape(co["ny"], co["nx"])
      iy, ix = np.nonzero(occ)
      if len(ix):
        cx = co["x0"] + (ix + 0.5) * co["res"]
        cy = co["y0"] + (iy + 0.5) * co["res"]
        pts = np.stack([cx, cy, np.full(len(ix), 0.03)], axis=1)
        rr.log("world/cam_occupancy", rr.Points3D(pts, colors=[(255, 180, 40)], radii=co["res"] * 0.5))
      else:
        rr.log("world/cam_occupancy", rr.Clear(recursive=True))

    for k, v in sub.alive.items():
      rr.log(f"alive/{k}", rr.Scalars(int(v)))
    fk.step()

if __name__ == "__main__":
  main()
