"""BENCH calibration of the SLAM camera mount R_CAM (camera <- gimbal-yaw-stage).

All calibration must be bench-doable (no field / surveyed-tag access), so this
uses a CHECKERBOARD at an arbitrary fixed pose + gimbal YAW motion. No tags, no
survey, no field.

The SLAM camera is on the gimbal YAW stage only, so R_wc = Rz(gimbal_yaw)·R_CAM
and p_cam_w = p_axis + Rz(gimbal_yaw)·T_CAM, with R_CAM (tilt) and T_CAM (the
optical-center lever arm off the yaw axis) fixed. We recover BOTH from one
pure-yaw sweep over a fixed board:

  R_CAM tilt — as the gimbal yaws, the board's apparent rotation axis in the
  camera is world-up-in-camera = R_CAM.T·ẑ (constant). Average it. (Rotation
  only, so this part ignores the board's metric scale.)

  T_CAM lever arm — the camera swings on a circle about the yaw axis, so with the
  board fixed, R_wc·t_ct + p_cam_w is constant ⇒ a_i + T_CAM = Rz(-yaw)·N with
  a_i = R_CAM·t_ct_i; a per-session linear LSQ gives the in-plane T_CAM. This
  part DOES need metric scale, so pass the real square size.

Single yaw axis ⇒ two things can't be seen on the bench: the ABSOLUTE heading
(which way world +x points) — supplied at game time by the AprilTag yaw fix
(yaw_offset) — and T_CAM's VERTICAL component (it doesn't move as you yaw, so it
can't bias planar position; we output T_CAM_z = 0). We DO pin the heading GAUGE
(see "heading-gauge alignment" below) so the runtime `heading` equals the camera's
forward azimuth — otherwise it comes out ~90° off and rotates navd/wheel motion.

Bench setup: robot powered (camerad + commsd up), a checkerboard fixed in view;
slew the gimbal yaw. Narrow FOV / small board is fine: do several SHORT sweeps,
repositioning the (still) board between them — a detection gap starts a new
"session", and T_CAM is solved per-session (R_CAM combines across all). cols/rows
= INNER corner counts (default 9x6); use an asymmetric board (cols != rows) so
the corner order can't flip 180deg. square = square size in mm (default 12.5).
Run:  python -m cv.tools.calib_slam_cam <addr> [cols] [rows] [square_mm]  (Ctrl-C to finish)
"""
import sys

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

from ..system.core import messaging
from ..system.common.gimbal import GimbalBuffer
from ..slam import common

_MIN_PAIR_DYAW = np.radians(10.0)   # need this much yaw separation for a clean axis
# Reject a pair whose recovered rotation magnitude disagrees with |Δyaw| by more
# than this — that's a board flip OR a cross-session pair (board was repositioned
# between mini-sweeps), neither of which is a pure-yaw rotation. This is what lets
# you do many short sweeps at different board placements and combine them safely.
_ANG_TOL = np.radians(6.0)
# A detection gap longer than this means the board was repositioned → new session.
# T_CAM is solved per-session (the board pose must be fixed within a session).
_SESSION_GAP = 1.0

def main():
  if len(sys.argv) < 2:
    print("usage: python -m cv.tools.calib_slam_cam <addr> [cols] [rows]  (default 9x6 inner corners)")
    sys.exit(1)
  addr = sys.argv[1]
  cols = int(sys.argv[2]) if len(sys.argv) > 2 else 9   # inner-corner counts; override per board
  rows = int(sys.argv[3]) if len(sys.argv) > 3 else 6
  square = (float(sys.argv[4]) if len(sys.argv) > 4 else 12.5) / 1000.0   # mm -> m; T_CAM needs scale (our board = 12.5mm)
  obj = np.zeros((rows * cols, 3), np.float32)
  obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square   # metric squares (T_CAM lever arm)
  crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

  sub = messaging.Sub(["camera_feed_full"], poll="camera_feed_full", addr=addr)   # full frames via the framed bridge (DEBUG>=1)
  gsub = messaging.Sub(["gimbal_state"], conflate=False, addr=addr)
  gbuf = GimbalBuffer()
  R_ct, yaws, tvecs, ts = [], [], [], []

  print(f"checkerboard {cols}x{rows} inner corners @ {square * 1000:.1f}mm. Narrow FOV is "
        f"fine: do several SHORT yaw sweeps, repositioning the (still) board between them "
        f"so it stays in view (a detection gap = new session). Get the board LARGE in "
        f"frame — T_CAM needs clean PnP translation. 'q' or Ctrl-C to finish.")
  show = True
  try:
    while True:
      sub.update(timeout=200)
      for m in gsub.drain("gimbal_state"): gbuf.push(m)
      cam = sub["camera_feed_full"]
      if cam is None or not sub.updated["camera_feed_full"]: continue
      rgb = np.frombuffer(cam["frame"], np.uint8).reshape(common.IMG_H, common.IMG_W, 3)
      gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
      found, corners = cv2.findChessboardCorners(gray, (cols, rows))
      if found:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
        ok, rvec, tvec = cv2.solvePnP(obj, corners, common.K, np.zeros((1, 5), np.float32))
        gp = gbuf.interpolate(float(cam["ct"]))
        if ok and gp is not None:
          R_ct.append(cv2.Rodrigues(rvec)[0]); tvecs.append(tvec.reshape(3))
          yaws.append(float(gp[0])); ts.append(float(cam["ct"]))
      if show:
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if found: cv2.drawChessboardCorners(vis, (cols, rows), corners, found)
        yr = np.degrees(np.ptp(yaws)) if yaws else 0.0
        cv2.putText(vis, f"n={len(R_ct)}  yaw_range={yr:.0f}deg  {'FOUND' if found else 'no board'}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if found else (0, 0, 255), 2)
        try:
          cv2.imshow("calib_slam_cam", vis)
          if (cv2.waitKey(1) & 0xFF) == ord('q'): break
        except cv2.error:
          show = False; print("(no display available — continuing headless; Ctrl-C to finish)")
  except KeyboardInterrupt:
    pass
  finally:
    try: cv2.destroyAllWindows()
    except cv2.error: pass

  if len(R_ct) < 2 or np.ptp(yaws) < _MIN_PAIR_DYAW:
    print("not enough data / yaw range — sweep the gimbal yaw more"); return

  # Relative board-in-camera rotation R_ct[i]·R_ct[j].T is by (yaw_j - yaw_i)
  # about world-up-in-camera (R_CAM.T·ẑ). Recover that axis (aligned by Δyaw),
  # DROP pairs whose recovered angle disagrees with |Δyaw| (board flip / bad PnP),
  # and weight by rotation size (big rotations define the axis far more sharply).
  axes, wts, ang_err, dropped = [], [], [], 0
  for i in range(len(R_ct)):
    for j in range(i + 1, len(R_ct)):
      dy = yaws[j] - yaws[i]
      if abs(dy) < _MIN_PAIR_DYAW: continue
      rv = Rot.from_matrix(R_ct[i] @ R_ct[j].T).as_rotvec()
      ang = np.linalg.norm(rv)
      if ang < 1e-3: continue
      if abs(ang - abs(dy)) > _ANG_TOL: dropped += 1; continue   # inconsistent → outlier
      axes.append(rv / ang * np.sign(dy)); wts.append(abs(dy)); ang_err.append(abs(ang - abs(dy)))
  if not axes:
    print("all pairs rejected — yaw scale/sign or board detection is off"); return
  axes = np.array(axes); wts = np.array(wts)
  up_cam = (axes * wts[:, None]).sum(0); up_cam /= np.linalg.norm(up_cam)
  spread = float(np.degrees(np.mean([np.arccos(np.clip(a @ up_cam, -1, 1)) for a in axes])))
  # Sign disambiguation = the gimbal-yaw convention. For an upright forward camera
  # world-up must be ≈ camera -y (up in the image). If it comes out +y, the gimbal
  # yaw increases opposite to Rz → negate yaw_gi in the runtime model; the true
  # up (and the R_CAM below) is then -up_cam.
  yaw_flipped = bool(up_cam[1] > 0)
  if yaw_flipped: up_cam = -up_cam
  print(f"\nn={len(R_ct)}  pairs={len(axes)}  yaw_range={np.degrees(np.ptp(yaws)):.0f}deg")
  print(f"axis spread = {spread:.2f}deg (low = clean)   |angle-Δyaw| = {np.degrees(np.mean(ang_err)):.2f}deg "
        f"(≈0 confirms yaw is radians & scaled right)")
  print(f"world-up in camera = {up_cam.round(3).tolist()} (should be ≈ [0,-1,0])")
  if yaw_flipped:
    print("NOTE: gimbal-yaw sign is FLIPPED vs Rz — negate yaw_gi in the runtime "
          "model; this R_CAM is for the negated convention.")

  # R_CAM tilt: shortest rotation taking up_cam → world +z (fixes how world-up sits
  # in the camera). R_CAM maps camera axes into the yaw-stage frame.
  z = np.array([0.0, 0.0, 1.0])
  v = np.cross(up_cam, z); s = float(np.linalg.norm(v)); c = float(up_cam @ z)
  if s < 1e-9:
    R_CAM = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
  else:
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_CAM = np.eye(3) + K + K @ K * ((1 - c) / s**2)
  # Heading-gauge alignment. The shortest rotation above leaves an arbitrary spin
  # about world-z, but the runtime needs `heading` (gimbal_yaw + δψ) to mean the
  # camera's world-forward AZIMUTH — the wheel-velocity update and navd both assume
  # forward = Rz(heading)·x̂. So rotate the gauge about z until the optical axis
  # (R_CAM·ẑ) has zero azimuth: heading 0 ⇒ camera looks along world +x. δψ still
  # supplies the absolute heading at runtime; this only pins the convention. T_CAM
  # is solved below in this SAME aligned R_CAM frame (a_i = R_CAM·t_ct), so they
  # share one gauge and the runtime δψ removes it for both. WITHOUT this the gauge
  # is ~90° off → the heading/navd/wheel-velocity directions are all rotated.
  g = float(np.arctan2((R_CAM @ z)[1], (R_CAM @ z)[0]))
  cg, sg = np.cos(-g), np.sin(-g)
  R_CAM = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]]) @ R_CAM
  print("\nR_CAM (camera <- yaw-stage, heading-gauge aligned), paste into cv/slam/common.py:")
  print("R_CAM = np.array(" + np.array2string(R_CAM, precision=5, separator=", ",
        max_line_width=100).replace("\n", "\n                 ") + ", dtype=np.float32)")

  # ---- T_CAM: optical-center lever arm off the yaw axis (in-plane part) ----
  # Board fixed in world ⇒ R_wc·t_ct + p_cam_w is constant, and
  # p_cam_w = p_axis + Rz(yaw_eff)·T_CAM, so   a_i + T_CAM = Rz(-yaw_eff)·N_sess
  # with a_i = R_CAM·t_ct_i. The heading (yaw_offset) and T_CAM_z fold into the
  # per-session N → unobservable here (heading absorbed at runtime by tags; the
  # vertical offset can't bias planar p). Linear LSQ for [T_x, T_y, N per session].
  ye = (-np.array(yaws) if yaw_flipped else np.array(yaws))     # runtime yaw convention
  a = (R_CAM @ np.array(tvecs)[:, :, None])[:, :2, 0]           # a_i in-plane (N,2)
  sess = np.cumsum(np.concatenate([[0], (np.diff(ts) > _SESSION_GAP).astype(int)]))
  ns = int(sess.max()) + 1
  M = np.zeros((2 * len(a), 2 + 2 * ns)); b = np.zeros(2 * len(a))
  c, s = np.cos(ye), np.sin(ye)
  for k in range(len(a)):                                       # row_x: T_x - cosΨ·Nx - sinΨ·Ny = -a_x
    sx = 2 + 2 * sess[k]
    M[2*k, 0], M[2*k, sx], M[2*k, sx+1], b[2*k] = 1.0, -c[k], -s[k], -a[k, 0]
    M[2*k+1, 1], M[2*k+1, sx], M[2*k+1, sx+1], b[2*k+1] = 1.0, s[k], -c[k], -a[k, 1]
  sol = np.linalg.lstsq(M, b, rcond=None)[0]
  T_CAM = np.array([sol[0], sol[1], 0.0], np.float32)
  rms = float(np.sqrt(np.mean((M @ sol - b) ** 2)))
  print(f"\nsessions={ns}  T_CAM lever-arm |xy| = {np.hypot(sol[0], sol[1]) * 1000:.1f}mm  "
        f"fit RMS = {rms * 1000:.1f}mm   (vertical offset unobservable → 0)")
  print("T_CAM = np.array(" + np.array2string(T_CAM, precision=5, separator=", ") + ", dtype=np.float32)")

if __name__ == "__main__":
  main()
