"""Live camera intrinsics calibration.

Produces REAL_CAMERA_MATRIX / REAL_DIST_COEFFS — the calibration camerad uses to build its
real→canonical warp. Critically this must run on RAW (un-warped) frames: camerad publishes the
already-undistorted canonical image on `camera_feed`, so calibrating off that is circular and wrong.

To avoid fighting camerad for the camera (which triggers its usbreset and freezes everything), we do
NOT open the camera here. Run camerad with CALIB=1 so it publishes the raw pre-warp frame on
`camera_feed_raw`, and this tool subscribes to it. camerad already handles both the real camera and
WEBCAM=<idx> (dev), so this tool is camera-agnostic.

  CALIB=1 python -m cv.system.camerad.camerad     # real camera; or add WEBCAM=<idx> for dev
  python -m cv.tools.calibrate                     # in another shell

Keys:  s = capture (board found + novel)   d = discard last   c = calibrate now   q = quit
Ctrl-C also finalizes (calibrates if enough views) — handy if the GUI keyboard is flaky.
"""
import time, argparse
from pathlib import Path

import cv2
import numpy as np

from ..system.core import messaging

# Must match camerad's pre-warp resize target.
CALIB_W, CALIB_H = 512, 256
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
COVER_COLS, COVER_ROWS = 8, 4   # frame-coverage grid; distortion needs corners near the edges

def grab(sub):
  """Return a 512x256 BGR frame, or None if no new frame is available (never blocks/hangs)."""
  sub.update(timeout=200)
  msg = sub["camera_feed_raw"]
  if msg is None or not sub.updated["camera_feed_raw"]:
    return None
  frame = np.frombuffer(msg["frame"], dtype=np.uint8).reshape(CALIB_H, CALIB_W, 3)
  return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def novel_enough(corners, accepted, min_centroid_dist=40.0):
  """True if this view's corner centroid is far from every accepted view (anti-duplicate)."""
  c = corners.reshape(-1, 2).mean(axis=0)
  for prev in accepted:
    if np.linalg.norm(c - prev.reshape(-1, 2).mean(axis=0)) < min_centroid_dist:
      return False
  return True

def mark_coverage(grid, corners):
  for (px, py) in corners.reshape(-1, 2):
    cx = min(COVER_COLS - 1, int(px / CALIB_W * COVER_COLS))
    cy = min(COVER_ROWS - 1, int(py / CALIB_H * COVER_ROWS))
    grid[cy, cx] += 1

def draw_coverage(frame, grid):
  cw, ch = CALIB_W // COVER_COLS, CALIB_H // COVER_ROWS
  for cy in range(COVER_ROWS):
    for cx in range(COVER_COLS):
      if grid[cy, cx] > 0:
        x0, y0 = cx * cw, cy * ch
        cv2.rectangle(frame, (x0, y0), (x0 + cw, y0 + ch), (40, 120, 40), 1)

def finalize(objpoints, imgpoints, out_path):
  """Calibrate, report reprojection error, print copyable values, save .npz."""
  if len(objpoints) < 3:
    print(f"\nnot enough views ({len(objpoints)}) to calibrate")
    return
  print(f"\ncalibrating on {len(objpoints)} views...")
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (CALIB_W, CALIB_H), None, None)
  per_view = []
  for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    per_view.append(cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj))
  per_view = np.array(per_view)
  print(f"overall RMS reprojection error: {ret:.4f} px")
  if ret > 1.0:
    print("WARNING: RMS > 1px — recapture with more edge coverage and varied angles")
  print(f"worst view: #{int(np.argmax(per_view))} at {per_view.max():.3f} px")
  print("\n# auto-loaded from the saved .npz by plated; or paste into plated.py:")
  print("REAL_CAMERA_MATRIX =", repr(mtx).replace("array(", "").replace(")", ""))
  print("REAL_DIST_COEFFS =", repr(dist).replace("array(", "").replace(")", ""))
  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, mtx=mtx, dist=dist, rms=ret, image_size=np.array([CALIB_W, CALIB_H]))
  print(f"\nsaved → {out_path}")

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
  ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
  ap.add_argument("--square", type=float, default=12.7, help="square size in mm")
  ap.add_argument("--min-views", type=int, default=15)
  ap.add_argument("--addr", type=str, default="127.0.0.1", help="camerad address")
  ap.add_argument("--out", type=str,
                  default=str(Path(__file__).parent.parent.parent / "weights" / "camera_calib.npz"))
  args = ap.parse_args()

  objp = np.zeros((args.rows * args.cols, 3), np.float32)
  objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

  sub = messaging.Sub(["camera_feed_raw"], addr=args.addr)
  print(f"subscribing to camera_feed_raw @ {args.addr} — run camerad with CALIB=1")

  objpoints, imgpoints = [], []
  coverage = np.zeros((COVER_ROWS, COVER_COLS), dtype=int)
  last_hb = 0.0
  frames_seen = 0

  try:
    while True:
      frame = grab(sub)
      now = time.monotonic()

      if frame is None:
        ph = np.zeros((CALIB_H, CALIB_W, 3), np.uint8)
        cv2.putText(ph, "waiting for frames (camerad CALIB=1?)", (8, CALIB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("calibrate", ph)
        if (cv2.waitKey(50) & 0xFF) == ord("q"):
          break
        if now - last_hb > 1.0:
          print(f"[calib] no frames yet — is camerad running with CALIB=1? (views {len(objpoints)})")
          last_hb = now
        continue

      frames_seen += 1
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), FIND_FLAGS)
      corners_refined = None
      if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners_refined, found)

      draw_coverage(frame, coverage)
      covered = int((coverage > 0).sum())
      hud = f"views {len(objpoints)}/{args.min_views}  coverage {covered}/{COVER_COLS*COVER_ROWS}"
      cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (0, 255, 0) if found else (0, 0, 255), 1)
      if len(objpoints) >= args.min_views:
        cv2.putText(frame, "enough views — press 'c' to calibrate", (8, CALIB_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
      frame = cv2.resize(frame, (2048, 1024))
      cv2.imshow("calibrate", frame)
      key = cv2.waitKey(1) & 0xFF

      if key == ord("s") and found:
        if novel_enough(corners_refined, imgpoints):
          objpoints.append(objp.copy())
          imgpoints.append(corners_refined)
          mark_coverage(coverage, corners_refined)
          print(f"[calib] captured view {len(objpoints)}")
        else:
          print("[calib] skipped: too similar to an existing view — move the board")
      elif key == ord("d") and objpoints:
        objpoints.pop(); imgpoints.pop()
        coverage[:] = 0
        for c in imgpoints:
          mark_coverage(coverage, c)
        print(f"[calib] discarded last view ({len(objpoints)} remain)")
      elif key == ord("c") and len(objpoints) >= args.min_views:
        finalize(objpoints, imgpoints, args.out)
      elif key == ord("q"):
        break

      if now - last_hb > 1.0:
        print(f"[calib] frames {frames_seen}  views {len(objpoints)}/{args.min_views}  "
              f"board={'yes' if found else 'no'}")
        last_hb = now
  except KeyboardInterrupt:
    print("\n[calib] interrupted — finalizing")
    finalize(objpoints, imgpoints, args.out)

  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
