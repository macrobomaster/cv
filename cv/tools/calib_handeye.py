"""Hand-eye extrinsics: solve R_MOUNT (camera→gimbal) and T_MOUNT (camera optical center in the
gimbal-end-effector frame) from a bag recorded while sweeping the gimbal over a STATIC plate.

Principle: a world-fixed target must map to a CONSTANT gimbal-inertial position
    pos_gi = G(yaw,pitch) · (R_MOUNT · pos_cam + T_MOUNT)
across every gimbal pose. So we fit R_MOUNT, T_MOUNT to minimize the spread of pos_gi over the
sweep. This also quantifies the camera lever-arm — the "is the camera below the pivot, by how much"
question — as the y component of T_MOUNT.

Record with the gun pointed at a fixed plate, slowly panning/tilting the gimbal over its range:
    python -m cv.tools.bag handeye.bag --no-camera     # while sweeping
    python -m cv.tools.calib_handeye handeye.bag

Writes weights/handeye_calib.npz (R_MOUNT, T_MOUNT) and prints copyable values for autoaim/common.py.
"""
import argparse, math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .bag import load_bag
from ..autoaim.common import R_MOUNT as R_NOMINAL, plate_screw_points
from ..system.plated.plated import R_yaw, R_pitch, _pnp, _SCALE_PX, GIMBAL_STALE_GAP

def _interp_gimbal(samples, t):
  """Linear-interpolate (yaw_gi, pitch_gi) at t; None if outside the buffer by > stale gap."""
  ts = samples[:, 0]
  if t < ts[0] - GIMBAL_STALE_GAP or t > ts[-1] + GIMBAL_STALE_GAP:
    return None
  yaw = np.interp(t, ts, samples[:, 1])
  pitch = np.interp(t, ts, samples[:, 2])
  return yaw, pitch

def build_observations(bag):
  """→ list of (pos_cam (3,), yaw, pitch). One per valid autoaim frame with a gimbal bracket."""
  gimbal = bag.get("gimbal_state", [])
  if not gimbal: raise SystemExit("no gimbal_state in bag")
  gs = np.array([[d["t_stamp"], d["yaw_gi"], d["pitch_gi"]] for _, d in gimbal])
  gs = gs[np.argsort(gs[:, 0])]

  obs = []
  for _, d in bag.get("autoaim", []):
    if not d.get("valid"): continue
    g = _interp_gimbal(gs, d["t_capture"])
    if g is None: continue
    corners = np.array(d["corners"], dtype=np.float32).reshape(4, 2) * _SCALE_PX
    pos_cam = _pnp(corners, plate_screw_points(d["number"]))
    if pos_cam is None: continue
    obs.append((pos_cam, g[0], g[1]))
  return obs

def fit(obs):
  Gs = np.array([R_yaw(y) @ R_pitch(p) for _, y, p in obs])      # (N,3,3)
  pcam = np.array([pc for pc, _, _ in obs])                       # (N,3)

  def residuals(x):
    R = R_NOMINAL @ Rotation.from_rotvec(x[:3]).as_matrix()       # perturb off the nominal axis-map
    t = x[3:]
    pgi = np.einsum("nij,nj->ni", Gs, pcam @ R.T + t)             # G·(R·pcam + t)
    return (pgi - pgi.mean(axis=0)).ravel()                       # spread about the (free) target

  res = least_squares(residuals, np.zeros(6), method="lm")
  R = R_NOMINAL @ Rotation.from_rotvec(res.x[:3]).as_matrix()
  t = res.x[3:]
  pgi = np.einsum("nij,nj->ni", Gs, pcam @ R.T + t)
  return R, t, pgi

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("bag")
  ap.add_argument("--out", default=str(Path(__file__).parent.parent.parent / "weights" / "handeye_calib.npz"))
  args = ap.parse_args()

  obs = build_observations(load_bag(args.bag))
  if len(obs) < 20:
    raise SystemExit(f"only {len(obs)} usable frames — sweep the gimbal more over a static plate")
  yaws = np.array([y for _, y, _ in obs]); pitches = np.array([p for _, _, p in obs])
  print(f"{len(obs)} frames  yaw span {math.degrees(yaws.ptp()):.0f}°  pitch span {math.degrees(pitches.ptp()):.0f}°")
  if yaws.ptp() < math.radians(20) or pitches.ptp() < math.radians(15):
    print("WARNING: small pose span — extrinsics will be poorly observed; sweep wider")

  R, t, pgi = fit(obs)
  spread_mm = pgi.std(axis=0) * 1000
  print(f"\ntarget pos_gi = {np.round(pgi.mean(axis=0), 3).tolist()} m")
  print(f"residual spread (1σ): x={spread_mm[0]:.1f} y={spread_mm[1]:.1f} z={spread_mm[2]:.1f} mm "
        f"(lower = better; this is how constant the static target ended up)")
  dr = Rotation.from_matrix(R_NOMINAL.T @ R).as_rotvec()
  print(f"misalignment from nominal axis-map: {np.round(np.degrees(dr), 2).tolist()}° (rotvec)")
  print(f"camera lever-arm T_MOUNT: {np.round(t * 1000, 1).tolist()} mm  (y<0 ⇒ camera below pivot)")

  print("\n# paste into cv/autoaim/common.py:")
  print("R_MOUNT =", np.array2string(R, separator=", ", prefix="R_MOUNT = "))
  print("T_MOUNT = np.array(" + str(np.round(t, 5).tolist()) + ")")
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.out, R_MOUNT=R, T_MOUNT=t, residual_mm=spread_mm, n=len(obs))
  print(f"\nsaved → {args.out}")

if __name__ == "__main__":
  main()
