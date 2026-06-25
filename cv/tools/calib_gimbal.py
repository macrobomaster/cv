"""Gimbal system-ID: command safe aim_error pulses and fit the response to recover
DELTA_INPUT (command→motion onset), GIMBAL_TAU (first-order settle), GIMBAL_OMEGA_MAX (peak slew).

Also classifies the response as SETTLE (firmware servos aim_error to a position) vs RAMP (firmware
treats it as a rate command) — the open question for decisiond's control law.

Run with decisiond STOPPED (it also publishes aim_error). The gimbal WILL move — keep the area clear:
    python -m cv.tools.calib_gimbal --axis yaw

Sweeps several amplitudes (small → K_JOYSTICK/TAU, large → OMEGA_MAX). Prints estimates and saves
weights/gimbal_sysid.npz (raw traces for inspection in Rerun).
"""
import argparse, time, math
from pathlib import Path

import numpy as np

from ..system.core import messaging

def _analyze_step(t, ang, rate, t_on, t_off, sign):
  """One pulse (commanded direction = `sign`). Returns onset delay, STEADY velocity v_ss (mean rate
  over the plateau — noise-robust, the right metric for a rate command), the velocity-rise tau, and a
  settle/ramp flag. v_ss is signed in the command direction (negative ⇒ moved the wrong way)."""
  t = np.asarray(t); ang = np.asarray(ang); rate = np.asarray(rate)
  drive = (t >= t_on) & (t < t_off)
  if drive.sum() < 8: return None
  td, rd = t[drive], rate[drive] * sign            # command-direction positive
  n = len(rd)
  peak = float(np.max(np.abs(rd)))
  v_ss = float(np.mean(rd[-max(3, int(0.4 * n)):]))   # steady velocity over the last 40%
  late = float(np.mean(np.abs(rd[-max(2, n // 5):])))
  settle = late < 0.25 * peak                         # rate decayed → position servo; else rate cmd

  floor = max(0.15 * peak, 1e-3)
  moved = np.where(np.abs(rd) > floor)[0]                 # magnitude → catches wrong-direction motion too
  onset_delay = float(td[moved[0]] - t_on) if len(moved) else float("nan")
  tau = None
  if not settle and abs(v_ss) > floor and len(moved):
    cross = np.where(np.abs(rd) >= 0.632 * abs(v_ss))[0]
    cross = cross[cross >= moved[0]]
    if len(cross): tau = float(td[cross[0]] - td[moved[0]])
  return {"onset_delay": onset_delay, "v_ss": v_ss, "peak": peak, "settle": settle, "tau": tau}

def aggregate(results):
  """Rate-command system-ID from the STEADY velocity. For a rate command v_ss ≈ K_JOYSTICK·amp until
  it clamps at OMEGA_MAX, so OMEGA_MAX = max(v_ss); the unsaturated points (v_ss < 0.85·max) give
  K_JOYSTICK as the slope of a through-origin line fit (robust to the per-pulse noise that wrecked the
  old peak metric); TAU is the velocity-rise constant from those points. DELTA_INPUT uses only pulses
  that clearly moved (v_ss above noise), so sub-noise small amps don't corrupt the onset."""
  res = [r for r in results if r["onset_delay"] == r["onset_delay"]]  # drop NaN-onset (no motion)
  if not res:
    return {"n": 0}
  amps = np.array([r["amp"] for r in res]); vss = np.array([r["v_ss"] for r in res])
  vmag = np.abs(vss)                                  # gain magnitude (sign handled via n_wrongdir)
  omega_max = float(np.max(vmag))
  unsat = vmag < 0.85 * omega_max
  reached = bool((~unsat).sum() >= 2)                 # ≥2 points plateaued → saturation observed
  if unsat.sum() >= 2 and np.sum(amps[unsat] ** 2) > 0:
    k = float(np.sum(vmag[unsat] * amps[unsat]) / np.sum(amps[unsat] ** 2))   # LS slope through origin
  else:
    k = float(np.median(vmag / np.maximum(amps, 1e-9)))
  taus = [r["tau"] for r, u in zip(res, unsat) if u and r["tau"] is not None]
  clear = [r for r in res if abs(r["v_ss"]) > 0.3]    # onset only from pulses well above noise
  onsets = np.array([r["onset_delay"] for r in (clear or res)])
  return {
    "delta_input": (float(np.median(onsets)), float(np.std(onsets))),
    "omega_max": omega_max, "omega_reached": reached, "k_joystick": k,
    "tau": (float(np.median(taus)), float(np.std(taus))) if taus else None,
    "n_settle": sum(r["settle"] for r in res), "n": len(res),
    "n_saturated": int((~unsat).sum()), "n_wrongdir": int(np.sum(vss < 0)),
  }

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--axis", choices=["yaw", "pitch"], default="pitch")
  ap.add_argument("--amps", default="0.05,0.1,0.2,0.4,0.7",
                  help="comma list of pulse amplitudes. Small→K_JOYSTICK/TAU (unsaturated), large→OMEGA_MAX (saturated)")
  ap.add_argument("--reps", type=int, default=2, help="+/- pulse pairs per amplitude")
  ap.add_argument("--hold", type=float, default=0.4, help="pulse hold time (s)")
  ap.add_argument("--gap", type=float, default=0.6, help="rest between pulses (s)")
  ap.add_argument("--addr", default="127.0.0.1")
  ap.add_argument("--out", default=str(Path(__file__).parent.parent.parent / "weights" / "gimbal_sysid.npz"))
  args = ap.parse_args()

  amps = [float(a) for a in args.amps.split(",")]
  pub = messaging.Pub(["aim_error"])
  sub = messaging.Sub(["gimbal_state"], conflate=False, addr=args.addr)
  ax_key = "yaw_gi" if args.axis == "yaw" else "pitch_gi"
  rate_key = "yaw_rate_gi" if args.axis == "yaw" else "pitch_rate_gi"
  xy = ("x" if args.axis == "yaw" else "y")

  pulses = [(a, s) for a in amps for _ in range(args.reps) for s in (1.0, -1.0)]
  print(f"system-ID on {args.axis}: {len(pulses)} pulses over amps {amps} rad — gimbal WILL move. "
        "decisiond must be stopped. Ctrl-C to abort (returns to zero).")
  trace_t, trace_cmd, trace_ang, trace_rate = [], [], [], []
  windows = []  # (t_on, t_off, signed_amp)

  def send(v): pub.send("aim_error", {"x": v if xy == "x" else 0.0, "y": v if xy == "y" else 0.0})
  def drain():
    # drain ONLY — do NOT also call sub.update(); update() pulls one msg off the same socket that
    # drain() then can't return, and at the gimbal_state rate that eats almost every sample.
    for d in sub.drain("gimbal_state"):
      trace_ang.append(d[ax_key]); trace_rate.append(d[rate_key]); trace_t.append(d["t_stamp"])
      trace_cmd.append(cmd[0])

  cmd = [0.0]
  try:
    for amp, sign in pulses:
      cmd[0] = sign * amp
      t_on = time.monotonic(); windows.append([t_on, None, sign * amp])
      while time.monotonic() - t_on < args.hold: send(cmd[0]); drain(); time.sleep(0.002)
      windows[-1][1] = time.monotonic()
      cmd[0] = 0.0
      t_rest = time.monotonic()
      while time.monotonic() - t_rest < args.gap: send(0.0); drain(); time.sleep(0.002)
  finally:
    for _ in range(10): send(0.0); time.sleep(0.005)

  t = np.array(trace_t); ang = np.array(trace_ang); rate = np.array(trace_rate)
  if len(t) < 20:
    raise SystemExit("too few gimbal_state samples — is commsd publishing gimbal_state?")

  results = []
  for w in windows:
    r = _analyze_step(t, ang, rate, w[0], w[1], math.copysign(1.0, w[2]))
    if r: r["amp"] = abs(w[2]); results.append(r)
  if not results:
    raise SystemExit("no clean pulses detected — increase amplitudes or check the aim_error sign")
  agg = aggregate(results)
  if agg["n"] == 0:
    raise SystemExit("no gimbal motion detected in any pulse — check the aim_error sign/wiring and that the gimbal is enabled")

  print(f"\n{agg['n']} pulses analyzed ({agg['n_saturated']} slew-saturated)")
  if agg["n_wrongdir"] > agg["n"] // 2:
    print("WARNING: most pulses moved AGAINST the command — flip AIM_*_SIGN in commsd (wrong actuation sign)")
  di, dis = agg["delta_input"]
  print(f"DELTA_INPUT (onset delay):    {di*1000:.1f} ± {dis*1000:.1f} ms"
        f"   (slightly inflated by ~gimbal_state sampling latency)")
  if agg["omega_reached"]:
    print(f"GIMBAL_OMEGA_MAX (slew ceiling): {agg['omega_max']:.2f} rad/s")
  else:
    print(f"GIMBAL_OMEGA_MAX: ≥ {agg['omega_max']:.2f} rad/s (LOWER BOUND — never saturated; "
          f"add larger --amps until steady velocity stops growing with amplitude)")
  if agg["tau"]:
    tau, taus = agg["tau"]
    print(f"GIMBAL_TAU (velocity-rise const): {tau*1000:.1f} ± {taus*1000:.1f} ms  (from unsaturated pulses)")
  else:
    print("GIMBAL_TAU: n/a — no unsaturated pulse cleanly rose; adjust --amps")
  print(f"K_JOYSTICK (rad/s per aim_error unit): {agg['k_joystick']:.2f}")
  print(f"response: {agg['n_settle']}/{agg['n']} pulses SETTLE → "
        + ("POSITION servo (aim_error is a position error — decisiond is correct)"
           if agg["n_settle"] > agg["n"]//2
           else "RAMP → RATE command (decisiond's velocity feedforward is the right design for this)"))

  # Starting kp = the stability ceiling 1/(2·(dead time + rise lag)); tune UP to ringing, back off ~30%.
  di = agg["delta_input"][0]
  lag = di + (agg["tau"][0] if agg["tau"] else 0.0)
  kp = round(1.0 / (2.0 * lag), 1) if lag > 0 else 3.0
  ax = args.axis
  print(f"\n# paste into cv/autoaim/common.py (tune kp up to the ringing limit, back off ~30%):")
  print(f'  AIM_GAINS["{ax}"]        = dict(k_joystick={agg["k_joystick"]:.2f}, kp={kp}, ki=0.0, kd=0.0)')
  tau_s = f'{agg["tau"][0]:.3f}' if agg["tau"] else "<n/a>"
  print(f'  GIMBAL_TAU["{ax}"]       = {tau_s}')
  print(f'  GIMBAL_OMEGA_MAX["{ax}"] = {agg["omega_max"]:.2f}'
        + ("" if agg["omega_reached"] else "   # LOWER BOUND — rerun with larger --amps to saturate"))

  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.out, t=t, cmd=np.array(trace_cmd), ang=ang, rate=rate,
           windows=np.array(windows, dtype=float), axis=args.axis)
  print(f"saved traces → {args.out}")

if __name__ == "__main__":
  main()
