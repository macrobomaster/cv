"""Gimbal system-ID: command safe aim_error pulses and fit the response to recover
DELTA_INPUT (command→motion onset), GIMBAL_TAU (first-order settle), GIMBAL_OMEGA_MAX (peak slew).

Also classifies the response as SETTLE (firmware servos aim_error to a position) vs RAMP (firmware
treats it as a rate command) — the open question for decisiond's control law.

Run with decisiond STOPPED (it also publishes aim_error). The gimbal WILL move — keep amplitude
small and the area clear:
    python -m cv.tools.calib_gimbal --axis pitch --amp 0.05 --hold 0.4 --steps 6

Prints estimates and saves weights/gimbal_sysid.npz (raw traces for inspection in Rerun).
"""
import argparse, time
from pathlib import Path

import numpy as np

from ..system.core import messaging

def _analyze_step(t, ang, rate, t_on, t_off, rate_floor=None):
  """One pulse: command steps at t_on, releases at t_off. Returns onset delay, peak rate,
  classification (settle/ramp), and tau (if settle)."""
  t = np.asarray(t); ang = np.asarray(ang); rate = np.asarray(rate)
  drive = (t >= t_on) & (t < t_off)
  if drive.sum() < 4: return None
  td, ad, rd = t[drive], ang[drive], rate[drive]
  peak = float(np.max(np.abs(rd)))
  floor = rate_floor if rate_floor is not None else max(0.05 * peak, 1e-3)

  moved = np.where(np.abs(rd) > floor)[0]
  if len(moved) == 0: return None
  onset_delay = float(td[moved[0]] - t_on)

  a0 = float(ad[0])
  ad_rel = ad - a0
  total = float(ad_rel[-1])
  # SETTLE: rate decays to near zero before release (position servo).
  # RAMP: rate stays high at release (velocity command → angle keeps moving).
  tail_rate = float(np.mean(np.abs(rd[-max(2, len(rd)//5):])))
  settle = tail_rate < 0.25 * peak

  tau = None
  if settle and abs(total) > 1e-3:
    # time to 63% of final from onset → first-order tau
    target = a0 + 0.632 * total
    cross = np.where((ad_rel / total) >= 0.632)[0] if total > 0 else np.where((ad_rel / total) >= 0.632)[0]
    if len(cross):
      tau = float(td[cross[0]] - (t_on + onset_delay))
  return {"onset_delay": onset_delay, "peak_rate": peak, "settle": settle,
          "tau": tau, "total_move": total, "tail_rate": tail_rate}

def aggregate(results):
  """Separate TAU from OMEGA_MAX. For an unsaturated first-order pulse peak_rate ≈ amp/TAU
  (grows linearly with amp); once slew-limited it clamps at OMEGA_MAX. So the peak/amp ratio is
  flat in the linear region and DROPS where it saturates — that drop is how we know OMEGA_MAX was
  actually reached (vs just the biggest unsaturated pulse). TAU comes from the unsaturated pulses."""
  peaks = np.array([r["peak_rate"] for r in results])
  amps = np.array([r["amp"] for r in results])
  onsets = np.array([r["onset_delay"] for r in results])
  ratio = peaks / np.maximum(amps, 1e-9)                       # ≈ 1/TAU until saturation
  ref = float(np.median(ratio[amps <= np.median(amps)]))       # ratio in the small-amp (linear) half
  saturated = ratio < 0.7 * ref                                # ratio fell off → slew-limited
  reached = bool(saturated.any())
  unsat_taus = [r["tau"] for r, s in zip(results, saturated) if not s and r["tau"] is not None]
  return {
    "delta_input": (float(onsets.mean()), float(onsets.std())),
    "omega_max": float(peaks.max()), "omega_reached": reached,
    "k_joystick": ref,   # rad/s per aim_error unit (linear-region slope) → decisiond K_JOYSTICK
    "tau": (float(np.mean(unsat_taus)), float(np.std(unsat_taus))) if unsat_taus else None,
    "n_settle": sum(r["settle"] for r in results), "n": len(results), "n_saturated": int(saturated.sum()),
  }

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--axis", choices=["yaw", "pitch"], default="pitch")
  ap.add_argument("--amps", default="0.02,0.05,0.15,0.30",
                  help="comma list of pulse amplitudes (rad). Small→TAU (unsaturated), large→OMEGA_MAX (saturated)")
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
    r = _analyze_step(t, ang, rate, w[0], w[1])
    if r: r["amp"] = abs(w[2]); results.append(r)
  if not results:
    raise SystemExit("no clean pulses detected — increase amplitudes or check the aim_error sign")
  agg = aggregate(results)

  print(f"\n{agg['n']} pulses analyzed ({agg['n_saturated']} slew-saturated)")
  di, dis = agg["delta_input"]
  print(f"DELTA_INPUT (onset delay):    {di*1000:.1f} ± {dis*1000:.1f} ms"
        f"   (slightly inflated by ~gimbal_state sampling latency)")
  if agg["omega_reached"]:
    print(f"GIMBAL_OMEGA_MAX (peak slew): {agg['omega_max']:.2f} rad/s")
  else:
    print(f"GIMBAL_OMEGA_MAX: ≥ {agg['omega_max']:.2f} rad/s (LOWER BOUND — never saturated; "
          f"add larger --amps until peak rate stops growing with amplitude)")
  if agg["tau"]:
    tau, taus = agg["tau"]
    print(f"GIMBAL_TAU (first-order settle): {tau*1000:.1f} ± {taus*1000:.1f} ms  (from unsaturated pulses)")
  else:
    print("GIMBAL_TAU: n/a — no unsaturated settle; add smaller --amps")
  print(f"K_JOYSTICK (rad/s per aim_error unit): {agg['k_joystick']:.2f}  → decisiond K_JOYSTICK / AIM_KP")
  print(f"response: {agg['n_settle']}/{agg['n']} pulses SETTLE → "
        + ("POSITION servo (aim_error is a position error — decisiond is correct)"
           if agg["n_settle"] > agg["n"]//2
           else "RAMP → RATE command (decisiond should send absolute target, not position error!)"))

  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.out, t=t, cmd=np.array(trace_cmd), ang=ang, rate=rate,
           windows=np.array(windows, dtype=float), axis=args.axis)
  print(f"saved traces → {args.out}")

if __name__ == "__main__":
  main()
