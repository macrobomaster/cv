"""Record all messaging topics to one file for offline calibration / debugging.

Subscribes non-conflated and drains every tick, so it captures each message the publishers emit
(not just the latest). Each record is length-prefixed CBOR: {topic, t (monotonic recv), data}.
The calib tools (calib_handeye, calib_gimbal) replay a bag offline.

  python -m cv.tools.bag out.bag              # record everything until Ctrl-C
  python -m cv.tools.bag out.bag --no-camera  # skip the big camera frames

Replay in code:  from cv.tools.bag import load_bag
"""
import time, struct, argparse

import cbor2

from ..system.core import messaging

ALL_TOPICS = [
  "autoaim", "plate", "gimbal_state", "aim_error", "aim_angle", "shoot", "chassis_velocity",
  "game_running", "team_color", "robot_type", "comms_rates", "slam_pose",
]
CAMERA_TOPICS = ["camera_feed", "camera_feed_raw"]

def load_bag(path:str) -> dict:
  """Return {topic: [(t_recv, data), ...]} in record order."""
  out: dict = {}
  with open(path, "rb") as f:
    while True:
      header = f.read(4)
      if len(header) < 4: break
      (n,) = struct.unpack("<I", header)
      rec = cbor2.loads(f.read(n))
      out.setdefault(rec["topic"], []).append((rec["t"], rec["data"]))
  return out

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("out", help="output .bag file")
  ap.add_argument("--addr", default="127.0.0.1")
  ap.add_argument("--no-camera", action="store_true", help="skip camera_feed/_raw (large frames)")
  args = ap.parse_args()

  topics = list(ALL_TOPICS) + ([] if args.no_camera else CAMERA_TOPICS)
  sub = messaging.Sub(topics, conflate=False, addr=args.addr)

  counts = {t: 0 for t in topics}
  t0 = time.monotonic()
  last_hb = t0
  print(f"recording {len(topics)} topics → {args.out}  (Ctrl-C to stop)")
  try:
    with open(args.out, "wb") as f:
      while True:
        # drain ONLY — sub.update() would pull one msg/topic off the socket that drain() then misses.
        time.sleep(0.002)
        now = time.monotonic()
        for topic in topics:
          for data in sub.drain(topic):
            blob = cbor2.dumps({"topic": topic, "t": now, "data": data})
            f.write(struct.pack("<I", len(blob)) + blob)
            counts[topic] += 1
        if now - last_hb > 1.0:
          total = sum(counts.values())
          print(f"\r{now-t0:6.1f}s  {total} msgs  " +
                "  ".join(f"{t}={c}" for t, c in counts.items() if c), end="", flush=True)
          last_hb = now
  except KeyboardInterrupt:
    pass
  dur = time.monotonic() - t0
  print(f"\nstopped after {dur:.1f}s")
  for t, c in counts.items():
    if c: print(f"  {t:18s} {c:7d}  ({c/max(dur,1e-3):.1f} Hz)")

if __name__ == "__main__":
  main()
