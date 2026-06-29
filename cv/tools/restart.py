"""Request a supervisor-managed daemon restart.

  python -m cv.tools.restart autoaimd
  python -m cv.tools.restart --list
"""

import argparse, sys, time

from ..system.__main__ import PROCS
from ..system.core.keyvalue import kv_get, kv_put

SERVICES = tuple(p.name for p in PROCS)

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("service", nargs="?", help="service name to restart")
  ap.add_argument("--list", action="store_true", help="print restartable service names and exit")
  ap.add_argument("--wait", type=float, default=5.0, help="seconds to wait for supervisor acknowledgement (0 disables)")
  args = ap.parse_args()

  if args.list:
    print("\n".join(SERVICES))
    return 0

  if args.service is None:
    ap.error("service is required unless --list is used")
  if args.service not in SERVICES:
    ap.error(f"unknown service {args.service!r}; valid services: {', '.join(SERVICES)}")

  kv_put("restart", args.service, True)
  if args.wait <= 0:
    print(f"requested restart of {args.service}")
    return 0

  deadline = time.monotonic() + args.wait
  while time.monotonic() < deadline:
    if kv_get("restart", args.service) is not True:
      print(f"restarted {args.service}")
      return 0
    time.sleep(0.05)

  print(f"requested restart of {args.service}, but supervisor did not acknowledge within {args.wait:g}s", file=sys.stderr)
  return 1

if __name__ == "__main__":
  raise SystemExit(main())
