from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put

def run():
  pub = messaging.Pub(["aim_error"])
  sub = messaging.Sub(["autoaim"])

  while True:
    sub.update(10)

    autoaim = sub["autoaim"]
    if sub.updated["autoaim"] and autoaim is not None:
      cl = autoaim["cl"]
      clp = autoaim["clp"]
      if cl == 1 and clp > 0.6:
        x = autoaim["x"]
        y = autoaim["y"]
        pub.send("aim_error", {"x": x, "y": y})
