"""Top-down field map editor — define the nav map (walls + goal points) for navd.

Renders the field with the surveyed AprilTags, lets you paint rectangular walls and
place goal points, previews the robot-radius inflation the planner actually sees, draws
the planner's ROUTE between consecutive goals (green; red straight = no path) so you can
sanity-check where it would drive, and exports a NAV_MAP JSON that navd loads (`NAV_MAP` env):

  {bounds:[x0,y0,x1,y1], res, robot_radius, walls:[{rect:[x0,y0,x1,y1]}], goals:[{x,y,label}]}

The state machine reads the saved goals (rename labels → "home"/"center"/… in the JSON)
and injects them as nav_goal at runtime; navd plans around the walls.

  python -m cv.tools.path_editor [out.json]

Mode:  [w] walls   ·   [g] goals          (current mode shown in the HUD)
Walls: left-click a corner, left-click the opposite corner → rectangle
Goals: left-click to drop a point (auto-labeled g0,g1,…)
Both:  right-click remove nearest · [z] undo · [x] clear mode · [s] save · [l] load · [q] quit
"""
import sys, json, math, argparse

import cv2
import numpy as np

from ..slam import common
from ..nav.occupancy import OccupancyGrid, ROBOT_RADIUS
from ..nav import planner

CANVAS_MAX = 1100        # px, longest canvas side
PAD = 55                 # px, border around the field
NORMAL_LEN = 0.6         # m, drawn length of each tag's face-normal arrow
RES = 0.10               # m, grid resolution for the inflation preview / export
SNAP_PX = 14             # px, right-click removes a wall/goal within this radius

# Colors (BGR)
C_BG, C_GRID = (48, 45, 45), (72, 72, 72)
C_TAG, C_NORMAL = (220, 180, 0), (0, 230, 230)
C_WALL, C_INFLATE = (40, 40, 40), (150, 150, 150)
C_GOAL, C_PEND, C_RUBBER = (0, 165, 255), (0, 255, 0), (0, 210, 0)
C_PATH, C_BLOCKED = (60, 220, 60), (0, 0, 230)   # planned route between goals; red = no path
C_TEXT = (235, 235, 235)

class View:
  def __init__(self):
    self.x0, self.y0, self.x1, self.y1 = common.FIELD_BOUNDS   # 12 m × 8 m play area
    self.scale = (CANVAS_MAX - 2 * PAD) / max(self.x1 - self.x0, self.y1 - self.y0)
    self.W = int((self.x1 - self.x0) * self.scale) + 2 * PAD
    self.H = int((self.y1 - self.y0) * self.scale) + 2 * PAD

  def to_px(self, wx, wy):
    return (int((wx - self.x0) * self.scale) + PAD, int((self.y1 - wy) * self.scale) + PAD)
  def to_world(self, px, py):
    return ((px - PAD) / self.scale + self.x0, self.y1 - (py - PAD) / self.scale)

def build_grid(view, walls):
  g = OccupancyGrid(view.x0, view.y0, view.x1, view.y1, RES)
  for x0, y0, x1, y1 in walls: g.add_rect(x0, y0, x1, y1)
  return g, g.inflated(ROBOT_RADIUS)

def plan_segments(inflated, goals):
  """The planner's route between each consecutive goal pair (exactly what navd would drive),
  or None for a pair with no path. Recomputed only when walls/goals change (not per frame)."""
  return [planner.plan(inflated, (goals[i][0], goals[i][1]), (goals[i + 1][0], goals[i + 1][1]))
          for i in range(len(goals) - 1)]

# --- rendering (pure: returns a BGR image, so it's testable headless) ---
def render(view, state, grid, inflated, planned):
  img = np.full((view.H, view.W, 3), C_BG, np.uint8)

  # inflation halo then raw walls (from the grids → exactly what the planner sees)
  for iy in range(grid.ny):
    for ix in range(grid.nx):
      if not inflated.occ[iy, ix]: continue
      p0 = view.to_px(*grid.cell_to_world(ix, iy))
      d = int(math.ceil(RES * view.scale))
      cv2.rectangle(img, (p0[0] - d // 2 - 1, p0[1] - d // 2 - 1), (p0[0] + d, p0[1] + d),
                    C_WALL if grid.occ[iy, ix] else C_INFLATE, -1)

  for wx in range(math.ceil(view.x0), math.floor(view.x1) + 1):
    x, _ = view.to_px(wx, view.y0)
    cv2.line(img, (x, PAD), (x, view.H - PAD), C_GRID, 1)
    cv2.putText(img, str(wx), (x - 4, view.H - PAD + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GRID, 1)
  for wy in range(math.ceil(view.y0), math.floor(view.y1) + 1):
    _, y = view.to_px(view.x0, wy)
    cv2.line(img, (PAD, y), (view.W - PAD, y), C_GRID, 1)
    cv2.putText(img, str(wy), (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GRID, 1)

  for tid, (R_wt, t_wt) in sorted(common.TAG_FIELD_MAP.items()):
    tx, ty = float(t_wt[0]), float(t_wt[1])
    p = view.to_px(tx, ty)
    tip = view.to_px(tx + NORMAL_LEN * float(R_wt[0, 2]), ty + NORMAL_LEN * float(R_wt[1, 2]))
    cv2.arrowedLine(img, p, tip, C_NORMAL, 2, tipLength=0.35)
    cv2.rectangle(img, (p[0] - 6, p[1] - 6), (p[0] + 6, p[1] + 6), C_TAG, -1)
    cv2.putText(img, str(tid), (p[0] + 9, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1)

  # planned routes between consecutive goals (what navd would drive) — red straight = blocked
  for i, seg in enumerate(planned):
    a, b = state["goals"][i], state["goals"][i + 1]
    if seg is None:
      cv2.line(img, view.to_px(a[0], a[1]), view.to_px(b[0], b[1]), C_BLOCKED, 1)
    elif len(seg) >= 2:
      cv2.polylines(img, [np.array([view.to_px(x, y) for x, y in seg], np.int32)], False, C_PATH, 2)

  for i, (gx, gy, lbl) in enumerate(state["goals"]):
    p = view.to_px(gx, gy)
    cv2.circle(img, p, 6, C_GOAL, -1)
    txt = f"{lbl or f'g{i}'} ({gx:.2f}, {gy:.2f})"
    tx = p[0] - 8 - 9 * len(txt) if gx > view.x1 - 3.0 else p[0] + 8   # left-anchor near the right edge
    cv2.putText(img, txt, (tx, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1)

  # pending wall corner + rubber-band to the mouse
  if state["mode"] == "walls" and state["pend"] is not None and state["mouse"] is not None:
    cv2.rectangle(img, view.to_px(*state["pend"]), view.to_px(*state["mouse"]), C_RUBBER, 1)
    cv2.circle(img, view.to_px(*state["pend"]), 4, C_PEND, -1)

  mx, my = state["mouse"] if state["mouse"] is not None else (0.0, 0.0)
  hud = [f"cursor: ({mx:.2f}, {my:.2f}) m      mode: {state['mode'].upper()}   "
         f"walls: {len(state['walls'])}   goals: {len(state['goals'])}   out: {state['out']}",
         "[w]alls [g]oals  |  L-click add  R-click remove  |  [z]undo [x]clear [s]ave [l]oad [q]uit"]
  for i, line in enumerate(hud):
    cv2.putText(img, line, (12, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1)
  return img

# --- file io ---
def save(view, state):
  data = {"bounds": [view.x0, view.y0, view.x1, view.y1], "res": RES, "robot_radius": ROBOT_RADIUS,
          "walls": [{"rect": [float(v) for v in w]} for w in state["walls"]],
          "goals": [{"x": float(x), "y": float(y), "label": lbl or f"g{i}"}
                    for i, (x, y, lbl) in enumerate(state["goals"])]}
  with open(state["out"], "w") as f: json.dump(data, f, indent=2)
  print(f"saved {len(data['walls'])} walls, {len(data['goals'])} goals → {state['out']}")

def load(state):
  with open(state["out"]) as f: data = json.load(f)
  state["walls"] = [tuple(w["rect"]) for w in data.get("walls", [])]
  state["goals"] = [(g["x"], g["y"], g.get("label", "")) for g in data.get("goals", [])]
  print(f"loaded {len(state['walls'])} walls, {len(state['goals'])} goals from {state['out']}")

def _nearest(items, pt, key):
  if not items: return None
  d = [math.hypot(key(it)[0] - pt[0], key(it)[1] - pt[1]) for it in items]
  j = int(np.argmin(d))
  return j, d[j]

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out", nargs="?", default="nav_map.json", help="output NAV_MAP JSON")
  args = ap.parse_args()
  view = View()
  state = {"walls": [], "goals": [], "mode": "walls", "pend": None, "mouse": None, "out": args.out}
  try: load(state)
  except (FileNotFoundError, json.JSONDecodeError): pass

  def on_mouse(event, px, py, flags, _):
    w = view.to_world(px, py)
    state["mouse"] = w
    if event == cv2.EVENT_LBUTTONDOWN:
      if state["mode"] == "walls":
        if state["pend"] is None: state["pend"] = w
        else: state["walls"].append((state["pend"][0], state["pend"][1], w[0], w[1])); state["pend"] = None
      else:
        state["goals"].append((w[0], w[1], ""))
        print(f"goal g{len(state['goals']) - 1} = [{w[0]:.2f}, {w[1]:.2f}]")   # paste into stated
    elif event == cv2.EVENT_RBUTTONDOWN:
      if state["mode"] == "walls":
        n = _nearest(state["walls"], w, lambda r: ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2))
        if n is not None: state["walls"].pop(n[0])
      else:
        n = _nearest(state["goals"], w, lambda g: (g[0], g[1]))
        if n is not None and n[1] * view.scale <= SNAP_PX * 3: state["goals"].pop(n[0])

  win = "map_editor"
  cv2.namedWindow(win)
  cv2.setMouseCallback(win, on_mouse)
  grid = inflated = planned = None
  dirty = None
  while True:
    key = (len(state["walls"]), len(state["goals"]))   # rebuild grid + replan only on a change
    if key != dirty:
      grid, inflated = build_grid(view, state["walls"])
      planned = plan_segments(inflated, state["goals"])
      dirty = key
    cv2.imshow(win, render(view, state, grid, inflated, planned))
    k = cv2.waitKey(20) & 0xFF
    if k in (ord("q"), 27): break
    elif k == ord("w"): state["mode"], state["pend"] = "walls", None
    elif k == ord("g"): state["mode"], state["pend"] = "goals", None
    elif k == ord("z"): state[state["mode"]] and state[state["mode"]].pop()
    elif k == ord("x"): state[state["mode"]].clear()
    elif k == ord("s"): save(view, state)
    elif k == ord("l"):
      try: load(state)
      except (FileNotFoundError, json.JSONDecodeError) as e: print(f"load failed: {e}")
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
