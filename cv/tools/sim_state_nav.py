"""Offline stated/navd simulator with a scrubbable Rerun recording.

This does not start the supervisor, open ZMQ sockets, touch serial, camera, or tinygrad.
It directly ticks stated's state machine and navd's pure planner/follower code with
synthetic messages, then writes a .rrd timeline:

  python -m cv.tools.sim_state_nav --out state_nav_sim.rrd
  rerun state_nav_sim.rrd

Map data comes from code: slam.common.FIELD_BOUNDS, FIELD_WALLS, TAG_FIELD_MAP.
NAV_MAP is intentionally not used here; navd's runtime default is the same code-defined
field_grid().
"""
import argparse, math, time
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from ..slam import common
from ..system.common.geometry import rotz, wrap_pi
from ..system.navd import navd
from ..system.stated.states import TEAM_GOALS, PLAY_STYLES, RETREAT_HP, make_state_machine

DEFAULT_OUT = "state_nav_sim.rrd"
DEFAULT_DURATION = 180.0
DEFAULT_DT = 0.05
TRAIL_LOG_DT = 0.2
ROBOT_RADIUS_VIS = 0.18
ZONE_HATCH_SPACING = 0.35
ZONE_HATCH_Z = 0.014

class SimCtx:
  def __init__(self):
    self.data = {}
    self.updated = {}
    self.now = 0.0
    self.match_elapsed = 0.0
    self.entered = False

  def __getitem__(self, service):
    return self.data.get(service)

  def set(self, service, value, updated=True):
    self.data[service] = value
    self.updated[service] = updated

  def begin_tick(self, now):
    self.now = now
    self.updated = {k: False for k in self.data}

class CapturePub:
  def __init__(self):
    self.messages = []

  def clear(self):
    self.messages.clear()

  def send(self, service, data):
    self.messages.append((service, data))

  def last(self, service):
    for s, data in reversed(self.messages):
      if s == service: return data
    return None

def parse_window(s):
  try:
    a, b = [float(x) for x in s.split(":", 1)]
  except ValueError as e:
    raise argparse.ArgumentTypeError("expected START:END") from e
  if b < a: raise argparse.ArgumentTypeError("window END must be >= START")
  return a, b

def parse_enemy(s):
  """x,y or x,y,r or x,y,r,t0,t1. Active times are sim seconds."""
  try:
    vals = [float(x) for x in s.split(",")]
  except ValueError as e:
    raise argparse.ArgumentTypeError("enemy must be comma-separated floats") from e
  if len(vals) == 2: return vals[0], vals[1], navd.ENEMY_RADIUS, 0.0, math.inf
  if len(vals) == 3: return vals[0], vals[1], vals[2], 0.0, math.inf
  if len(vals) == 5:
    if vals[4] < vals[3]: raise argparse.ArgumentTypeError("enemy t1 must be >= t0")
    return tuple(vals)
  raise argparse.ArgumentTypeError("expected x,y or x,y,r or x,y,r,t0,t1")

def parse_hp_loss(s):
  try:
    t, amount = [float(x) for x in s.split(":", 1)]
  except ValueError as e:
    raise argparse.ArgumentTypeError("expected TIME:AMOUNT") from e
  if t < 0.0: raise argparse.ArgumentTypeError("HP loss time must be >= 0")
  if amount < 0.0: raise argparse.ArgumentTypeError("HP loss amount must be >= 0")
  return t, amount

def in_windows(t, windows):
  return any(a <= t <= b for a, b in windows)

def default_start(team):
  x, y = TEAM_GOALS[team]["home"]
  cx, cy = TEAM_GOALS[team]["center"]
  return x, y, math.degrees(math.atan2(cy - y, cx - x))

def q_wb_from_heading(heading):
  Rz = rotz(-heading if common.YAW_FLIPPED else heading)
  q = Rotation.from_matrix(Rz @ common.R_CAM).as_quat()  # xyzw
  return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

def slam_pose(now, p_xy, v_w, heading):
  return {
    "t": now,
    "p_w": [float(p_xy[0]), float(p_xy[1]), 0.0],
    "v_w": [float(v_w[0]), float(v_w[1]), 0.0],
    "q_wb": q_wb_from_heading(heading),
    "cov_pos": np.diag([0.01, 0.01, 0.01]).astype(np.float32).flatten().tolist(),
    "n_tags": 1,
  }

def gimbal_state(now, yaw, pitch, yaw_rate, pitch_rate):
  return {
    "t_stamp": now,
    "yaw_gi": float(yaw), "pitch_gi": float(pitch),
    "yaw_rate_gi": float(yaw_rate), "pitch_rate_gi": float(pitch_rate),
  }

def step_axis(cur, target, max_rate, dt, wrap=False):
  err = wrap_pi(target - cur) if wrap else target - cur
  step = max(-max_rate * dt, min(max_rate * dt, err))
  nxt = cur + step
  if wrap: nxt = wrap_pi(nxt)
  return nxt, step / max(dt, 1e-9)

def state_chain(sm):
  out = [sm.current]
  cur = sm.current
  while hasattr(cur, "states"):
    idx = getattr(cur, "idx", 0)
    if idx >= len(cur.states) or not getattr(cur, "child_started", False): break
    cur = cur.states[idx]
    out.append(cur)
  return out

def current_tag_choice(chain, pose):
  for state in reversed(chain):
    sel = getattr(state, "_select_tag", None)
    if callable(sel):
      heading = state._heading_world(pose["q_wb"])
      if heading is None: return None
      score_heading = getattr(state, "_look_at_heading_w", None)
      if score_heading is None:
        motion_heading = getattr(state, "_motion_heading_world", None)
        score_heading = motion_heading(pose) if callable(motion_heading) else None
      best = sel(pose, heading, heading if score_heading is None else score_heading)
      if best is not None: return int(best[1]), tuple(float(x) for x in best[2])
  return None

def zone_hatches(corners, spacing=ZONE_HATCH_SPACING):
  """Diagonal hatch segments clipped to a simple polygon, avoiding Mesh3D alpha blending."""
  if len(corners) < 3: return []
  pts = [np.array([float(x), float(y)], np.float64) for x, y in corners]
  d = np.array([1.0, 1.0], np.float64) / math.sqrt(2.0)
  n = np.array([-d[1], d[0]], np.float64)
  proj = [float(p @ n) for p in pts]
  h = math.floor(min(proj) / spacing) * spacing + spacing * 0.5
  h_end = max(proj)
  out = []
  while h <= h_end:
    hits = []
    for i, p0 in enumerate(pts):
      p1 = pts[(i + 1) % len(pts)]
      s0, s1 = float(p0 @ n - h), float(p1 @ n - h)
      if abs(s0) < 1e-9 and abs(s1) < 1e-9:
        hits += [p0, p1]
      elif abs(s0) < 1e-9:
        hits.append(p0)
      elif s0 * s1 < 0.0:
        hits.append(p0 + (p1 - p0) * (-s0 / (s1 - s0)))
    uniq = []
    for p in hits:
      if not any(np.linalg.norm(p - q) < 1e-6 for q in uniq): uniq.append(p)
    uniq.sort(key=lambda p: float(p @ d))
    for a, b in zip(uniq[0::2], uniq[1::2]):
      if np.linalg.norm(b - a) > 1e-6: out.append([[float(a[0]), float(a[1]), ZONE_HATCH_Z], [float(b[0]), float(b[1]), ZONE_HATCH_Z]])
    h += spacing
  return out

def log_field_zones():
  for z in common.FIELD_ZONES:
    corners = common.zone_corners(z)
    pts = [[float(x), float(y), 0.012] for x, y in corners]
    loop = pts + [pts[0]]
    color = [int(c) for c in z["color"]]
    ent = "world/zones/" + z["name"].replace(" ", "_")
    hatches = zone_hatches(corners)
    if hatches: rr.log(ent + "/hatch", rr.LineStrips3D(hatches, colors=[color], radii=[0.01]), static=True)
    rr.log(ent + "/outline", rr.LineStrips3D([loop], colors=[color], radii=[0.02]), static=True)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    rr.log(ent + "/label", rr.Points3D([[cx, cy, 0.03]], colors=[color], radii=0.02, labels=[z["name"]]), static=True)

def log_static_scene(static_grid, robot_radius, team):
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  x0, y0, x1, y1 = common.FIELD_BOUNDS
  rr.log("world/field", rr.LineStrips3D([[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0]]],
          colors=[(110, 110, 110)], radii=[0.012]), static=True)
  log_field_zones()

  if common.FIELD_WALLS:
    rects = [tuple(r) for r in common.FIELD_WALLS]
    centers, half_sizes = [], []
    for r in rects:
      xlo, xhi = min(r[0], r[2]), max(r[0], r[2])
      ylo, yhi = min(r[1], r[3]), max(r[1], r[3])
      centers.append([(xlo + xhi) / 2, (ylo + yhi) / 2, 0.15])
      half_sizes.append([(xhi - xlo) / 2, (yhi - ylo) / 2, 0.15])
    rr.log("world/walls", rr.Boxes3D(centers=centers, half_sizes=half_sizes, colors=[(70, 70, 70)]),
           static=True)
    inflated = static_grid.inflated(robot_radius)
    iy, ix = np.nonzero(inflated.occ & ~static_grid.occ)
    if len(ix):
      pts = []
      for y, x in zip(iy, ix):
        wx = static_grid.x0 + (x + 0.5) * static_grid.res
        wy = static_grid.y0 + (y + 0.5) * static_grid.res
        pts.append([wx, wy, 0.01])
      rr.log("world/inflated_walls", rr.Points3D(pts, colors=[(120, 120, 120, 80)], radii=static_grid.res * 0.45),
             static=True)

  if common.TAG_FIELD_MAP:
    origins, vectors, labels = [], [], []
    for tid, (R_wt, t_wt) in sorted(common.TAG_FIELD_MAP.items()):
      origins.append([float(t_wt[0]), float(t_wt[1]), float(t_wt[2])])
      vectors.append((R_wt[:, 2] * 0.45).astype(float).tolist())
      labels.append(str(tid))
    rr.log("world/tags", rr.Points3D(origins, colors=[(220, 180, 0)], radii=0.045, labels=labels),
           static=True)
    rr.log("world/tag_normals", rr.Arrows3D(origins=origins, vectors=vectors, colors=[(0, 220, 220)]),
           static=True)

  goals = []
  labels = []
  colors = []
  for t, gs in TEAM_GOALS.items():
    for name, (gx, gy) in gs.items():
      goals.append([gx, gy, 0.03])
      labels.append(f"{t}_{name}")
      colors.append((80, 180, 255) if t == team else (130, 130, 130))
  rr.log("world/team_goals", rr.Points3D(goals, colors=colors, radii=0.07, labels=labels), static=True)
  rr.log("scalars/state_flags", rr.SeriesLines(names=["game", "autoaim", "spinning", "has_goal"]), static=True)
  rr.log("scalars/chassis_velocity", rr.SeriesLines(names=["x_gimbal", "y_gimbal", "speed_world"]), static=True)
  rr.log("scalars/robot_hp", rr.SeriesLines(names=["hp"]), static=True)
  rr.log("scalars/hp_loss", rr.SeriesLines(names=["loss"]), static=True)
  rr.log("scalars/state_change", rr.SeriesLines(names=["change"]), static=True)
  rr.log("state_changes/markers", rr.SeriesLines(names=["change"]), static=True)

def log_state_change(sim_t):
  rr.set_time("sim_time", duration=sim_t)
  rr.log("state_changes/markers", rr.Scalars(1.0))

def log_hp_loss(sim_t):
  rr.set_time("sim_time", duration=sim_t)
  rr.log("hp_loss/markers", rr.Scalars(1.0))

def log_dynamic(sim_t, p_xy, v_w, heading, trail, state_name, state_id, game_running, autoaim_valid, spinning,
                goal_xy, path, robots, selected_tag, state_setpoint, chassis_cmd, lookahead, state_changed, robot_hp,
                hp_loss_event):
  rr.set_time("sim_time", duration=sim_t)
  pos = [float(p_xy[0]), float(p_xy[1]), 0.04]
  speed = float(np.linalg.norm(v_w))
  label = f"{state_name}  v={speed:.2f}m/s"
  rr.log("world/robot", rr.Points3D([pos], colors=[(60, 180, 255)], radii=ROBOT_RADIUS_VIS, labels=[label]))
  heading_vec = [math.cos(heading) * 0.55, math.sin(heading) * 0.55, 0]
  rr.log("world/robot_heading", rr.Arrows3D(origins=[pos], vectors=[heading_vec], colors=[(60, 180, 255)]))
  if len(trail) >= 2:
    rr.log("world/trail", rr.LineStrips3D([[p.tolist() for p in trail]], colors=[(60, 180, 255)], radii=[0.01]))

  if goal_xy is None:
    rr.log("world/nav_goal", rr.Clear(recursive=True))
  else:
    rr.log("world/nav_goal", rr.Points3D([[goal_xy[0], goal_xy[1], 0.05]], colors=[(255, 0, 255)], radii=0.09))
  rr.log("world/nav_path", rr.LineStrips3D([[[x, y, 0.025] for x, y in path]], colors=[(0, 220, 0)], radii=[0.02])
         if len(path) >= 2 else rr.Clear(recursive=True))
  rr.log("world/lookahead", rr.Points3D([[lookahead[0], lookahead[1], 0.04]], colors=[(0, 255, 140)], radii=0.06)
         if lookahead is not None else rr.Clear(recursive=True))
  if robots:
    rr.log("world/dynamic_obstacles", rr.Points3D([[x, y, 0.08] for x, y, r in robots], colors=[(255, 70, 70)],
           radii=[r for x, y, r in robots]))
  else:
    rr.log("world/dynamic_obstacles", rr.Clear(recursive=True))

  if selected_tag is None:
    rr.log("world/selected_tag", rr.Clear(recursive=True))
  else:
    tid, tag_pos = selected_tag
    rr.log("world/selected_tag", rr.Points3D([[tag_pos[0], tag_pos[1], tag_pos[2]]], colors=[(0, 255, 255)],
           radii=0.08, labels=[f"look tag {tid}"]))
    rr.log("world/selected_tag/los",
           rr.LineStrips3D([[[p_xy[0], p_xy[1], 0.08], [tag_pos[0], tag_pos[1], tag_pos[2]]]],
                           colors=[(0, 255, 255)], radii=[0.012]))

  if state_setpoint is None:
    rr.log("world/state_setpoint", rr.Clear(recursive=True))
  else:
    yaw = state_setpoint["yaw"]
    rr.log("world/state_setpoint", rr.Arrows3D(origins=[pos], vectors=[[math.cos(yaw) * 0.8, math.sin(yaw) * 0.8, 0]],
                                               colors=[(255, 230, 0)]))

  rr.log("scalars/state_id", rr.Scalars(float(state_id)))
  rr.log("scalars/selected_tag", rr.Scalars(-1 if selected_tag is None else selected_tag[0]))
  rr.log("scalars/state_flags", rr.Scalars([int(game_running), int(autoaim_valid), int(spinning), int(goal_xy is not None)]))
  rr.log("scalars/chassis_velocity", rr.Scalars([float(chassis_cmd[0]), float(chassis_cmd[1]), speed]))
  rr.log("scalars/robot_hp", rr.Scalars(float(robot_hp)))
  rr.log("scalars/hp_loss", rr.Scalars(float(hp_loss_event)))
  rr.log("scalars/state_change", rr.Scalars(float(state_changed)))
  rr.log("scalars/gimbal_yaw_deg", rr.Scalars(math.degrees(heading)))

def run(args):
  start = args.start if args.start is not None else default_start(args.team)
  p_xy = np.array(start[:2], np.float64)
  gimbal_yaw = math.radians(start[2])
  gimbal_pitch = 0.0
  gimbal_yaw_rate = gimbal_pitch_rate = 0.0
  v_w = np.zeros(2, np.float64)

  static_grid, robot_radius = navd.field_grid()
  sm = make_state_machine(args.team, args.style)
  state_ids = {s.name: i for i, s in enumerate(sm.states)}
  ctx, pub = SimCtx(), CapturePub()
  t0 = time.monotonic()

  injected = last_goal = pursuit = None
  inj_label = None
  last_goal_t = -math.inf
  last_plan = -math.inf
  v_prev = 0.0
  trail = []
  prev_state_name = None
  prev_robot_hp = None
  hp_losses = sorted(args.hp_loss)
  trail_log_every = max(1, int(round(TRAIL_LOG_DT / args.dt)))
  gimbal_max_rate = math.radians(args.gimbal_rate_deg)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  rr.init("state_nav_sim", spawn=args.spawn)
  rr.save(str(out))
  log_static_scene(static_grid, robot_radius, args.team)

  steps = int(math.ceil(args.duration / args.dt)) + 1
  for k in range(steps):
    sim_t = k * args.dt
    now = t0 + sim_t
    game_running = args.game_start <= sim_t and (args.game_stop < 0 or sim_t <= args.game_stop)
    ctx.match_elapsed = sim_t - args.game_start if game_running else 0.0
    autoaim_valid = game_running and in_windows(sim_t, args.autoaim_window)
    robot_hp = max(0.0, args.hp_start - sum(amount for t_loss, amount in hp_losses if sim_t >= t_loss))
    hp_loss_event = prev_robot_hp is not None and robot_hp < prev_robot_hp
    prev_robot_hp = robot_hp
    pose = slam_pose(now, p_xy, v_w, gimbal_yaw)
    gs = gimbal_state(now, gimbal_yaw, gimbal_pitch, gimbal_yaw_rate, gimbal_pitch_rate)

    ctx.begin_tick(now)
    ctx.set("game_running", game_running, True)
    ctx.set("team_color", args.team, True)
    ctx.set("play_style", {"style": args.style}, True)
    ctx.set("autoaim", {
      "valid": autoaim_valid, "detected": autoaim_valid,
      "confidence": 1.0 if autoaim_valid else 0.0,
    }, True)
    ctx.set("gimbal_state", gs, True)
    ctx.set("slam_pose", pose, True)
    ctx.set("robot_hp", robot_hp, True)
    pub.clear()
    sm.tick(ctx, pub)

    nav_goal = pub.last("nav_goal")
    if nav_goal is not None:
      injected = np.array([float(nav_goal["x"]), float(nav_goal["y"])], np.float64)
      inj_label = nav_goal.get("label", "goal")
      last_goal_t = now
    if now - last_goal_t > navd.NAV_GOAL_TIMEOUT:
      injected = last_goal = pursuit = None
      inj_label = None
    goal_xy = injected

    robots = [(x, y, r) for x, y, r, t_start, t_end in args.enemy if t_start <= sim_t <= t_end]
    need_plan = (goal_xy is not None and (last_goal is None or pursuit is None or not np.allclose(goal_xy, last_goal)
                                          or now - last_plan > navd.PLAN_DT))
    if need_plan:
      path = navd.plan_path(static_grid, robot_radius, p_xy, goal_xy, robots)
      pursuit = navd.PurePursuit(path) if path is not None and len(path) >= 2 else None
      last_goal = np.asarray(goal_xy, np.float64).copy()
      last_plan = now

    fwd = navd.gimbal_heading(pose["q_wb"])
    chassis_cmd = (0.0, 0.0)
    lookahead = None
    path = pursuit.P.tolist() if pursuit is not None else []
    navigating = goal_xy is not None and pursuit is not None and fwd is not None and not pursuit.done(p_xy)
    if navigating:
      target, brake = pursuit.update(p_xy)
      lookahead = target
      left = (-fwd[1], fwd[0])
      v_target = min(navd.V_MAX, math.sqrt(2.0 * navd.ACCEL * max(0.0, brake - navd.POS_DEADBAND)))
      v_cmd = max(0.0, min(v_target, v_prev + navd.ACCEL * args.dt))
      v_prev = v_cmd
      to = target - p_xy
      dist = float(math.hypot(to[0], to[1]))
      if v_cmd > 0.0 and dist >= 1e-6:
        vx, vy = v_cmd * to[0] / dist, v_cmd * to[1] / dist
        chassis_cmd = (vx * fwd[0] + vy * fwd[1], vx * left[0] + vy * left[1])
    else:
      v_prev = 0.0

    if fwd is not None:
      left = np.array([-fwd[1], fwd[0]], np.float64)
      v_w = chassis_cmd[0] * np.array(fwd, np.float64) + chassis_cmd[1] * left
    else:
      v_w = np.zeros(2, np.float64)
    p_xy = p_xy + v_w * args.dt

    sp = pub.last("state_setpoint")
    if sp is not None:
      gimbal_yaw, gimbal_yaw_rate = step_axis(gimbal_yaw, float(sp["yaw"]), gimbal_max_rate, args.dt, wrap=True)
      gimbal_pitch, gimbal_pitch_rate = step_axis(gimbal_pitch, float(sp["pitch"]), gimbal_max_rate, args.dt)
    else:
      gimbal_yaw_rate = gimbal_pitch_rate = 0.0

    chain = state_chain(sm)
    state_name = "/".join(s.name for s in chain)
    selected_tag = current_tag_choice(chain, pose)
    spinning = pub.last("spinning") is True
    if k % trail_log_every == 0:
      trail.append(np.array([p_xy[0], p_xy[1], 0.03], np.float64))
    state_changed = state_name != prev_state_name
    if state_changed:
      log_state_change(sim_t)
      prev_state_name = state_name
    if hp_loss_event:
      log_hp_loss(sim_t)
    log_dynamic(sim_t, p_xy, v_w, gimbal_yaw, trail, state_name, state_ids.get(sm.current.name, -1), game_running,
                autoaim_valid, spinning, None if goal_xy is None else goal_xy.tolist(), path, robots, selected_tag, sp,
                chassis_cmd, lookahead, state_changed, robot_hp, hp_loss_event)

  print(f"wrote {out}  field_walls={len(common.FIELD_WALLS)}  style={args.style}  last_goal={inj_label}")

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", default=DEFAULT_OUT, help="output .rrd recording")
  ap.add_argument("--spawn", action="store_true", help="also spawn a Rerun viewer while writing the .rrd")
  ap.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="simulation duration in seconds")
  ap.add_argument("--dt", type=float, default=DEFAULT_DT, help="simulation timestep in seconds")
  ap.add_argument("--team", choices=sorted(TEAM_GOALS), default="blue")
  ap.add_argument("--style", choices=sorted(PLAY_STYLES), default="balanced")
  ap.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                  help="initial robot pose; default is team home facing center")
  ap.add_argument("--game-start", type=float, default=0.0, help="time when game_running becomes true")
  ap.add_argument("--game-stop", type=float, default=-1.0, help="time when game_running becomes false; <0 means never")
  ap.add_argument("--autoaim-window", action="append", type=parse_window, default=[], metavar="START:END",
                  help="make autoaim valid during this interval; repeatable")
  ap.add_argument("--enemy", action="append", type=parse_enemy, default=[], metavar="X,Y[,R[,T0,T1]]",
                  help="dynamic obstacle circle; repeatable")
  ap.add_argument("--hp-start", type=float, default=400.0, help="initial robot HP")
  ap.add_argument("--hp-loss", action="append", type=parse_hp_loss, default=[], metavar="TIME:AMOUNT",
                  help=f"subtract HP at sim time; repeatable. Retreat threshold is {RETREAT_HP:g}")
  ap.add_argument("--gimbal-rate-deg", type=float, default=360.0, help="simulated gimbal slew limit in deg/s")
  args = ap.parse_args()
  if args.duration <= 0: raise SystemExit("--duration must be > 0")
  if args.dt <= 0: raise SystemExit("--dt must be > 0")
  if args.hp_start < 0: raise SystemExit("--hp-start must be >= 0")
  run(args)

if __name__ == "__main__":
  main()
