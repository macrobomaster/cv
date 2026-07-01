import math, sys, time
from collections import deque

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ..system.core import messaging
from ..system.core.helpers import FrequencyKeeper
from ..autoaim.common import IMG_H, IMG_W

# Gimbal-inertial is z-up RH (x=forward, y=left, z=up) — already Rerun FLU, so this is identity.
def gi_to_flu(p) -> list:
  p = np.asarray(p, dtype=float)
  return [float(p[0]), float(p[1]), float(p[2])]

CLASS_COLOR = {
  "TRACKING": [0,   255, 0,   200],
  "UNKNOWN":  [180, 180, 180, 120],
  "LOST":     [80,  80,  80,  120],
}

VISUAL_SERVICES = [
  "camera_feed_full", "autoaim", "plate", "gimbal_state", "aim_angle", "aim_error", "shoot", "chassis_velocity",
  "game_running", "team_color", "robot_type", "comms_rates",
]
COMMS_RATE_SERVICES = [
  "gimbal_state", "aim_error", "shoot", "chassis_velocity", "spinning", "game_running", "team_color", "robot_type",
]
PLOT_TIME_WINDOW_S = 20.0

def axis_arrows(origin=(0.0, 0.0, 0.0), length=0.3):
  return rr.Arrows3D(
    origins=[origin] * 3,
    vectors=[[length, 0, 0], [0, length, 0], [0, 0, length]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=["x", "y", "z"],
  )

def time_series_view(name, contents):
  return rrb.TimeSeriesView(
    name=name,
    origin="/",
    contents=contents,
    axis_x=rrb.TimeAxis(view_range=rr.TimeRange(
      start=rrb.TimeRangeBoundary.cursor_relative(seconds=-PLOT_TIME_WINDOW_S),
      end=rrb.TimeRangeBoundary.cursor_relative(),
    )),
  )

rr.init("cv", spawn=True)
rr.send_blueprint(rrb.Blueprint(
  rrb.TimePanel(timeline="time", play_state="following"),
  rrb.Grid(
    rrb.Spatial3DView(name="World", origin="/gi"),
    rrb.Spatial2DView(name="Camera", origin="/raw_camera"),
    rrb.Tabs(
      time_series_view("Gimbal Angles", ["/scalars/gimbal_yaw_gi", "/scalars/gimbal_pitch_gi"]),
      time_series_view("Gimbal Rates", ["/scalars/gimbal_yaw_rate", "/scalars/gimbal_pitch_rate"]),
      time_series_view("Target XYZ", ["/scalars/target_x", "/scalars/target_y", "/scalars/target_z"]),
      time_series_view("Distance", [
        "/scalars/dist/filtered", "/scalars/dist/measured", "/scalars/dist/filtered_hi", "/scalars/dist/filtered_lo", "/scalars/dist/sigma",
      ]),
      time_series_view("Aim", [
        "/scalars/aim_error_yaw_deg", "/scalars/aim_error_pitch_deg", "/scalars/aim_angle_x_deg", "/scalars/aim_angle_y_deg",
      ]),
      time_series_view("Autoaim", ["/scalars/autoaim_valid", "/scalars/autoaim_confidence"]),
      time_series_view("Commands", ["/scalars/shoot"]),
      name="Plots",
    ),
    rrb.Tabs(
      time_series_view("Performance", ["/fps", "/scalars/model_infer_ms"]),
      time_series_view("Comms", ["/rates_hz"]),
      time_series_view("Alive", ["/alive"]),
      name="System",
    ),
    grid_columns=2,
  ),
  auto_layout=False,
  auto_views=False,
  collapse_panels=False,
))

# Gimbal-inertial world: z-up RH (+x forward, +y left, +z up) = Rerun FLU.
rr.log("gi", rr.ViewCoordinates.FLU, static=True)
rr.log("gi/origin", axis_arrows(length=0.5), static=True)

addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
sub = messaging.Sub(VISUAL_SERVICES, addr=addr)

# One 0/1 step line per service, each offset into its own lane so the stack reads like a state
# timeline but still follows the cursor (StateTimelineView can't — it auto-fits once, no scroll).
alive_services = sorted(sub.services)
rr.log("alive", rr.SeriesLines(names=alive_services), static=True)
rr.log("fps", rr.SeriesLines(names=["camera", "model"]), static=True)
rr.log("rates_hz", rr.SeriesLines(names=COMMS_RATE_SERVICES), static=True)

chassis_pos = deque(maxlen=500)
fk = FrequencyKeeper(30)
last_cam_log = 0.0
last_cam_fid, last_cam_ct, cam_fps = None, 0.0, None
last_model_fid, last_model_ct, model_fps = None, 0.0, None
last_target_id = -1

while True:
  sub.update()
  now = time.monotonic()
  rr.set_time("time", duration=now)

  comms_rates = sub["comms_rates"]
  if sub.updated["comms_rates"] and comms_rates is not None:
    rates = comms_rates["rates"]
    rr.log("rates_hz", rr.Scalars([rates.get(service, 0.0) for service in COMMS_RATE_SERVICES]))

  # Camera fps from the source frame counter. The sub is conflated, so we only see the newest
  # frame each tick — but Δfid counts every frame camerad produced between the two we saw, so
  # Δfid/Δct recovers the true capture rate regardless of this loop's rate.
  camera_feed = sub["camera_feed_full"]   # full frames come from the framed bridge (DEBUG>=1)
  if sub.updated["camera_feed_full"] and camera_feed is not None and "fid" in camera_feed:
    fid, ct = camera_feed["fid"], camera_feed["ct"]
    if last_cam_fid is not None and fid > last_cam_fid and ct > last_cam_ct:
      fps = (fid - last_cam_fid) / (ct - last_cam_ct)
      cam_fps = fps if cam_fps is None else 0.9 * cam_fps + 0.1 * fps   # EMA smooth
    last_cam_fid, last_cam_ct = fid, ct

  # 2D camera feed (rate-limit)
  if camera_feed is not None and time.monotonic() - last_cam_log > 0.1:
    frame = np.frombuffer(camera_feed["frame"], dtype=np.uint8).reshape(IMG_H, IMG_W, 3)
    rr.log("raw_camera/feed", rr.Image(frame).compress(70))
    last_cam_log = time.monotonic()

  # Gimbal state — orientation gizmo at origin
  gimbal_state = sub["gimbal_state"]
  if sub.updated["gimbal_state"] and gimbal_state is not None:
    rr.log("scalars/gimbal_yaw_gi", rr.Scalars(math.degrees(gimbal_state["yaw_gi"])))
    rr.log("scalars/gimbal_pitch_gi", rr.Scalars(math.degrees(gimbal_state["pitch_gi"])))
    rr.log("scalars/gimbal_yaw_rate", rr.Scalars(math.degrees(gimbal_state["yaw_rate_gi"])))
    rr.log("scalars/gimbal_pitch_rate", rr.Scalars(math.degrees(gimbal_state["pitch_rate_gi"])))
    # Aim direction in gi (z-up RH): +x rotated by yaw about +z (rotz), elevated by pitch about +z.
    yaw, pitch = gimbal_state["yaw_gi"], gimbal_state["pitch_gi"]
    aim_dir = np.array([math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), math.sin(pitch)])
    rr.log("gi/gimbal_aim", rr.Arrows3D(origins=[gi_to_flu([0, 0, 0])],
                                         vectors=[gi_to_flu(aim_dir.tolist())],
                                         colors=[[255, 255, 0]]))

  # Plate v2
  plate = sub["plate"]
  if sub.updated["plate"] and plate is not None:
    cls = plate["class"]
    target_id = plate["target_id"]
    pos = np.array(plate["pos_gi"])
    vel = np.array(plate["vel_gi"])
    cov = np.array(plate["cov_pos"])
    color = CLASS_COLOR.get(cls, [200, 200, 200, 200])

    rr.log("gi/target/pos", rr.Points3D([gi_to_flu(pos)], colors=[color], radii=[0.04],
                                         labels=[f"{cls} #{target_id}"]))
    rr.log("gi/target/vel", rr.Arrows3D(origins=[gi_to_flu(pos)],
                                        vectors=[gi_to_flu(vel.tolist())],
                                        colors=[color]))
    # Position uncertainty as a sphere with radius = sqrt(trace(cov)/3) — 1σ-ish.
    sigma = float(math.sqrt(max(0.0, np.trace(cov) / 3.0)))
    rr.log("gi/target/cov", rr.Points3D([gi_to_flu(pos)], colors=[[*color[:3], 80]],
                                         radii=[sigma]))

    rr.log("scalars/target_x", rr.Scalars(float(pos[0])))
    rr.log("scalars/target_y", rr.Scalars(float(pos[1])))
    rr.log("scalars/target_z", rr.Scalars(float(pos[2])))

    # Distance: Kalman-filtered vs raw PnP measurement, with a radial 1σ band from cov.
    dist_filt = float(np.linalg.norm(pos))
    rr.log("scalars/dist/filtered", rr.Scalars(dist_filt))
    if dist_filt > 1e-6:
      u = pos / dist_filt                       # line-of-sight unit vector
      sigma_r = float(math.sqrt(max(0.0, u @ cov @ u)))   # radial 1σ along LOS
      rr.log("scalars/dist/filtered_hi", rr.Scalars(dist_filt + sigma_r))
      rr.log("scalars/dist/filtered_lo", rr.Scalars(dist_filt - sigma_r))
      rr.log("scalars/dist/sigma", rr.Scalars(sigma_r))
    pos_meas = plate.get("pos_meas")
    if pos_meas is not None:
      rr.log("scalars/dist/measured", rr.Scalars(float(np.linalg.norm(pos_meas))))

  # Autoaim raw — scalars + 2D corner overlay on the camera feed (so detections are visible
  # straight from autoaimd, without plated running).
  autoaim = sub["autoaim"]
  if sub.updated["autoaim"] and autoaim is not None:
    rr.log("scalars/autoaim_valid", rr.Scalars(int(autoaim["valid"])))
    rr.log("scalars/autoaim_confidence", rr.Scalars(float(autoaim["confidence"])))

    # model throughput via Δfid/Δt_capture (same conflate-robust trick as camera_fps); below camera_fps
    # means inference is the bottleneck. infer_ms is the raw forward-pass time.
    if "fid" in autoaim:
      mfid, mct = autoaim["fid"], autoaim["t_capture"]
      if last_model_fid is not None and mfid > last_model_fid and mct > last_model_ct:
        mfps = (mfid - last_model_fid) / (mct - last_model_ct)
        model_fps = mfps if model_fps is None else 0.9 * model_fps + 0.1 * mfps
      last_model_fid, last_model_ct = mfid, mct
      rr.log("scalars/model_infer_ms", rr.Scalars(float(autoaim["infer_ms"])))

    if autoaim["detected"]:
      # corners are [0,1] image-normalized, order TL, TR, BL, BR.
      c = autoaim["corners"]
      px = [[c[2*i] * IMG_W, c[2*i + 1] * IMG_H] for i in range(4)]
      tl, tr, bl, br = px
      # green when it passes the valid gate, amber when detected-but-low-confidence.
      col = [0, 255, 0] if autoaim["valid"] else [255, 180, 0]
      label = f"{autoaim['color']} #{autoaim['number']} {autoaim['confidence']:.2f}"
      rr.log("raw_camera/feed/box",
             rr.LineStrips2D([[tl, tr, br, bl, tl]], colors=[col], radii=[1.0], labels=[label]))
      # per-corner dots, distinctly colored to expose TL/TR/BL/BR ordering for PnP debugging.
      rr.log("raw_camera/feed/corners",
             rr.Points2D(px, colors=[[255, 0, 0], [0, 255, 0], [0, 128, 255], [255, 255, 0]],
                         radii=[3.0]))
    else:
      rr.log("raw_camera/feed/box", rr.Clear(recursive=False))
      rr.log("raw_camera/feed/corners", rr.Clear(recursive=False))

  # Aim error
  aim_error = sub["aim_error"]
  if sub.updated["aim_error"] and aim_error is not None:
    rr.log("scalars/aim_error_yaw_deg", rr.Scalars(math.degrees(aim_error["x"])))
    rr.log("scalars/aim_error_pitch_deg", rr.Scalars(math.degrees(aim_error["y"])))

  aim_angle = sub["aim_angle"]
  if sub.updated["aim_angle"] and aim_angle is not None:
    rr.log("scalars/aim_angle_x_deg", rr.Scalars(aim_angle["x"]))
    rr.log("scalars/aim_angle_y_deg", rr.Scalars(aim_angle["y"]))

  shoot = sub["shoot"]
  if sub.updated["shoot"] and shoot is not None:
    rr.log("scalars/shoot", rr.Scalars(int(shoot)))

  # Chassis trail (integrated commanded velocity — not actual odometry)
  chassis_velocity = sub["chassis_velocity"]
  if sub.updated["chassis_velocity"] and chassis_velocity is not None:
    last = chassis_pos[-1] if chassis_pos else (0.0, 0.0, 0.0)
    # chassis_velocity {x:forward, y:left} integrated in the z-up gi ground plane (z=0).
    new = (last[0] + chassis_velocity["x"] * 0.05, last[1] + chassis_velocity["y"] * 0.05, 0.0)
    chassis_pos.append(new)
    if len(chassis_pos) >= 2:
      rr.log("gi/chassis_trail",
             rr.LineStrips3D([[gi_to_flu(p) for p in chassis_pos]], colors=[[120, 120, 255]]))

  # Lane i sits at baseline i (dead) and rises to i+0.8 (alive); logged every tick so the step
  # lines stay flat between transitions instead of ramping diagonally to the next sample.
  rr.log("alive", rr.Scalars([i + 0.8 * sub.alive[service] for i, service in enumerate(alive_services)]))
  rr.log("fps", rr.Scalars([cam_fps or 0.0, model_fps or 0.0]))

  fk.step()
