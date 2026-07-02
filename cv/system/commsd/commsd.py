import time
from pathlib import Path

import serial

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from .protocol import Protocol, Command, State

COMM_RATES_HZ = {
  "gimbal_state": 200.0,
  "raw_imu": 200.0,
  "chassis_odom": 50.0,
  "aim_error": 200.0,
  "shoot": 100.0,
  "chassis_velocity": 50.0,
  "spinning": 10.0,
  "chassis_align": 10.0,
  "game_running": 1.0,
  "team_color": 0.2,
  "robot_type": 0.2,
  "barrel_heat": 10.0,
  "robot_hp": 10.0,
}
COMMS_RATE_WINDOW = 1.0

def due(next_comm_at, comm):
  t = time.monotonic()
  if t < next_comm_at[comm]:
    return False
  next_comm_at[comm] += 1 / COMM_RATES_HZ[comm]
  if next_comm_at[comm] < t:
    next_comm_at[comm] = t + 1 / COMM_RATES_HZ[comm]
  return True

def fresh(sub, service):
  return sub.alive[service] or sub.updated[service]

def comm_msg(protocol, comm_counts, comm, command, *args):
  comm_counts[comm] += 1
  return protocol.msg(command, *args)

def publish_comm_rates(pub, comm_counts, comm_rates, last_rate_at):
  t = time.monotonic()
  dt = t - last_rate_at
  if dt < COMMS_RATE_WINDOW:
    return last_rate_at
  for comm in comm_rates:
    comm_rates[comm] = comm_counts[comm] / dt
    comm_counts[comm] = 0
  pub.send("comms_rates", {"t": t, "rates": comm_rates.copy()})
  return t

def run():
  kv_put("watchdog", "commsd", time.monotonic())

  if Path("/dev/ttyTHS1").exists():
    device = "/dev/ttyTHS1"
  else:
    device = "/dev/ttyUSB0"
  logger.info(f"using device {device}")
  port = serial.Serial(device, 921600, timeout=1)
  protocol = Protocol(port)

  pub = messaging.Pub(["game_running", "team_color", "robot_type", "comms_rates", "chassis_odom", "barrel_heat", "robot_hp"])
  gimbal_pub = messaging.Pub(["gimbal_state", "raw_imu"], conflate=False)
  sub = messaging.Sub(["aim_error", "shoot", "chassis_velocity", "spinning", "chassis_align"])
  next_comm_at = {comm: 0.0 for comm in COMM_RATES_HZ}
  comm_counts = {comm: 0 for comm in COMM_RATES_HZ}
  comm_rates = {comm: 0.0 for comm in COMM_RATES_HZ}
  last_rate_at = time.monotonic()
  last_wd = time.monotonic()

  while True:
    if time.monotonic() - last_wd > 1:
      kv_put("watchdog", "commsd", time.monotonic())
      last_wd = time.monotonic()

    timeout_ms = max(0, int((min(next_comm_at.values()) - time.monotonic()) * 1000))
    sub.update(timeout=timeout_ms)

    if due(next_comm_at, "gimbal_state"):
      gs = comm_msg(protocol, comm_counts, "gimbal_state", Command.GIMBAL_STATE)
      if gs is not None:
        pitch_angle, yaw_angle, pitch_rate, yaw_rate = gs
        gimbal_pub.send("gimbal_state", {
          "t_stamp": time.monotonic(),
          "yaw_gi": yaw_angle,
          "yaw_rate_gi": yaw_rate,
          "pitch_gi": pitch_angle * -1,
          "pitch_rate_gi": pitch_rate * -1,
        })

    if due(next_comm_at, "raw_imu"):
      ra = comm_msg(protocol, comm_counts, "raw_imu", Command.RAW_ACCEL)
      if ra is not None:
        ax, ay, az, delta = ra
        gimbal_pub.send("raw_imu", {
          "t": time.monotonic(),
          "accel": [ax, ay, az],
        })

    if due(next_comm_at, "chassis_odom"):
      co = comm_msg(protocol, comm_counts, "chassis_odom", Command.CHASSIS_ODOM)
      if co is not None:
        vx, vy = co
        pub.send("chassis_odom", {"t": time.monotonic(), "vx": vx, "vy": vy})

    aim_error = sub["aim_error"]
    shoot = sub["shoot"]
    chassis_velocity = sub["chassis_velocity"]

    if aim_error is not None and due(next_comm_at, "aim_error"):
      if fresh(sub, "aim_error"):
        x = aim_error["x"] * -1
        y = aim_error["y"] * -1
        comm_msg(protocol, comm_counts, "aim_error", Command.AIM_ERROR, x, y)
      else:
        comm_msg(protocol, comm_counts, "aim_error", Command.AIM_ERROR, 0.0, 0.0)

    if shoot is not None and due(next_comm_at, "shoot"):
      if fresh(sub, "shoot"):
        comm_msg(protocol, comm_counts, "shoot", Command.CONTROL_SHOOT, 0xff if shoot else 0x00)
      else:
        comm_msg(protocol, comm_counts, "shoot", Command.CONTROL_SHOOT, 0x00)

    if chassis_velocity is not None and due(next_comm_at, "chassis_velocity"):
      if fresh(sub, "chassis_velocity"):
        x = chassis_velocity["x"]
        y = chassis_velocity["y"]
        comm_msg(protocol, comm_counts, "chassis_velocity", Command.MOVE_ROBOT, x, y)
      else:
        comm_msg(protocol, comm_counts, "chassis_velocity", Command.MOVE_ROBOT, 0.0, 0.0)

    if due(next_comm_at, "spinning"):
      if fresh(sub, "spinning"):
        comm_msg(protocol, comm_counts, "spinning", Command.CONTROL_SPINNING, 0xff)
      else:
        comm_msg(protocol, comm_counts, "spinning", Command.CONTROL_SPINNING, 0x00)

    if due(next_comm_at, "chassis_align"):
      if fresh(sub, "chassis_align"):
        comm_msg(protocol, comm_counts, "chassis_align", Command.CHASSIS_ALIGN, 0xff)
      else:
        comm_msg(protocol, comm_counts, "chassis_align", Command.CHASSIS_ALIGN, 0x00)

    if due(next_comm_at, "game_running"):
      game_running = comm_msg(protocol, comm_counts, "game_running", Command.CHECK_STATE, State.GAME_RUNNING.value)
      if game_running is not None:
        pub.send("game_running", True if game_running[1] == 0x00 else False)

    if due(next_comm_at, "team_color"):
      team_color = comm_msg(protocol, comm_counts, "team_color", Command.CHECK_STATE, State.TEAM_COLOR.value)
      if team_color is not None:
        pub.send("team_color", "red" if team_color[1] == 0x00 else "blue")

    if due(next_comm_at, "robot_type"):
      robot_type = comm_msg(protocol, comm_counts, "robot_type", Command.CHECK_STATE, State.ROBOT_TYPE.value)
      if robot_type is not None:
        pub.send("robot_type", "sentry" if robot_type[1] == 0x00 else "standard")

    if due(next_comm_at, "barrel_heat"):
      ba = comm_msg(protocol, comm_counts, "barrel_heat", Command.BARREL_HEAT)
      if ba is not None:
        limit, current = ba
        pub.send("barrel_heat", {"limit": limit, "current": current})

    if due(next_comm_at, "robot_hp"):
      ro = comm_msg(protocol, comm_counts, "robot_hp", Command.ROBOT_HP)
      if ro is not None:
        pub.send("robot_hp", ro[0])

    last_rate_at = publish_comm_rates(pub, comm_counts, comm_rates, last_rate_at)
