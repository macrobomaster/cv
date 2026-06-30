import time

import numpy as np
from tinygrad.helpers import getenv

import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis

# === camera ===

def setup_aravis():
  Aravis.update_device_list()
  devices = Aravis.get_n_devices()
  if devices == 0:
    raise Exception("no cameras found")

  print(f"using {Aravis.get_device_id(0)}")
  cam = Aravis.Camera.new(Aravis.get_device_id(0))
  cam.stop_acquisition()
  dev = cam.get_device()
  dev.set_string_feature_value("UserSetSelector", "Default")
  dev.execute_command("UserSetLoad")
  dev.set_string_feature_value("DeviceLinkThroughputLimitMode", "Off")
  try:
    dev.set_string_feature_value("ADCBitDepth", "ADCBitDepth_12")
  except: pass
  cam.set_pixel_format_from_string("RGB8Packed")
  try:
    dev.set_boolean_feature_value("SuperBayerEnable", True)
  except: pass
  cam.set_binning(2, 2)
  bw, bh = 1440 // 2, 1080 // 2
  ch = bw // 2
  cam.set_region(0, (bh - ch) // 2, bw, ch)
  dev.set_string_feature_value("ExposureAuto", "Off")
  cam.set_exposure_time(4000)
  dev.set_string_feature_value("GainAuto", "Off")
  cam.set_gain(getenv("GAIN", 17.0))
  dev.set_string_feature_value("BalanceWhiteAuto", "Once")
  dev.set_string_feature_value("AcquisitionMode", "Continuous")
  dev.set_string_feature_value("TriggerMode", "Off")
  dev.set_boolean_feature_value("AcquisitionFrameRateEnable", False)

  strm = cam.create_stream()
  payload = cam.get_payload()
  for _ in range(8):
    strm.push_buffer(Aravis.Buffer.new_allocate(payload))
  cam.start_acquisition()

  # converge wb
  for _ in range(60):
    strm.push_buffer(strm.pop_buffer())
  ratios = {}
  for ch in ("Red", "Green", "Blue"):
    dev.set_string_feature_value("BalanceRatioSelector", ch)
    try: ratios[ch] = dev.get_integer_feature_value("BalanceRatio")
    except Exception: ratios[ch] = dev.get_float_feature_value("BalanceRatio")
  print(f"white balance ratios (lock with BalanceWhiteAuto Off): {ratios}")
  print(f"resulting frame rate: {dev.get_float_feature_value('ResultingFrameRate'):.1f} fps")
  try:
    link_bps = dev.get_integer_feature_value("DeviceLinkSpeed")  # bytes/s; SuperSpeed ~5e8, High-Speed/USB2 ~6e7
    print(f"device link speed: {link_bps / 1e6:.0f} MB/s ({'SuperSpeed' if link_bps > 1e8 else 'HIGH-SPEED — USB2 fallback, check cable/port!'})")
  except Exception: pass

  ts_hz = dev.get_integer_feature_value("DeviceTimestampIncrement")
  return cam, strm, dev, ts_hz, latch_timestamp_offset(dev, ts_hz)

def latch_timestamp_offset(dev, ts_hz) -> int:
  t0 = time.monotonic_ns()
  dev.execute_command("DeviceTimestampLatch")
  t1 = time.monotonic_ns()
  dev_ns = dev.get_integer_feature_value("DeviceTimestamp") * 1_000_000_000 // ts_hz
  return (t0 + t1) // 2 - dev_ns

def get_aravis_frame_view(cam, strm, ts_hz, offset):
  # 2s: free-run delivers a frame every ~5-13ms, so a 2s gap with the link up is a
  # stream stall (not a slow frame). Surface the SDK stats (silent in dmesg) and bail
  # to the caller's usbreset+restart instead of blocking until the 10s watchdog SIGKILLs.
  buf = strm.timeout_pop_buffer(2_000_000)
  if buf is None:
    n_done, n_fail, n_under = strm.get_statistics()
    raise TimeoutError(f"frame stall: completed={n_done} failures={n_fail} underruns={n_under}")
  while (nb := strm.try_pop_buffer()) is not None:
    strm.push_buffer(buf)
    buf = nb
  ct = (buf.get_timestamp() * 1_000_000_000 // ts_hz + offset) / 1e9
  status = buf.get_status()
  if status != Aravis.BufferStatus.SUCCESS:
    strm.push_buffer(buf)
    return None, 0.0, None
  img_raw = np.frombuffer(buf.get_data(), dtype=np.uint8).reshape(cam.get_region()[3], cam.get_region()[2], 3)
  return img_raw, ct, buf

def get_aravis_frame(cam, strm, ts_hz, offset):
  img_raw, ct, buf = get_aravis_frame_view(cam, strm, ts_hz, offset)
  if img_raw is None: return None, 0.0
  try:
    # copy before requeue: in free-run the camera may immediately refill this buffer
    return img_raw.copy(), ct
  finally:
    strm.push_buffer(buf)
