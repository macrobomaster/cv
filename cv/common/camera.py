import cv2
import numpy as np

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
  cam.set_pixel_format_from_string("YUV422_YUYV_Packed")
  # camera is 1440x1080 or 4:3, crop to be 2:1
  cam.set_region(0, 0, 1440, 720)
  cam.set_exposure_time(10000)
  cam.set_gain(17)
  cam.set_binning(2, 2)
  cam.set_frame_rate(90)
  dev.set_string_feature_value("AcquisitionMode", "Continuous")
  cam.set_trigger("Software")

  strm = cam.create_stream()
  cam.start_acquisition()

  cam.software_trigger()
  payload = cam.get_payload()
  strm.push_buffer(Aravis.Buffer.new_allocate(payload))

  return cam, strm

def get_aravis_frame(cam, strm):
  cam.software_trigger()
  buf = strm.pop_buffer()
  img_data = buf.get_data()
  img_raw = np.frombuffer(img_data, dtype=np.uint8).reshape(cam.get_region()[3], cam.get_region()[2], 2)
  img = cv2.cvtColor(img_raw, cv2.COLOR_YUV2RGB_YUYV)
  strm.push_buffer(buf)
  return img
