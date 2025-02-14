import time

import cv2

from ..common import BASE_PATH
from ..common.camera import setup_aravis, get_aravis_frame

if __name__ == "__main__":
  cam, strm = setup_aravis()

  st = time.perf_counter()
  writer = None
  record = False
  while True:
    img = get_aravis_frame(cam, strm)
    # resize
    img = cv2.resize(img, (512, 256))
    # increase brightness
    # img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    dt = time.perf_counter() - st
    st = time.perf_counter()

    if record and writer is not None:
      writer.write(img)

    cv2.putText(img, f"{1/dt:.2f} FPS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
    cv2.putText(img, f"{'RECORDING' if record else ''}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 2)

    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("r"):
      # start recording
      record = str(BASE_PATH / "record" / f"{time.time()}.mp4")
      writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (512, 256))
    elif key == ord("s") and writer is not None:
      # stop recording
      print(f"recorded to {record}")
      record = False
      writer.release()

    # time.sleep(0.025)

  cv2.destroyAllWindows()
  cam.stop_acquisition()
