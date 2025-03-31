import time, threading
from queue import Queue

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad.helpers import GlobalCounters, getenv
import cv2

from .model import Model
from .common import pred
from ..common import BASE_PATH
from ..common.camera import setup_aravis, get_aravis_frame

def frame_thread_fn(stop_event: threading.Event, frame_queue: Queue):
  cam, strm = setup_aravis()

  while not stop_event.is_set():
    sft = time.perf_counter()
    img = get_aravis_frame(cam, strm)
    img = cv2.resize(img, (512, 256))
    ft = time.perf_counter() - sft
    frame_queue.put((img, ft))

  cam.stop_acquisition()

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  stop_event = threading.Event()
  frame_queue = Queue(maxsize=1)
  frame_thread = threading.Thread(target=frame_thread_fn, args=(stop_event, frame_queue), daemon=True)
  frame_thread.start()

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)
  if getenv("HALF", 0) == 1:
    for key, param in get_state_dict(model).items():
      if "norm" in key: continue
      if ".n" in key: continue
      param.replace(param.half()).realize()

  st = time.perf_counter()
  img, ft = frame_queue.get()
  imgt = Tensor(img, dtype=dtypes.uint8)
  print(f"first frame aquisition time: {ft:.3f}")
  try:
    while True:
      GlobalCounters.reset()

      # run model
      spt = time.perf_counter()
      model_out = pred(model, imgt).tolist()[0]
      pt = time.perf_counter() - spt

      img, ft = frame_queue.get()
      imgt = Tensor(img, dtype=dtypes.uint8, device="NPY")

      # copy from gpu to cpu
      smt = time.perf_counter()
      cl, clp, x, y, colorm, colorp, numberm, numberp = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5], model_out[6], model_out[7]
      match colorm:
        case 1: colorm = "red"
        case 2: colorm = "blue"
        case _: colorm = "blank"
      mt = time.perf_counter() - smt

      dt = time.perf_counter() - st
      st = time.perf_counter()
      print(f"frame aquisition time: {ft:.3f}, python time: {pt:.3f}, model time: {mt:.3f}, total time: {dt:.3f}")
      cv2.putText(img, f"{1/dt:.2f} FPS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)

      # draw the annotation
      cv2.putText(img, f"{cl}: {clp:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      if cl == 1 and clp > 0.6:
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        # cv2.putText(img, f"{dist:.3f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"{colorm}: {colorp:.3f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"{numberm}: {numberp:.3f}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      # display
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imshow("img", img)

      key = cv2.waitKey(1)
      if key == ord("q"): break
  except KeyboardInterrupt:
    pass

  cv2.destroyAllWindows()

  stop_event.set()
  while not frame_queue.empty(): _ = frame_queue.get()
  frame_thread.join()
