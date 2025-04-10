import glob

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad.helpers import GlobalCounters, getenv
import cv2
import numpy as np

from .model import Model
from .common import pred
from ..common import BASE_PATH

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)
  if getenv("HALF", 0) == 1:
    for key, param in get_state_dict(model).items():
      if "norm" in key: continue
      if ".n" in key: continue
      param.replace(param.half()).realize()
  model.fuse()

  preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  i = 0
  while i < len(preprocessed_train_files):
    GlobalCounters.reset()

    file = preprocessed_train_files[i]
    img = cv2.imread(file)
    img = cv2.resize(img, (512, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # predict
    model_out = pred(model, Tensor(img, device="NPY")).tolist()[0]
    colorm, colorp, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, numberm, numberp = model_out
    match colorm:
      case 0: colorm = "none"
      case 1: colorm = "red"
      case 2: colorm = "blue"
      case 3: colorm = "blank"

    # draw the annotation
    cv2.putText(img, f"{colorm}: {colorp:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if colorm != "none" and colorp > 0.0:
      xc, yc = int(xc), int(yc)
      cv2.circle(img, (xc, yc), 5, (0, 255, 0), -1)
      cv2.circle(img, (int(xtl), int(ytl)), 5, (0, 255, 0), -1)
      cv2.circle(img, (int(xtr), int(ytr)), 5, (0, 255, 0), -1)
      cv2.circle(img, (int(xbl), int(ybl)), 5, (0, 255, 0), -1)
      cv2.circle(img, (int(xbr), int(ybr)), 5, (0, 255, 0), -1)
      # cv2.putText(img, f"{dist:.3f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      cv2.putText(img, f"{numberm}: {numberp:.3f}", (xc, yc - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      plate_width, plate_height = 0.095, 0.104
      square_points = np.array([
        [-plate_width/2, plate_height/2, 0], # bottom left
        [plate_width/2, plate_height/2, 0], # bottom right
        [plate_width/2, -plate_height/2, 0], # top right
        [-plate_width/2, -plate_height/2, 0], # top left
      ])

      image_points = np.array([
        [xbl, ybl],
        [xbr, ybr],
        [xtr, ytr],
        [xtl, ytl],
      ], dtype=np.float32).reshape(-1, 1, 2)

      f = 2 * 6
      sx, sy = 4.96, 3.72
      width, height = 512, 256
      camera_matrix = np.array([
        [width*f/sx, 0, width/2],
        [0, height*f/sy, height/2],
        [0, 0, 1],
      ], dtype=np.float32)
      dist_coeffs = np.zeros((4, 1))

      # ransac
      _, rvec, tvec = cv2.solvePnP(square_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

      # project 3d points to 2d
      imgpts, _ = cv2.projectPoints(square_points, rvec, tvec, camera_matrix, dist_coeffs)
      imgpts = imgpts.astype(int)
      cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 2)
      cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 2)
      cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 255, 0), 2)
      cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0, 255, 0), 2)

      # draw frame axes
      cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)
    if key == ord("q"): break
    elif key == ord("a"): i -= 1
    elif key == ord("f"): i += 100
    else: i += 1

  cv2.destroyAllWindows()
