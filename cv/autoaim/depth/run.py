import glob
from functools import partial
from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.extra.onnx import OnnxRunner
from tinygrad.helpers import GlobalCounters, trange, tqdm
import onnx
import cv2
import numpy as np

from ...common import BASE_PATH
from ..common import get_annotation

if __name__ == "__main__":
  model = onnx.load(Path(__file__).parent.parent.parent.parent / "weights/metric3d_v2_small.onnx")

  # change resize to linear not cubic
  for node in model.graph.node:
    if node.op_type == "Resize":
      for attr in node.attribute:
        if attr.name == "mode":
          if attr.s == b"cubic":
            print(f"changing {node.name} mode to linear")
            attr.s = b"linear"

  runner = OnnxRunner(model)
  @partial(TinyJit, prune=True)
  def run_onnx_jit(**kwargs):
    outputs = runner(kwargs)
    return outputs["predicted_depth"], outputs["predicted_normal"]

  # load image
  preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  print(f"found {len(preprocessed_train_files)} files")
  for i in trange(len(preprocessed_train_files)):
    GlobalCounters.reset()

    file = preprocessed_train_files[i]
    anno = get_annotation(file)
    if anno.detected == 0 or anno.dist > 0:
      continue
    x, y = int(anno.x * 559), int((1 - anno.y) * 279)
    img = cv2.imread(file)

    img = cv2.resize(img, (560, 280))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Tensor(np.array([img], dtype=np.float32)).permute(0, 3, 1, 2)

    depth_map, normal_map = run_onnx_jit(pixel_values=img)
    depth_map = depth_map.numpy()
    # normal_map = normal_map.permute(0, 2, 3, 1).numpy()

    # visualize depth map
    depth_map = depth_map[0]
    # get dist from depth map
    dist = depth_map[y, x]
    # write dist to file
    with open(Path(file).with_suffix(".depth"), "w") as f:
      f.write(f"{dist}")

    depth_map_view = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map_view = 1 - depth_map_view
    cv2.circle(depth_map_view, (x, y), 2, (0, 255, 0), -1)
    cv2.putText(depth_map_view, f"{dist:.3f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("depth", depth_map_view)

    # visualize normal map
    # normal_map = normal_map[0]
    # normal_map = np.clip(normal_map, -1, 1)
    # normal_map = (normal_map + 1) / 2
    # normal_map = (normal_map * 255).astype(np.uint8)
    # print(normal_map.shape)
    # cv2.imshow("normal", normal_map)

    key = cv2.waitKey(1)
    if key == ord("q"): break

  cv2.destroyAllWindows()
