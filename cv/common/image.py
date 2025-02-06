from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import cv2
import numpy as np

def bgr_to_yuv420(img):
  height, width = img.shape[:2]
  img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)

  # extract U and V channels
  u = img[height:height+height//4].reshape(-1, width//2)
  v = img[height+height//4:height+height//2].reshape(-1, width//2)

  # seperate the Y channel into 4 channels
  y0 = img[:height:2, :width:2]
  y1 = img[1:height:2, :width:2]
  y2 = img[:height:2, 1:width:2]
  y3 = img[1:height:2, 1:width:2]

  # stack the channels
  return np.stack([y0, y1, y2, y3, u, v], axis=-1)

def bgr_to_yuv420_tensor(img:Tensor) -> Tensor:
  coeffs = Tensor([[0.299, 0.587, 0.114],
                   [-0.168736, -0.331264, 0.5],
                   [0.5, -0.418688, -0.081312]])
  yuv = img[:, :, :, ::-1].matmul(coeffs.T)
  yuv = yuv.add(Tensor([0, 128, 128]))

  y0 = yuv[:, ::2, ::2, 0]
  y1 = yuv[:, 1::2, ::2, 0]
  y2 = yuv[:, ::2, 1::2, 0]
  y3 = yuv[:, 1::2, 1::2, 0]
  u = yuv[:, :, :, 1].avg_pool2d(2, 2)
  v = yuv[:, :, :, 2].avg_pool2d(2, 2)

  return Tensor.stack(y0, y1, y2, y3, u, v, dim=-1).clamp(0, 255).cast(dtypes.uint8)

if __name__ == "__main__":
  img = cv2.imread("test.png")
  img = cv2.resize(img, (812, 1080))

  yuv_tensor = bgr_to_yuv420_tensor(Tensor(img).unsqueeze(0))
  yuv = yuv_tensor.numpy()[0]
  yuv_ref = bgr_to_yuv420(img)
  for i in range(6):
    # put reference next to our implementation
    cv2.imshow("yuv", np.hstack([yuv_ref[..., i], yuv[..., i]]))
    cv2.waitKey(0)
