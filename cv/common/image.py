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
