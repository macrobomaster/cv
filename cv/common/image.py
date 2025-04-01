from tinygrad.tensor import Tensor
import cv2
import numpy as np

def rgb_to_yuv420(img):
  height, width = img.shape[:2]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)

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

def rgb_to_yuv420_tensor(img:Tensor) -> Tensor:
  if not hasattr(rgb_to_yuv420_tensor, "coeffs"):
    coeffs = Tensor([[0.299, 0.587, 0.114],
                   [-0.168736, -0.331264, 0.5],
                   [0.5, -0.418688, -0.081312]])
    setattr(rgb_to_yuv420_tensor, "coeffs", coeffs)
  else:
    coeffs = getattr(rgb_to_yuv420_tensor, "coeffs")

  yuv = img.matmul(coeffs.T)

  y0 = yuv[:, ::2, ::2, 0]
  y1 = yuv[:, 1::2, ::2, 0]
  y2 = yuv[:, ::2, 1::2, 0]
  y3 = yuv[:, 1::2, 1::2, 0]
  u = yuv[:, :, :, 1].add(128).avg_pool2d(2, 2)
  v = yuv[:, :, :, 2].add(128).avg_pool2d(2, 2)

  return Tensor.stack(y0, y1, y2, y3, u, v, dim=-1).clamp(0, 255)

def alpha_overlay(img, background, x, y):
  alpha = img[:, :, 3] / 255.0
  alpha = alpha[:, :, np.newaxis]
  img = img[:, :, :3]
  background[y:y+img.shape[0], x:x+img.shape[1]] = img * alpha + background[y:y+img.shape[0], x:x+img.shape[1]] * (1 - alpha)

def resize_crop(img, target_width, target_height):
  if img is None or img.size == 0:
    raise ValueError("Input image is empty.")
  if target_width <= 0 or target_height <= 0:
    raise ValueError("Target width and height must be positive.")

  original_height, original_width = img.shape[:2]
  target_aspect = target_width / target_height
  original_aspect = original_width / original_height

  # Calculate scaling factor and intermediate size
  if original_aspect > target_aspect:
    # Original image is wider than target aspect ratio
    scale_ratio = target_height / original_height
    new_width = int(scale_ratio * original_width)
    new_height = target_height
  else:
    # Original image is taller or same aspect ratio
    scale_ratio = target_width / original_width
    new_width = target_width
    new_height = int(scale_ratio * original_height)

  # Resize the image
  # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging generally
  interpolation = cv2.INTER_AREA if scale_ratio < 1 else cv2.INTER_LINEAR
  resized_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)

  # Calculate cropping coordinates (center crop)
  start_x = (new_width - target_width) // 2
  start_y = (new_height - target_height) // 2
  # Make sure start coordinates are not negative (can happen with floating point inaccuracies)
  start_x = max(0, start_x)
  start_y = max(0, start_y)

  end_x = start_x + target_width
  end_y = start_y + target_height

  # Crop the image
  # Ensure cropping bounds do not exceed resized image dimensions
  cropped_img = resized_img[start_y:min(end_y, new_height), start_x:min(end_x, new_width)]

  # Final resize to guarantee exact target dimensions (handles potential rounding errors)
  # If cropped_img dimensions are already correct, this won't change much
  final_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=interpolation)

  return final_img

if __name__ == "__main__":
  img = cv2.imread("test.png")
  img = cv2.resize(img, (812, 1080))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  yuv_tensor = rgb_to_yuv420_tensor(Tensor(img).unsqueeze(0))
  yuv = yuv_tensor.numpy()[0]
  yuv_ref = rgb_to_yuv420(img)
  for i in range(6):
    # put reference next to our implementation
    cv2.imshow("yuv", np.hstack([yuv_ref[..., i], yuv[..., i]]))
    cv2.waitKey(0)
