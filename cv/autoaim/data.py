from tinygrad.helpers import getenv
import random
import albumentations as A
import cv2
import numpy as np

from .syndata import generate_sample, generate_sequence
from .common import T, IMG_H, IMG_W
from ..common.dataloader import DataloaderProc

# Augmentation is split by temporal scope. SEQ_PIPELINE holds scene/sensor properties that stay
# stable across a short clip (white balance, exposure, color, shadows, focus, resolution) — its
# params are sampled once and applied identically to every frame via additional_targets, so the
# sequence doesn't flicker frame-to-frame. FRAME_PIPELINE holds effects that genuinely differ each
# frame (motion blur, sensor noise) and is applied independently per frame.
SEQ_PIPELINE = A.Compose([
  A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
  A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-20, 20), val_shift_limit=0, p=0.5),
  A.OneOf([
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.3),
  ], p=0.2),
  A.Defocus(radius=(1, 5), p=0.05),
  A.OneOf([
    A.PlanckianJitter(mode="cied"),
    A.PlanckianJitter(),
  ], p=0.5),
], additional_targets={f"image{t}": "image" for t in range(1, T)})
# Per-frame sensor noise (motion blur is handled separately, size-aware, below).
FRAME_NOISE = A.Compose([
  A.OneOf([
    A.GaussNoise(std_range=(0.05, 0.2), p=0.5),
    A.ISONoise(p=0.5),
  ], p=0.25),
])

# Size-aware motion blur. A fixed up-to-17px kernel smears a small (<~20px) plate into mush while the
# label still carries pixel-precise corners — unlearnable supervision. So cap the kernel to a
# fraction of the target plate's apparent size: large plates still get the full blur, tiny plates
# stay legible. Negatives (no target) get the full blur.
FRAME_BLUR_MAX = 17     # max motion-blur kernel (px)
FRAME_BLUR_P = 0.5
BLUR_SIZE_FRAC = 0.6    # cap kernel to this fraction of the target's apparent (screw-quad) size

def _target_apparent_px(class_id, corners_8):
  if class_id <= 0: return None
  k = np.array(corners_8, dtype=np.float32).reshape(4, 2) * np.array([IMG_W, IMG_H], dtype=np.float32)
  return max(float(np.linalg.norm(k[i] - k[(i + 1) % 4])) for i in range(4))

def _size_aware_motion_blur(img, target_px):
  if random.random() > FRAME_BLUR_P: return img
  cap = FRAME_BLUR_MAX if target_px is None else int(min(FRAME_BLUR_MAX, BLUR_SIZE_FRAC * target_px))
  if cap % 2 == 0: cap -= 1  # MotionBlur needs an odd kernel
  if cap < 3: return img     # plate too small to blur without erasing it
  return A.MotionBlur(blur_limit=(3, cap), p=1.0)(image=img)["image"]

# Label format: [class_id, c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y, has_class, has_corners, target_color] — 12 values
# corners are normalized to image dims and may fall outside [0, 1] for partial plates;
# target_color is 0=red or 1=blue (the color the bot is hunting at this sample).

def load_single_file(file) -> dict[str, bytes]:
  if file.startswith("fake:"):
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    class_id = 0
    corners_8 = [0.0] * 8
    target_color_id = 0
  elif file.startswith("syn:"):
    img, class_id, corners_8, target_color_id = generate_sample(file)
  else:
    raise ValueError("unknown file type")

  img = SEQ_PIPELINE(image=img)["image"]
  img = _size_aware_motion_blur(img, _target_apparent_px(class_id, corners_8))
  img = FRAME_NOISE(image=img)["image"]

  has_class = 1.0
  has_corners = 1.0 if class_id > 0 else 0.0
  label = np.array([class_id] + corners_8 + [has_class, has_corners, target_color_id], dtype=np.float32)
  return {
    "x": img.tobytes(),
    "y": label.tobytes(),
  }

def load_sequence_file(file) -> dict[str, bytes]:
  if file.startswith("fake:"):
    imgs = np.zeros((T, IMG_H, IMG_W, 3), dtype=np.uint8)
    class_id = 0
    corners_8 = [0.0] * 8
    target_color_id = 0
  elif file.startswith("syn:"):
    frame_list, class_id, corners_8, target_color_id = generate_sequence(file, T=T)
    imgs = np.stack(frame_list, axis=0)  # (T, IMG_H, IMG_W, 3)
    # Sequence-consistent aug: one param sample applied identically to every frame, then per-frame
    # motion/noise on top.
    seq_out = SEQ_PIPELINE(image=imgs[0], **{f"image{t}": imgs[t] for t in range(1, T)})
    imgs[0] = seq_out["image"]
    for t in range(1, T): imgs[t] = seq_out[f"image{t}"]
    tgt_px = _target_apparent_px(class_id, corners_8)  # label is the last frame; size-cap blur to it
    for t in range(T):
      imgs[t] = _size_aware_motion_blur(imgs[t], tgt_px)
      imgs[t] = FRAME_NOISE(image=imgs[t])["image"]
  else:
    raise ValueError("unknown file type")

  has_class = 1.0
  has_corners = 1.0 if class_id > 0 else 0.0
  label = np.array([class_id] + corners_8 + [has_class, has_corners, target_color_id], dtype=np.float32)
  return {
    "x": np.ascontiguousarray(imgs).tobytes(),
    "y": label.tobytes(),
  }

def get_train_files():
  syn_files = [
    "syn:1_blank",
    "syn:3_blank",
    "syn:4_blank",
    "syn:5_blank",

    "syn:1_red",
    "syn:2_red",
    "syn:3_red",
    "syn:4_red",
    "syn:5_red",
    "syn:6_red",

    "syn:1_blue",
    "syn:2_blue",
    "syn:3_blue",
    "syn:4_blue",
    "syn:5_blue",
    "syn:6_blue",
  ] * 32000

  fake_files = [
    "fake:"
  ] * 32000

  if getenv("FAKEFILES", 0):
    return fake_files
  if getenv("SYNFILES", 0):
    return syn_files
  return syn_files + fake_files

def run():
  cv2.setNumThreads(0)
  cv2.ocl.setUseOpenCL(False)

  DataloaderProc(load_sequence_file).start()

if __name__ == "__main__":
  files = get_train_files()
  for file in files[::-1]:
    data = load_sequence_file(file)
    imgs = np.frombuffer(data["x"], dtype=np.uint8).copy()
    imgs = imgs.reshape((T, IMG_H, IMG_W, 3))
    anno = np.frombuffer(data["y"], dtype=np.float32)
    class_id = int(anno[0])
    corners = anno[1:9].reshape(4, 2)
    target_color = "red" if int(anno[11]) == 0 else "blue"
    print(f"class_id={class_id}, corners={corners.tolist()}, has_class={anno[9]}, has_corners={anno[10]}, target={target_color}")
    for t in range(T):
      img = cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR)
      cv2.putText(img, f"t={t}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      if t == T - 1 and class_id > 0:
        for i, (cx, cy) in enumerate(corners):
          px, py = int(cx * IMG_W), int(cy * IMG_H)
          cv2.circle(img, (px, py), 4, (0, 255, 255), -1)
          cv2.putText(img, str(i), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
      cv2.imshow("img", img)
      key = cv2.waitKey(0)
      if key == ord("q"):
        break
    if key == ord("q"):
      break
