import glob, random, json, math

import cv2
import albumentations as A
import numpy as np

from ...common import BASE_PATH
from ..common import IMG_H, IMG_W
from ...system.core.logging import logger

# --- Procedural background generation ---
def generate_procedural_background(h=IMG_H, w=IMG_W):
  kind = random.choice(["solid", "gradient", "noise", "rectangles"])
  if kind == "solid":
    color = [random.randint(0, 255) for _ in range(3)]
    img = np.full((h, w, 3), color, dtype=np.uint8)
  elif kind == "gradient":
    c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    angle = random.uniform(0, 2 * math.pi)
    # project pixel coords onto gradient direction
    ys, xs = np.mgrid[0:h, 0:w]
    t = (xs * math.cos(angle) + ys * math.sin(angle))
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    img = (c1[None, None, :] * (1 - t[:, :, None]) + c2[None, None, :] * t[:, :, None]).astype(np.uint8)
  elif kind == "noise":
    base = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    noise = np.random.randn(h, w, 3).astype(np.float32) * random.uniform(10, 60)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    ksize = random.choice([3, 5, 7])
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)
  else:  # rectangles
    color = [random.randint(0, 255) for _ in range(3)]
    img = np.full((h, w, 3), color, dtype=np.uint8)
    for _ in range(random.randint(3, 15)):
      rc = tuple(random.randint(0, 255) for _ in range(3))
      x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
      y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
      cv2.rectangle(img, (x1, y1), (x2, y2), rc, -1)
  return img

# --- Highlight desaturation (sensor clipping sim) ---
def apply_highlight_desat(img, p=0.5):
  if random.random() > p:
    return img
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
  thresh = random.uniform(150, 220)
  desat_factor = random.uniform(0.2, 0.6)
  v = hsv[:, :, 2]
  mask = v > thresh
  if mask.any():
    blend = np.clip((v[mask] - thresh) / (255.0 - thresh + 1e-6), 0, 1)
    hsv[:, :, 1][mask] *= (1.0 - blend) + blend * desat_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
  return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

PLATE_PIPELINE = A.Compose([
  A.RandomScale(scale_limit=(0.05-1, 0.5-1), p=1),
  A.Perspective(scale=(0.05, 0.2), keep_size=True, fit_output=True, p=1),
  A.SafeRotate(limit=(-90, 90), p=0.5),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
PLATE_PIPELINE_2 = A.Compose([
  A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.25),
  A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.25),
  # Heavier plate motion blur: spin-top / fast-translate plates routinely smear by 11-21 px
  A.MotionBlur(blur_limit=(5, 21), p=0.7),
  A.Downscale(scale_range=(0.25, 0.75), interpolation_pair={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_LINEAR}, p=0.2),
])
BACKGROUND_PIPELINE = A.Compose([
  A.RandomResizedCrop(size=(IMG_H, IMG_W), scale=(0.1, 1.0), ratio=(1.9, 2.1), p=1),
])
RESIZE_PIPELINE = A.Compose([
  A.LongestMaxSize(max_size=512, p=1),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

def euler_to_rvec(roll_deg, pitch_deg, yaw_deg):
  roll = math.radians(roll_deg)
  pitch = math.radians(pitch_deg)
  yaw = math.radians(yaw_deg)
  Rx = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
  Ry = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
  Rz = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
  R = Rz @ Ry @ Rx
  rvec, _ = cv2.Rodrigues(R)
  return rvec

# Canonical camera — zero distortion, randomized focal length per sample.
# Principal point is the image center (derived from IMG_H/IMG_W).
CANONICAL_CX, CANONICAL_CY = IMG_W / 2, IMG_H / 2

def _make_canonical_camera(fx=None):
  if fx is None:
    fx = random.uniform(300, 400)
  cam = np.array([[fx, 0, CANONICAL_CX],
                  [0, fx, CANONICAL_CY],
                  [0, 0, 1]], dtype=np.float32)
  return cam, np.zeros((1, 5), dtype=np.float32)

PLATE_WIDTH_SMALL, PLATE_HEIGHT_SMALL = 135, 125  # small armor (numbers 2-6), mm
PLATE_WIDTH_LARGE, PLATE_HEIGHT_LARGE = 230, 127  # large armor (number 1 / hero), mm

def plate_dims(number:int) -> tuple[float, float]:
  return (PLATE_WIDTH_LARGE, PLATE_HEIGHT_LARGE) if number == 1 else (PLATE_WIDTH_SMALL, PLATE_HEIGHT_SMALL)

def project_plate(plate, plate_alpha, plate_kps_local, img, x, y, z, rx, ry, rz, plate_w, plate_h, fx=None):
  """Project plate onto image at explicit pose. Returns projected keypoints (N,2) in image pixel
  coords, or None on failure. Composits plate into img in-place."""
  source_points = np.array([
    [0, 0],
    [plate.shape[1], 0],
    [plate.shape[1], plate.shape[0]],
    [0, plate.shape[0]],
  ], dtype=np.float32)

  plate_points_3d = np.array([
    [-plate_w / 2, -plate_h / 2, 0],
    [ plate_w / 2, -plate_h / 2, 0],
    [ plate_w / 2,  plate_h / 2, 0],
    [-plate_w / 2,  plate_h / 2, 0]
  ], dtype=np.float32)

  tvec = np.array([[x], [y], [z]], dtype=np.float32)
  rvec = euler_to_rvec(rx, ry, rz)

  cam_matrix, dist_coeffs = _make_canonical_camera(fx)
  projected_points_2d, _ = cv2.projectPoints(
    plate_points_3d,
    rvec,
    tvec,
    cam_matrix,
    dist_coeffs,
  )

  H, _ = cv2.findHomography(source_points, projected_points_2d)
  if H is None: return None

  warped_plate = cv2.warpPerspective(plate, H, (IMG_W, IMG_H))
  mask = (plate_alpha > 0.5).astype(np.uint8) * 255
  mask = cv2.warpPerspective(mask, H, (IMG_W, IMG_H))
  cv2.copyTo(warped_plate, mask, img)

  kps_array = np.array(plate_kps_local, dtype=np.float32).reshape(-1, 1, 2)
  projected_kps = cv2.perspectiveTransform(kps_array, H).reshape(-1, 2)
  return projected_kps

def random_plate(plate, plate_alpha, plate_kps_local, img, plate_w, plate_h, fx=None):
  """Sample random pose, project plate, return (projected_kps, pose_tuple)."""
  x, y, z = random.uniform(-400, 400), random.uniform(-200, 200), random.uniform(10, 2000)
  rx, ry, rz = random.uniform(-5, 5), random.uniform(-60, 60), random.uniform(-1, 1)
  kps = project_plate(plate, plate_alpha, plate_kps_local, img, x, y, z, rx, ry, rz, plate_w, plate_h, fx)
  return kps, (x, y, z), (rx, ry, rz)

def _normalize_and_validate_corners(projected_kps):
  """Returns (corners_8, has_corners) where corners_8 is a flat list of 8 normalized coords in [0,1]
  and has_corners is 0.0 if the plate is mostly out of frame or projection was degenerate."""
  if projected_kps is None or not np.all(np.isfinite(projected_kps)):
    return [0.0] * 8, 0.0
  center = projected_kps.mean(axis=0)
  margin = 50
  if (center[0] < -margin or center[0] > IMG_W + margin or
      center[1] < -margin or center[1] > IMG_H + margin):
    return [0.0] * 8, 0.0
  corners_norm = projected_kps / np.array([IMG_W, IMG_H], dtype=np.float32)
  corners_norm = np.clip(corners_norm, 0.0, 1.0)
  return corners_norm.flatten().tolist(), 1.0

# Unified class encoding: (color, number) → class_id
# 0: no plate
# blank: 1=1_blank, 2=3_blank, 3=4_blank, 4=5_blank
# red: 5=1_red, 6=2_red, 7=3_red, 8=4_red, 9=5_red, 10=6_red
# blue: 11=1_blue, 12=2_blue, 13=3_blue, 14=4_blue, 15=5_blue, 16=6_blue
_BLANK_MAP = {1: 1, 3: 2, 4: 3, 5: 4}
_RED_MAP   = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10}
_BLUE_MAP  = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16}

def encode_unified_class(color_str:str, number:int) -> int:
  match color_str:
    case "blank": return _BLANK_MAP.get(number, 0)
    case "red":   return _RED_MAP.get(number, 0)
    case "blue":  return _BLUE_MAP.get(number, 0)
    case _:       return 0

plate_images = {}
plate_corners = {}
background_images = []
def generate_sample(file) -> tuple[cv2.Mat, int, list[float]]:
  """Returns (image, class_id, corners_8) where corners_8 = [c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y]
  normalized to [0,1]. If the plate is off-frame, class_id=0 and corners_8 are zeros."""
  global plate_images, plate_corners, background_images

  plate = file.split(":")[1]
  number = int(plate.split("_")[0])
  color = plate.split("_")[1]
  if plate not in plate_images:
    logger.debug(f"loading plate {plate}")
    plate_img = cv2.imread(str(BASE_PATH / "armor_plate" / f"{plate}.png"), cv2.IMREAD_UNCHANGED)
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2RGBA)

    with open(str(BASE_PATH / "armor_plate" / f"{plate}.json"), "r") as f:
      keypoints = json.load(f)
    resized = RESIZE_PIPELINE(image=plate_img, keypoints=keypoints)
    plate_images[plate] = resized["image"]
    plate_corners[plate] = resized["keypoints"]

  if len(background_images) == 0:
    bg_files = glob.glob(str(BASE_PATH / "background" / "*"))
    logger.debug(f"loading {len(bg_files)} background images")
    for f in bg_files:
      bg_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
      bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
      background_images.append(bg_img)

  raw_plate = plate_images[plate]
  kps_local = plate_corners[plate]

  # Choose background: 50/50 real vs procedural if real available, else procedural
  use_procedural = len(background_images) == 0 or random.random() < 0.5
  if use_procedural:
    img = generate_procedural_background()
  else:
    raw_background = random.choice(background_images)
    img = BACKGROUND_PIPELINE(image=raw_background)["image"]

  # Apply hue/saturation jitter to background BEFORE plate compositing
  img = A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.7)(image=img)["image"]

  plate_rgba, plate_alpha = raw_plate[:, :, :3], raw_plate[:, :, 3]
  plate_rgba = PLATE_PIPELINE_2(image=plate_rgba)["image"]
  plate_w, plate_h = plate_dims(number)
  projected_kps, _, _ = random_plate(plate_rgba, plate_alpha, kps_local, img, plate_w, plate_h, fx=None)

  # Simulate sensor highlight desaturation
  img = apply_highlight_desat(img)

  corners_8, has_corners = _normalize_and_validate_corners(projected_kps)
  if has_corners == 0.0:
    class_id = 0
  else:
    class_id = encode_unified_class(color, number)

  return img, class_id, corners_8

def _make_background(fx=None):
  """Create a background image (shared across sequence frames)."""
  global background_images
  if len(background_images) == 0:
    bg_files = glob.glob(str(BASE_PATH / "background" / "*"))
    logger.debug(f"loading {len(bg_files)} background images")
    for f in bg_files:
      bg_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
      bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
      background_images.append(bg_img)

  use_procedural = len(background_images) == 0 or random.random() < 0.5
  if use_procedural:
    return generate_procedural_background()
  else:
    raw_background = random.choice(background_images)
    return BACKGROUND_PIPELINE(image=raw_background)["image"]

def _load_plate(plate_name):
  """Load and cache a plate image. Returns (plate_rgba, plate_alpha, number, color)."""
  global plate_images, plate_corners
  number = int(plate_name.split("_")[0])
  color = plate_name.split("_")[1]
  if plate_name not in plate_images:
    logger.debug(f"loading plate {plate_name}")
    plate_img = cv2.imread(str(BASE_PATH / "armor_plate" / f"{plate_name}.png"), cv2.IMREAD_UNCHANGED)
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2RGBA)
    with open(str(BASE_PATH / "armor_plate" / f"{plate_name}.json"), "r") as f:
      keypoints = json.load(f)
    resized = RESIZE_PIPELINE(image=plate_img, keypoints=keypoints)
    plate_images[plate_name] = resized["image"]
    plate_corners[plate_name] = resized["keypoints"]
  raw_plate = plate_images[plate_name]
  plate_rgba = raw_plate[:, :, :3]
  plate_alpha = raw_plate[:, :, 3]
  return plate_rgba, plate_alpha, number, color

def generate_sequence(file, T=4):
  """Generate a T-frame temporal sequence with physics-based plate motion.

  Returns: (images: list[ndarray(256,512,3)], class_id: int, corners_8: list[float])
  where corners_8 is the LAST frame's projected plate corners normalized to [0,1].
  If the last frame has no visible plate, class_id=0 and corners_8 are zeros.
  """
  plate_name = file.split(":")[1]
  plate_rgba_raw, plate_alpha, number, color = _load_plate(plate_name)
  kps_local = plate_corners[plate_name]
  class_id = encode_unified_class(color, number)
  plate_w, plate_h = plate_dims(number)

  # Shared sequence parameters
  fx = random.uniform(300, 400)
  bg_base = _make_background()
  bg_hue_aug = A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.7)

  # Select sequence type (appearance/spin_jump need T>=2)
  if T >= 2:
    seq_type = random.choices(
      ["continuous", "appearance", "spin_jump", "static"],
      weights=[50, 20, 15, 15], k=1
    )[0]
  else:
    seq_type = random.choices(
      ["continuous", "static"],
      weights=[50, 15], k=1
    )[0]

  # Sample spin + translation physics parameters
  # Robot center position (final frame)
  cx = random.uniform(-400, 400)
  cy = random.uniform(-200, 200)
  cz = random.uniform(200, 2000)

  # Plate offset from robot center
  r = random.uniform(150, 250)

  # Spin
  theta_final = random.uniform(0, 360)  # final spin angle (degrees)
  omega = random.uniform(-15, 15)  # deg/frame

  # Drive velocity (mm/frame)
  dvx = random.uniform(-15, 15)
  dvy = random.uniform(-3, 3)
  dvz = random.uniform(-30, 30)

  # Base rotation
  base_rx = random.uniform(-5, 5)
  base_ry = random.uniform(-30, 30)
  base_rz = random.uniform(-1, 1)

  # Rotation wobble (deg/frame)
  wrx = random.uniform(-0.5, 0.5)
  wrz = random.uniform(-0.5, 0.5)

  if seq_type == "static":
    omega = 0
    dvx, dvy, dvz = 0, 0, 0
    wrx, wrz = 0, 0

  # Compute per-frame poses (t=0 is oldest, t=T-1 is most recent)
  # Step backward from final frame
  poses = []  # list of (x, y, z, rx, ry, rz) per frame
  for t in range(T):
    dt = t - (T - 1)  # dt <= 0 for past frames, 0 for most recent
    theta_t = math.radians(theta_final + omega * dt)
    x_t = cx + r * math.sin(theta_t) + dvx * dt
    y_t = cy + dvy * dt
    z_t = cz + r * math.cos(theta_t) + dvz * dt
    ry_t = base_ry + (theta_final + omega * dt)
    rx_t = base_rx + wrx * dt
    rz_t = base_rz + wrz * dt
    poses.append((x_t, y_t, z_t, rx_t, ry_t, rz_t))

  # Handle sequence types
  no_plate_frames = set()  # frames with no plate visible
  jump_frame = None

  if seq_type == "appearance":
    # First K frames have no plate
    K = random.randint(1, T - 1)
    no_plate_frames = set(range(K))

  elif seq_type == "spin_jump":
    # At frame K, spin angle jumps by ~90 degrees (next physical plate)
    K = random.randint(1, T - 1)
    jump_frame = K
    jump_angle = random.choice([90, -90])
    # Recompute poses for frames before the jump with offset angle
    for t in range(K):
      dt = t - (T - 1)
      theta_t = math.radians(theta_final + omega * dt + jump_angle)
      x_t = cx + r * math.sin(theta_t) + dvx * dt
      y_t = cy + dvy * dt
      z_t = cz + r * math.cos(theta_t) + dvz * dt
      ry_t = base_ry + (theta_final + omega * dt + jump_angle)
      rx_t = base_rx + wrx * dt
      rz_t = base_rz + wrz * dt
      poses[t] = (x_t, y_t, z_t, rx_t, ry_t, rz_t)

  elif seq_type == "static":
    # All frames same pose + small noise
    base_pose = poses[-1]
    for t in range(T):
      poses[t] = (
        base_pose[0] + random.gauss(0, 2),
        base_pose[1] + random.gauss(0, 1),
        base_pose[2] + random.gauss(0, 5),
        base_pose[3] + random.gauss(0, 0.2),
        base_pose[4] + random.gauss(0, 0.3),
        base_pose[5] + random.gauss(0, 0.1),
      )

  # Per-frame augmentation pipeline (sensor noise + camera/scene motion blur on the composited frame)
  per_frame_aug = A.Compose([
    A.OneOf([
      A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
      A.ISONoise(p=0.5),
    ], p=0.3),
    A.MotionBlur(blur_limit=(5, 15), p=0.4),
  ])

  # Mark frames where plate faces away from camera (|ry| > 90°) as invisible
  for t in range(T):
    ry_t = poses[t][4]
    ry_norm = ((ry_t + 180) % 360) - 180  # normalize to [-180, 180]
    if abs(ry_norm) > 90:
      no_plate_frames.add(t)

  # Generate images
  images = []
  final_kps = None
  plate_rgba_aug = PLATE_PIPELINE_2(image=plate_rgba_raw)["image"]

  for t in range(T):
    img = bg_base.copy()
    img = bg_hue_aug(image=img)["image"]

    if t not in no_plate_frames:
      x_t, y_t, z_t, rx_t, ry_t, rz_t = poses[t]
      # Small per-frame measurement noise
      x_t += random.gauss(0, 1)
      y_t += random.gauss(0, 0.5)
      z_t += random.gauss(0, 2)
      rx_t += random.gauss(0, 0.1)
      ry_t += random.gauss(0, 0.2)
      rz_t += random.gauss(0, 0.05)
      kps_t = project_plate(plate_rgba_aug, plate_alpha, kps_local, img, x_t, y_t, z_t, rx_t, ry_t, rz_t, plate_w, plate_h, fx)
      if t == T - 1:
        final_kps = kps_t

    img = apply_highlight_desat(img)
    img = per_frame_aug(image=img)["image"]
    images.append(img)

  # Label is for the most recent frame (T-1)
  if (T - 1) in no_plate_frames:
    class_id = 0
    corners_8 = [0.0] * 8
  else:
    corners_8, has_corners = _normalize_and_validate_corners(final_kps)
    if has_corners == 0.0:
      class_id = 0

  return images, class_id, corners_8
