import glob, random, json, math

import cv2
import albumentations as A
import numpy as np

from ...common import BASE_PATH
from ..common import IMG_H, IMG_W, BIN_LO, BIN_HI, CANONICAL_CX, CANONICAL_CY, CANONICAL_FX_FY
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

# At inference camerad always warps to the single CANONICAL_FX_FY, so we train at that exact focal.
# A small jitter hedges against the real warp not being a perfect canonical (calibration error,
# distortion residual); set to 0.0 for dead-fixed.
FOCAL_JITTER = 0.05
def _sample_focal() -> float:
  return CANONICAL_FX_FY * random.uniform(1 - FOCAL_JITTER, 1 + FOCAL_JITTER)

# Canonical camera — zero distortion, focal sampled near CANONICAL_FX_FY per sample. CANONICAL_CX/CY
# come from autoaim.common (the shared canonical-pinhole convention).
def _make_canonical_camera(fx=None):
  if fx is None:
    fx = _sample_focal()
  cam = np.array([[fx, 0, CANONICAL_CX],
                  [0, fx, CANONICAL_CY],
                  [0, 0, 1]], dtype=np.float32)
  return cam, np.zeros((1, 5), dtype=np.float32)

PLATE_WIDTH_SMALL, PLATE_HEIGHT_SMALL = 135, 125  # small armor (numbers 2-6), mm
PLATE_WIDTH_LARGE, PLATE_HEIGHT_LARGE = 230, 127  # large armor (number 1 / hero), mm

def plate_dims(number:int) -> tuple[float, float]:
  return (PLATE_WIDTH_LARGE, PLATE_HEIGHT_LARGE) if number == 1 else (PLATE_WIDTH_SMALL, PLATE_HEIGHT_SMALL)

def project_plate(plate, plate_alpha, plate_kps_local, img, x, y, z, rx, ry, rz, plate_w, plate_h, fx=None):
  """Project plate onto image at explicit pose. Returns (projected_kps, H) where projected_kps
  is (N,2) in image pixel coords and H is the homography from texture to image space, or None
  on failure. Composits plate into img in-place."""
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
  return projected_kps, H

def _apply_emissive_leds(img, plate_rgb, plate_alpha, color_str, H, velocity_px=None):
  """Build an emissive LED layer from color-saturated texture pixels, warp into image space,
  and additively blend with bloom + optional directional smear. Mutates img in place.

  The plate PNGs carry strong red/blue LED colors already — we mask them by HSV hue+sat+val,
  multiply through the texture (preserves any gradient/variation), warp with the plate's
  homography H, then bloom and (optionally) smear along the per-frame image-space velocity.
  Sensor highlight clipping is handled by the existing apply_highlight_desat pass downstream.
  """
  if color_str not in ("red", "blue"): return  # blank plates have no lit LEDs

  hsv = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
  h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
  if color_str == "red":
    hue_mask = ((h < 15) | (h > 165)).astype(np.float32)  # OpenCV hue 0-179, red wraps
  else:
    hue_mask = ((h > 100) & (h < 130)).astype(np.float32)
  sat_mask = np.clip((s - 50) / 100.0, 0, 1)
  val_mask = np.clip((v - 100) / 100.0, 0, 1)
  alpha_ok = (plate_alpha > 128).astype(np.float32)
  mask = hue_mask * sat_mask * val_mask * alpha_ok
  if mask.max() < 0.1: return  # nothing colored enough to bloom

  # Over-saturate the LED region so the bloom reads as hot, then warp into image space
  boost = random.uniform(1.5, 3.0)
  led_layer_local = plate_rgb.astype(np.float32) * mask[..., None] * boost
  warped_led = cv2.warpPerspective(led_layer_local, H, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR)

  # LED-only motion smear: real spin-top plates trail the body in long colored arcs because
  # the LEDs are the only thing bright enough to register through the smear. The whole-frame
  # MotionBlur pass downstream is uniform, so we add this here on just the LED layer.
  if velocity_px is not None:
    vx, vy = float(velocity_px[0]), float(velocity_px[1])
    speed = math.hypot(vx, vy)
    if speed > 1.5:
      smear_len = int(np.clip(speed * random.uniform(0.5, 1.5), 5, 51))
      if smear_len % 2 == 0: smear_len += 1
      mid = smear_len // 2
      angle = math.atan2(vy, vx)
      kernel = np.zeros((smear_len, smear_len), dtype=np.float32)
      for k in range(-mid, mid + 1):
        kx = int(round(mid + k * math.cos(angle)))
        ky = int(round(mid + k * math.sin(angle)))
        if 0 <= kx < smear_len and 0 <= ky < smear_len:
          kernel[ky, kx] = 1.0
      kernel /= kernel.sum() + 1e-6
      warped_led = cv2.filter2D(warped_led, -1, kernel)

  # Wide halo + tight core, both additively blended
  bloom_sigma = random.uniform(1.0, 8.0)
  bloom_intensity = random.uniform(0.5, 1.5)
  bloom = cv2.GaussianBlur(warped_led, (0, 0), bloom_sigma)
  core = cv2.GaussianBlur(warped_led, (0, 0), 1.0)
  out = img.astype(np.float32) + bloom_intensity * (core * 1.0 + bloom * 0.5)
  img[...] = np.clip(out, 0, 255).astype(np.uint8)

def _estimate_plate_center_px(x, y, z, rx, ry, rz, fx):
  """Project the plate origin (0,0,0) to image pixel coords. Used for frame-to-frame velocity
  estimation so we can apply directional LED smear before calling project_plate."""
  cam, dist = _make_canonical_camera(fx)
  pt, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), euler_to_rvec(rx, ry, rz),
                            np.array([[x], [y], [z]], dtype=np.float32), cam, dist)
  return pt[0, 0]

def random_plate(plate, plate_alpha, plate_kps_local, img, plate_w, plate_h, fx=None):
  """Sample random pose, project plate, return (projected_kps, pose_tuple, rot_tuple, H)."""
  # log-uniform z so apparent plate size is roughly uniformly distributed; linear-z over-samples
  # far plates because apparent size ∝ 1/z. z_lo=200mm because below that a near plate's apparent
  # width exceeds the trainable [-0.5, 1.5] bin range at CANONICAL_FX_FY and the sample gets
  # rejected; z_hi=6000mm covers long-range (~6m) targets — a 135mm plate there is ~16px wide.
  x, y = random.uniform(-400, 400), random.uniform(-200, 200)
  z = math.exp(random.uniform(math.log(200), math.log(6000)))
  # rz (in-plane roll) widened so the model sees plates tilted by gimbal jitter, robot tilt, and
  # off-axis viewing — previously locked to ±1° which left the head fragile to any image rotation.
  rx, ry, rz = random.uniform(-5, 5), random.uniform(-60, 60), random.uniform(-30, 30)
  ret = project_plate(plate, plate_alpha, plate_kps_local, img, x, y, z, rx, ry, rz, plate_w, plate_h, fx)
  if ret is None: return None, (x, y, z), (rx, ry, rz), None
  kps, H = ret
  return kps, (x, y, z), (rx, ry, rz), H

def _normalize_and_validate_corners(projected_kps):
  """Returns (corners_8, has_corners) where corners_8 is a flat list of 8 normalized coords (NOT
  clipped — partial plates carry corners in [BIN_LO, BIN_HI] so the model can learn off-frame
  positions for PnP). has_corners is 0.0 if projection was degenerate or any corner falls outside
  the trainable bin range — DFL targets must stay representable."""
  if projected_kps is None or not np.all(np.isfinite(projected_kps)):
    return [0.0] * 8, 0.0
  corners_norm = projected_kps / np.array([IMG_W, IMG_H], dtype=np.float32)
  if (corners_norm < BIN_LO).any() or (corners_norm > BIN_HI).any():
    return [0.0] * 8, 0.0
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

# Multi-plate scene generation. ~20% of samples include 1-2 distractor plates so the model
# learns to disambiguate (pick the target_color plate closest to frame center) instead of
# locking onto whichever plate it sees first.
MULTI_PLATE_PROB = 0.2
ALL_PLATE_NUMBERS = [1, 2, 3, 4, 5, 6]
BLANK_PLATE_NUMBERS = [1, 3, 4, 5]
TARGET_COLORS = ("red", "blue")

def _random_distractor_plate(seed_color:str) -> str:
  """Pick a distractor plate name. Distribution: 60% same color (enemy teammate),
  30% opposite color (our own bot in frame), 10% blank (disabled plate)."""
  r = random.random()
  if r < 0.6 and seed_color in TARGET_COLORS:
    color = seed_color
  elif r < 0.9:
    color = "blue" if seed_color == "red" else ("red" if seed_color == "blue" else random.choice(TARGET_COLORS))
  else:
    color = "blank"
  number = random.choice(BLANK_PLATE_NUMBERS if color == "blank" else ALL_PLATE_NUMBERS)
  return f"{number}_{color}"

def _build_scene_plates(seed_plate_name:str) -> list[str]:
  """Plates to place in a scene. Seed first; with probability MULTI_PLATE_PROB also adds
  1-2 distractors. Duplicates de-duped."""
  plates = [seed_plate_name]
  if random.random() < MULTI_PLATE_PROB:
    seed_color = seed_plate_name.split("_")[1]
    for _ in range(random.randint(1, 2)):
      distractor = _random_distractor_plate(seed_color)
      if distractor not in plates: plates.append(distractor)
  return plates

def _select_target(projections, target_color:str):
  """projections: list of (color, number, projected_kps, center_px). Returns the projection
  matching target_color with the smallest distance from its center to the frame center,
  or None if no candidate. Blanks are never selected (caller filters by target_color)."""
  candidates = [p for p in projections if p[0] == target_color]
  if not candidates: return None
  fcx, fcy = IMG_W / 2.0, IMG_H / 2.0
  return min(candidates, key=lambda p: (p[3][0] - fcx) ** 2 + (p[3][1] - fcy) ** 2)

plate_images = {}
plate_corners = {}
background_images = []
def generate_sample(file, target_color:str|None=None) -> tuple[cv2.Mat, int, list[float], int]:
  """Returns (image, class_id, corners_8, target_color_id).
  - target_color: "red" or "blue" — color the bot is hunting. None → random 50/50.
  - target_color_id: 0=red, 1=blue.
  - Label is the closest-to-center plate matching target_color (blanks never targeted). If no
    matching plate is centrally placed, class_id=0 and corners_8=zeros.
  - 20% of samples include 1-2 distractor plates (mixed colors) so the model learns to pick
    the right one instead of any visible plate."""
  global plate_images, plate_corners, background_images

  if target_color is None: target_color = random.choice(TARGET_COLORS)
  target_color_id = 0 if target_color == "red" else 1

  seed_plate_name = file.split(":")[1]
  scene_plate_names = _build_scene_plates(seed_plate_name)

  # Lazy-load any plates we haven't seen yet
  for name in scene_plate_names:
    if name not in plate_images:
      logger.debug(f"loading plate {name}")
      plate_img = cv2.imread(str(BASE_PATH / "armor_plate" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
      plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2RGBA)
      with open(str(BASE_PATH / "armor_plate" / f"{name}.json"), "r") as f:
        keypoints = json.load(f)
      resized = RESIZE_PIPELINE(image=plate_img, keypoints=keypoints)
      plate_images[name] = resized["image"]
      plate_corners[name] = resized["keypoints"]

  if len(background_images) == 0:
    bg_files = glob.glob(str(BASE_PATH / "background" / "*"))
    logger.debug(f"loading {len(bg_files)} background images")
    for f in bg_files:
      bg_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
      bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
      background_images.append(bg_img)

  # Background (shared across all plates in the scene)
  use_procedural = len(background_images) == 0 or random.random() < 0.5
  if use_procedural:
    img = generate_procedural_background()
  else:
    raw_background = random.choice(background_images)
    img = BACKGROUND_PIPELINE(image=raw_background)["image"]
  img = A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.7)(image=img)["image"]

  # Place each plate at a random pose; collect projections for target selection
  fx = _sample_focal()  # one camera per scene
  projections = []  # list of (color, number, projected_kps, center_px)
  for name in scene_plate_names:
    raw_plate = plate_images[name]
    kps_local = plate_corners[name]
    number = int(name.split("_")[0])
    color = name.split("_")[1]
    plate_w, plate_h = plate_dims(number)
    plate_rgba = PLATE_PIPELINE_2(image=raw_plate[:, :, :3])["image"]
    plate_alpha = raw_plate[:, :, 3]

    projected_kps, _, _, H = random_plate(plate_rgba, plate_alpha, kps_local, img, plate_w, plate_h, fx=fx)
    if projected_kps is not None and H is not None:
      _apply_emissive_leds(img, plate_rgba, plate_alpha, color, H)
      projections.append((color, number, projected_kps, projected_kps.mean(axis=0)))

  img = apply_highlight_desat(img)

  target = _select_target(projections, target_color)
  if target is None:
    return img, 0, [0.0] * 8, target_color_id
  color, number, target_kps, _ = target
  corners_8, has_corners = _normalize_and_validate_corners(target_kps)
  if has_corners == 0.0:
    return img, 0, [0.0] * 8, target_color_id
  return img, encode_unified_class(color, number), corners_8, target_color_id

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

def _sample_plate_dynamics():
  """Sample per-plate motion parameters (shared kinematics regardless of seq_type)."""
  return {
    "cx": random.uniform(-400, 400),
    "cy": random.uniform(-200, 200),
    "cz": math.exp(random.uniform(math.log(200), math.log(2000))),
    "r":  random.uniform(150, 250),
    "theta_final": random.uniform(0, 360),
    "omega": random.uniform(-15, 15),
    "dvx": random.uniform(-15, 15),
    "dvy": random.uniform(-3, 3),
    "dvz": random.uniform(-30, 30),
    "base_rx": random.uniform(-5, 5),
    "base_ry": random.uniform(-30, 30),
    "base_rz": random.uniform(-30, 30),
    "wrx": random.uniform(-0.5, 0.5),
    "wrz": random.uniform(-2.0, 2.0),
  }

def _compute_plate_poses(dyn, T:int, seq_type:str, jump_frame=None, jump_angle=0):
  """Produce list of per-frame (x, y, z, rx, ry, rz) tuples from a dynamics dict."""
  poses = []
  for t in range(T):
    dt = t - (T - 1)
    angle_offset = jump_angle if (jump_frame is not None and t < jump_frame) else 0
    theta_t = math.radians(dyn["theta_final"] + dyn["omega"] * dt + angle_offset)
    x_t = dyn["cx"] + dyn["r"] * math.sin(theta_t) + dyn["dvx"] * dt
    y_t = dyn["cy"] + dyn["dvy"] * dt
    z_t = dyn["cz"] + dyn["r"] * math.cos(theta_t) + dyn["dvz"] * dt
    ry_t = dyn["base_ry"] + (dyn["theta_final"] + dyn["omega"] * dt + angle_offset)
    rx_t = dyn["base_rx"] + dyn["wrx"] * dt
    rz_t = dyn["base_rz"] + dyn["wrz"] * dt
    poses.append((x_t, y_t, z_t, rx_t, ry_t, rz_t))

  if seq_type == "static":
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
  return poses

def generate_sequence(file, T=4, target_color:str|None=None):
  """Generate a T-frame temporal sequence with physics-based plate motion.

  Returns (images, class_id, corners_8, target_color_id). Labels are for the LAST frame and
  apply to the target plate (closest-to-center plate matching target_color, blanks excluded).

  Multi-plate: ~20% of samples include 1-2 distractor plates. The seed plate (the file: arg)
  uses the chosen seq_type (continuous/appearance/spin_jump/static); distractor plates always
  use continuous motion regardless. If no target_color plate is centrally placed at T-1,
  class_id=0 and corners are zeros.
  """
  if target_color is None: target_color = random.choice(TARGET_COLORS)
  target_color_id = 0 if target_color == "red" else 1

  seed_plate_name = file.split(":")[1]
  scene_plate_names = _build_scene_plates(seed_plate_name)

  # Load all plates; augment each texture once per sequence so its appearance is consistent
  # across the T frames (real video has stable lighting/material per object within ~4 frames).
  plate_infos = []
  for name in scene_plate_names:
    plate_rgba_raw, plate_alpha, number, color = _load_plate(name)
    plate_infos.append({
      "name": name, "number": number, "color": color,
      "plate_rgba": PLATE_PIPELINE_2(image=plate_rgba_raw)["image"],
      "plate_alpha": plate_alpha,
      "kps_local": plate_corners[name],
      "plate_w": plate_dims(number)[0], "plate_h": plate_dims(number)[1],
    })

  fx = _sample_focal()
  bg_base = _make_background()
  bg_hue_aug = A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.7)

  # seq_type applies to the seed (first) plate. Distractors always use continuous motion.
  if T >= 2:
    seq_type = random.choices(["continuous", "appearance", "spin_jump", "static"],
                              weights=[50, 20, 15, 15], k=1)[0]
  else:
    seq_type = random.choices(["continuous", "static"], weights=[50, 15], k=1)[0]

  # Per-plate dynamics
  plate_dynamics = []
  for plate_idx in range(len(plate_infos)):
    eff_seq_type = seq_type if plate_idx == 0 else "continuous"
    dyn = _sample_plate_dynamics()
    if eff_seq_type == "static":
      dyn["omega"], dyn["dvx"], dyn["dvy"], dyn["dvz"], dyn["wrx"], dyn["wrz"] = 0, 0, 0, 0, 0, 0
    plate_dynamics.append((dyn, eff_seq_type))

  # appearance / spin_jump special config applies only to the seed plate
  no_plate_frames_per_plate = [set() for _ in plate_infos]
  jump_frame = None
  jump_angle = 0
  if seq_type == "appearance" and T >= 2:
    K = random.randint(1, T - 1)
    no_plate_frames_per_plate[0] = set(range(K))
  elif seq_type == "spin_jump" and T >= 2:
    K = random.randint(1, T - 1)
    jump_frame = K
    jump_angle = random.choice([90, -90])

  # Pre-compute poses per plate per frame
  plate_poses = []
  for plate_idx, (dyn, eff_seq_type) in enumerate(plate_dynamics):
    jf = jump_frame if plate_idx == 0 else None
    ja = jump_angle if plate_idx == 0 else 0
    plate_poses.append(_compute_plate_poses(dyn, T, eff_seq_type, jf, ja))

  # Mark frames where each plate faces away from camera (|ry| > 90°) as invisible
  for plate_idx, poses in enumerate(plate_poses):
    for t in range(T):
      ry_t = poses[t][4]
      ry_norm = ((ry_t + 180) % 360) - 180  # normalize to [-180, 180]
      if abs(ry_norm) > 90:
        no_plate_frames_per_plate[plate_idx].add(t)

  per_frame_aug = A.Compose([
    A.OneOf([
      A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
      A.ISONoise(p=0.5),
    ], p=0.3),
    A.MotionBlur(blur_limit=(5, 15), p=0.4),
  ])

  images = []
  final_projections = []  # (color, number, projected_kps, center_px) at frame T-1
  prev_centers = [None] * len(plate_infos)

  for t in range(T):
    img = bg_base.copy()
    img = bg_hue_aug(image=img)["image"]

    for plate_idx, info in enumerate(plate_infos):
      if t in no_plate_frames_per_plate[plate_idx]:
        prev_centers[plate_idx] = None
        continue
      # spin_jump only affects the seed plate's velocity continuity
      if plate_idx == 0 and t == jump_frame:
        prev_centers[plate_idx] = None

      x_t, y_t, z_t, rx_t, ry_t, rz_t = plate_poses[plate_idx][t]
      x_t += random.gauss(0, 1)
      y_t += random.gauss(0, 0.5)
      z_t += random.gauss(0, 2)
      rx_t += random.gauss(0, 0.1)
      ry_t += random.gauss(0, 0.2)
      rz_t += random.gauss(0, 0.05)

      curr_center = _estimate_plate_center_px(x_t, y_t, z_t, rx_t, ry_t, rz_t, fx)
      velocity_px = (curr_center - prev_centers[plate_idx]) if prev_centers[plate_idx] is not None else None

      ret = project_plate(info["plate_rgba"], info["plate_alpha"], info["kps_local"], img,
                          x_t, y_t, z_t, rx_t, ry_t, rz_t, info["plate_w"], info["plate_h"], fx)
      if ret is not None:
        kps_t, H_t = ret
        _apply_emissive_leds(img, info["plate_rgba"], info["plate_alpha"], info["color"], H_t, velocity_px=velocity_px)
        prev_centers[plate_idx] = curr_center
        if t == T - 1:
          final_projections.append((info["color"], info["number"], kps_t, curr_center))
      else:
        prev_centers[plate_idx] = None

    img = apply_highlight_desat(img)
    img = per_frame_aug(image=img)["image"]
    images.append(img)

  target = _select_target(final_projections, target_color)
  if target is None:
    return images, 0, [0.0] * 8, target_color_id
  color, number, target_kps, _ = target
  corners_8, has_corners = _normalize_and_validate_corners(target_kps)
  if has_corners == 0.0:
    return images, 0, [0.0] * 8, target_color_id
  return images, encode_unified_class(color, number), corners_8, target_color_id
