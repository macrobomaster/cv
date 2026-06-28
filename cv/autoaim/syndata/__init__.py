import glob, os, random, json, math

import cv2
import albumentations as A
import numpy as np

from ...common import BASE_PATH
from ..common import (IMG_H, IMG_W, BIN_LO, BIN_HI, CANONICAL_CX, CANONICAL_CY, CANONICAL_FX_FY,
                      SCREW_DIMS_SMALL, SCREW_DIMS_LARGE)
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
def apply_highlight_desat(img, p=0.5, thresh=None, desat_factor=None):
  if random.random() > p:
    return img
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
  if thresh is None: thresh = random.uniform(150, 220)
  if desat_factor is None: desat_factor = random.uniform(0.2, 0.6)
  v = hsv[:, :, 2]
  mask = v > thresh
  if mask.any():
    blend = np.clip((v[mask] - thresh) / (255.0 - thresh + 1e-6), 0, 1)
    hsv[:, :, 1][mask] *= (1.0 - blend) + blend * desat_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
  return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

PLATE_PIPELINE_2 = A.Compose([
  A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.25),
  A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.25),
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

# Beyond this yaw (degrees off head-on) a plate is too foreshortened to reliably localize or hit —
# the screw-quad collapses toward a line so PnP and aiming both degrade. Such plates are still
# RENDERED (the model must learn to see them and not fire) but are never picked as the positive
# target, so the sample becomes a negative. Plates past 90° face fully away and aren't drawn at all.
MAX_ENGAGE_YAW = 75.0

# A plate occluded by a nearer plate is unhittable. After depth-ordered compositing we require at
# least this fraction of a candidate's screw-quad to still show its own pixels; below it the plate
# is rendered (partly hidden) but dropped from the target set, so the sample becomes a negative.
MIN_TARGET_VISIBLE = 0.5

# Foreground occluders (poles / our own arm / robot edges / structure) composited in FRONT of all
# plates so the model learns partial-plate robustness. They're marked in the depth buffer too, so a
# target whose screws they cover drops below MIN_TARGET_VISIBLE and becomes a negative.
OCCLUDER_PROB = 0.3   # fraction of sequences with 1-2 foreground occluders
OCCLUDER_ID = -2      # id_buf sentinel for occluder pixels (not -1 background, not any plate idx)

# The seeded plate's yaw at the LABELED (final) frame. A free 0-360 spin lands on the far
# hemisphere ~half the time (faces away -> negative), flooding the set with negatives. With this
# probability the seed's final yaw is instead drawn UNIFORMLY across the engageable band
# [-MAX_ENGAGE_YAW, MAX_ENGAGE_YAW] — flat coverage so angled-but-hittable plates (60°, 75°) are
# sampled as densely as head-on, all as positives. The rest spin freely (full yaw) so sliver/
# back-facing negatives still appear. Raise to skew more positive (fewer yaw-driven negatives).
SEED_ENGAGEABLE_PROB = 0.7

# Depth bias for the plate's apparent size. Image-space position sampling keeps near (large) plates
# that the old metric sampling rejected off-frame, which floods positives with big plates and
# starves small-object learning. DEPTH_FAR_BIAS>1 skews the (log-uniform) depth toward far so small/
# distant plates stay well-represented; 1.0 = unbiased, ~2.5 ≈ old ≤40px share, higher = smaller.
# (Plates below ~20px need range beyond z=6000mm; raise that bound, not this, to reach them.)
DEPTH_FAR_BIAS = 2.5

# Hunt color (the model's target_color input) vs the seeded plate's color. The seed file fixes the
# plate color, so a uniformly-random hunt color mismatches it ~half the time -> the seed is the
# wrong color -> ineligible as the target -> negative. With this probability we set hunt = seed
# color so the on-screen plate is engageable; the rest are opposite-color "only the wrong color in
# frame" negatives. Seeds are 50/50 red/blue across the file list, so the color INPUT distribution
# stays balanced either way — only its correlation with the visible plate changes.
HUNT_MATCHES_SEED_PROB = 0.7

# Canonical camera — zero distortion, focal sampled near CANONICAL_FX_FY per sample. CANONICAL_CX/CY
# come from autoaim.common (the shared canonical-pinhole convention).
def _make_canonical_camera(fx=None):
  if fx is None:
    fx = _sample_focal()
  cam = np.array([[fx, 0, CANONICAL_CX],
                  [0, fx, CANONICAL_CY],
                  [0, 0, 1]], dtype=np.float32)
  return cam, np.zeros((1, 5), dtype=np.float32)

# Screw-hole rectangle (the keypoints), mm — from the shared source of truth in autoaim.common so
# the training geometry can't drift from plated's PnP. The homography warps the texture AND the
# keypoints together, so what these control is the screw-quad ASPECT RATIO the model learns (absolute
# scale is washed out by the z-randomization). Must match plate_screw_dims() in plated's PnP.
PLATE_WIDTH_SMALL, PLATE_HEIGHT_SMALL = SCREW_DIMS_SMALL[0]*1000, SCREW_DIMS_SMALL[1]*1000  # numbers 2-7
PLATE_WIDTH_LARGE, PLATE_HEIGHT_LARGE = SCREW_DIMS_LARGE[0]*1000, SCREW_DIMS_LARGE[1]*1000  # number 1 / hero

def plate_dims(number:int) -> tuple[float, float]:
  return (PLATE_WIDTH_LARGE, PLATE_HEIGHT_LARGE) if number == 1 else (PLATE_WIDTH_SMALL, PLATE_HEIGHT_SMALL)

# Plate motion blur is applied in OBJECT space (on the warped plate + its alpha, along the plate's
# own image velocity), not on the composited frame — so the static background stays sharp and the
# blur doesn't bleed background across the plate edge. Blur length = per-frame image displacement ×
# this exposure fraction (shutter open time as a fraction of the frame interval).
PLATE_BLUR_EXPOSURE = (0.3, 0.8)

# The plate texture is ~512px but renders at tens of px; warpPerspective (INTER_LINEAR) under-samples
# that downscale and aliases into non-physically crisp detail. Pre-blur the texture by the local
# downscale factor (band-limit to the target sampling rate) so far plates get the soft, low-detail
# look the real camera produces. TEXTURE_MTF_SIGMA is a floor so even large/near plates aren't
# perfectly sharp (approximates the lens MTF + canonical-warp resampling at inference).
TEXTURE_MTF_SIGMA = 0.6

def _motion_kernel(vx, vy, length):
  """Normalized line kernel along (vx, vy) of `length` px, for directional motion blur."""
  n = max(3, min(int(length) | 1, 51))  # odd, clamped
  mid = n // 2
  ang = math.atan2(vy, vx)
  k = np.zeros((n, n), dtype=np.float32)
  for i in range(-mid, mid + 1):
    kx, ky = int(round(mid + i * math.cos(ang))), int(round(mid + i * math.sin(ang)))
    if 0 <= kx < n and 0 <= ky < n: k[ky, kx] = 1.0
  k /= k.sum() + 1e-6
  return k

def project_plate(plate, plate_alpha, plate_kps_local, img, x, y, z, rx, ry, rz, plate_w, plate_h, fx=None, velocity_px=None):
  """Project plate onto image at explicit pose. Returns (projected_kps, H, mask) where projected_kps
  is (N,2) in image pixel coords, H is the homography from texture to image space, and mask is the
  warped binary alpha (IMG_H, IMG_W) of the composited pixels. Returns None on failure. Composits
  plate into img in-place."""
  # Anchor the homography on the four SCREW keypoints, not the texture corners. The screws are the
  # PnP feature and plate_points_3d is the screw rectangle (SCREW_DIMS), so mapping screws->rectangle
  # renders them at the true SCREW_DIMS aspect and lets the plate body extend correctly beyond them.
  # Using the texture corners stretched the screw quad to the texture's aspect (a sim-to-real bug).
  # plate_kps_local order is [TL, TR, BL, BR]; plate_points_3d below matches that order.
  source_points = np.array(plate_kps_local, dtype=np.float32)

  plate_points_3d = np.array([
    [-plate_w / 2, -plate_h / 2, 0],   # TL
    [ plate_w / 2, -plate_h / 2, 0],   # TR
    [-plate_w / 2,  plate_h / 2, 0],   # BL
    [ plate_w / 2,  plate_h / 2, 0],   # BR
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

  # Band-limit the texture for the downscale (see TEXTURE_MTF_SIGMA): pre-blur by the local downscale
  # factor so the INTER_LINEAR warp doesn't alias high-res detail into non-physical sharpness.
  p3 = cv2.perspectiveTransform(np.array([[[0, 0]], [[plate.shape[1], 0]], [[0, plate.shape[0]]]], dtype=np.float32), H).reshape(3, 2)
  downscale = 2.0 / (np.linalg.norm(p3[1] - p3[0]) / plate.shape[1] + np.linalg.norm(p3[2] - p3[0]) / plate.shape[0] + 1e-9)
  plate = cv2.GaussianBlur(plate, (0, 0), max(TEXTURE_MTF_SIGMA, downscale * 0.5))

  warped_plate = cv2.warpPerspective(plate, H, (IMG_W, IMG_H))
  # Continuous-alpha composite (not a binary copyTo): warp the real alpha channel, feather it a little
  # (lens/sensor edge softness, beyond the ~1px warp AA), then blend — so plate edges are soft and
  # sub-pixel-blended with the background instead of a razor-sharp aliased boundary the model cues on.
  warped_alpha = cv2.warpPerspective(plate_alpha, H, (IMG_W, IMG_H))
  warped_alpha = cv2.GaussianBlur(warped_alpha, (0, 0), random.uniform(0.6, 1.8)).astype(np.float32) / 255.0
  # Object-space directional motion blur: smear the plate AND its alpha along the plate's own image
  # velocity (so trailing edges go semi-transparent and blend right), leaving the background sharp.
  if velocity_px is not None:
    blur_len = math.hypot(float(velocity_px[0]), float(velocity_px[1])) * random.uniform(*PLATE_BLUR_EXPOSURE)
    if blur_len >= 2:
      k = _motion_kernel(velocity_px[0], velocity_px[1], blur_len)
      warped_plate = cv2.filter2D(warped_plate, -1, k)
      warped_alpha = cv2.filter2D(warped_alpha, -1, k)
  a = warped_alpha[..., None]
  img[...] = (warped_plate.astype(np.float32) * a + img.astype(np.float32) * (1.0 - a)).astype(np.uint8)
  mask = (warped_alpha > 0.5).astype(np.uint8) * 255  # footprint for id_buf / occlusion gate

  kps_array = np.array(plate_kps_local, dtype=np.float32).reshape(-1, 1, 2)
  projected_kps = cv2.perspectiveTransform(kps_array, H).reshape(-1, 2)
  return projected_kps, H, mask

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
    speed = math.hypot(float(velocity_px[0]), float(velocity_px[1]))
    if speed > 1.5:  # LEDs over-trail the body (they're the only thing bright enough to register)
      warped_led = cv2.filter2D(warped_led, -1, _motion_kernel(velocity_px[0], velocity_px[1],
                                                               np.clip(speed * random.uniform(0.5, 1.5), 5, 51)))

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

def _plate_image_velocity(dyn, dt, fx):
  """Per-frame image-space velocity (px) of the plate center from its smooth dynamics (jitter-free),
  so motion blur smears the plate along its ACTUAL motion. Works at T=1. Mirrors the position formula
  in _compute_plate_poses (centered finite difference over one frame)."""
  def xyz(d):
    th = math.radians(dyn["theta_final"] + dyn["omega"] * d)
    return (dyn["cx"] + dyn["r"] * math.sin(th) + dyn["dvx"] * d,
            dyn["cy"] + dyn["dvy"] * d,
            dyn["cz"] + dyn["r"] * math.cos(th) + dyn["dvz"] * d)
  return _estimate_plate_center_px(*xyz(dt + 0.5), 0, 0, 0, fx) - _estimate_plate_center_px(*xyz(dt - 0.5), 0, 0, 0, fx)

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
  # Partial plates (some corners off-frame) are valid PnP targets, but a plate whose CENTER is off
  # the visible frame has no on-screen anchor — it's effectively invisible. Mark it negative rather
  # than teaching the model to regress fully-exterior keypoints.
  if not (0.0 <= corners_norm[:, 0].mean() <= 1.0 and 0.0 <= corners_norm[:, 1].mean() <= 1.0):
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
# Mixed-color discrimination: when the seed IS the hunt color (a positive scene), add an opposite-
# color distractor this often so the model must hit the hunt-color plate and ignore the wrong one.
# Teaches "don't shoot the wrong color" from a positive, decoupling discrimination from the
# wrong-color-only negative rate.
MIXED_COLOR_PROB = 0.5
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

def _resolve_target_color(seed_plate_name:str, target_color:str|None) -> str:
  """Hunt color. If unset, match the seed's color with prob HUNT_MATCHES_SEED_PROB (so the seeded
  plate is usually an engageable positive), else pick randomly (yields opposite-color negatives)."""
  if target_color is not None: return target_color
  seed_color = seed_plate_name.split("_")[1]
  if seed_color not in TARGET_COLORS: return random.choice(TARGET_COLORS)  # blank seed: no positive anyway
  if random.random() < HUNT_MATCHES_SEED_PROB: return seed_color
  return "blue" if seed_color == "red" else "red"  # opposite -> wrong-color-only negative

def _build_scene_plates(seed_plate_name:str, target_color:str) -> list[str]:
  """Plates to place in a scene. Seed first. When the seed is the hunt color (a positive scene),
  with prob MIXED_COLOR_PROB add an opposite-(wrong-)color distractor so the model must hit the
  hunt plate and ignore the wrong color. Then with prob MULTI_PLATE_PROB add 1-2 generic distractors
  (teammates / blanks) for clutter. Duplicates de-duped."""
  plates = [seed_plate_name]
  seed_color = seed_plate_name.split("_")[1]
  if seed_color == target_color and random.random() < MIXED_COLOR_PROB:
    opp = "blue" if target_color == "red" else "red"
    plates.append(f"{random.choice(ALL_PLATE_NUMBERS)}_{opp}")
  if random.random() < MULTI_PLATE_PROB:
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
  """Single-frame sample = generate_sequence at T=1 — one generator, no train/eval drift.
  Returns (image, class_id, corners_8, target_color_id)."""
  images, class_id, corners_8, target_color_id = generate_sequence(file, T=1, target_color=target_color)
  return images[0], class_id, corners_8, target_color_id

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

def _sample_plate_dynamics(fx, engageable=False):
  """Sample per-plate motion parameters (shared kinematics regardless of seq_type). engageable draws
  the labeled-frame yaw (base_ry + theta_final) UNIFORMLY across [-MAX_ENGAGE_YAW, MAX_ENGAGE_YAW] —
  flat coverage so angled-but-hittable plates (60°, 75°) are as common as head-on, all positives —
  instead of a free 0-360 spin that faces away ~half the time.

  Position is sampled in IMAGE space: the plate's final-frame center is placed uniformly across the
  frame so it lands on-screen at every depth. Metric x/y sampling threw near plates fully off-frame
  (the model sees nothing -> wasted negative); here edge placements still yield partial plates (the
  model still sees most of the plate) but never fully-exterior ones. The orbit center cx/cy/cz is
  back-solved so the plate (center + r·[sinθ,·,cosθ]) lands at the chosen image point."""
  base_ry = random.uniform(-30, 30)
  # final-frame yaw = base_ry + theta_final; pick theta_final so the sum is uniform in the engageable
  # band, else free spin (full yaw, incl. sliver/back-facing negatives).
  theta_final = (random.uniform(-MAX_ENGAGE_YAW, MAX_ENGAGE_YAW) - base_ry) if engageable else random.uniform(0, 360)
  r = random.uniform(150, 250)
  z_plate = math.exp(math.log(200) + random.random() ** (1.0 / DEPTH_FAR_BIAS) * (math.log(6000) - math.log(200)))
  u, v = random.uniform(0, IMG_W), random.uniform(0, IMG_H)
  x_plate = (u - CANONICAL_CX) * z_plate / fx
  y_plate = (v - CANONICAL_CY) * z_plate / fx
  th = math.radians(theta_final)
  return {
    "cx": x_plate - r * math.sin(th),
    "cy": y_plate,
    "cz": z_plate - r * math.cos(th),
    "r":  r,
    "theta_final": theta_final,
    "omega": random.uniform(-15, 15),
    "dvx": random.uniform(-15, 15),
    "dvy": random.uniform(-3, 3),
    "dvz": random.uniform(-30, 30),
    "base_rx": random.uniform(-5, 5),
    "base_ry": base_ry,
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

def _sample_occluders():
  """Foreground occluders (poles / arm / robot edges) as rotated bars, sampled once per sequence so
  they're consistent across frames. Returns a list of (poly_int32, color_rgb)."""
  if random.random() >= OCCLUDER_PROB: return []
  occ = []
  for _ in range(random.randint(1, 2)):
    cx, cy = random.uniform(0, IMG_W), random.uniform(0, IMG_H)
    length = random.uniform(0.4, 1.5) * max(IMG_W, IMG_H)
    width = random.uniform(8, 100)
    ang = random.uniform(0, math.pi)
    dx, dy = math.cos(ang), math.sin(ang)
    nx, ny = -dy, dx
    hl, hw = length / 2, width / 2
    poly = np.array([
      [cx + dx*hl + nx*hw, cy + dy*hl + ny*hw],
      [cx + dx*hl - nx*hw, cy + dy*hl - ny*hw],
      [cx - dx*hl - nx*hw, cy - dy*hl - ny*hw],
      [cx - dx*hl + nx*hw, cy - dy*hl + ny*hw],
    ], dtype=np.int32)
    if random.random() < 0.7:  # mostly dark/gray structure (poles, metal, arm)
      g = random.randint(10, 90)
      color = tuple(int(max(0, min(255, g + random.randint(-12, 12)))) for _ in range(3))
    else:
      color = tuple(random.randint(0, 255) for _ in range(3))
    occ.append((poly, color))
  return occ

def _draw_occluders(img, id_buf, occluders):
  """Composite occluders in front of everything and mark them in id_buf (OCCLUDER_ID) so covered
  target screws count as hidden in the visibility gate."""
  for poly, color in occluders:
    cv2.fillConvexPoly(img, poly, color)
    m = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(m, poly, 1)
    id_buf[m > 0] = OCCLUDER_ID

def generate_sequence(file, T=4, target_color:str|None=None):
  """Generate a T-frame temporal sequence with physics-based plate motion.

  Returns (images, class_id, corners_8, target_color_id). Labels are for the LAST frame and
  apply to the target plate (closest-to-center plate matching target_color, blanks excluded).

  Multi-plate: ~20% of samples include 1-2 distractor plates. The seed plate (the file: arg)
  uses the chosen seq_type (continuous/appearance/spin_jump/static); distractor plates always
  use continuous motion regardless. If no target_color plate is centrally placed at T-1,
  class_id=0 and corners are zeros.
  """
  seed_plate_name = file.split(":")[1]
  target_color = _resolve_target_color(seed_plate_name, target_color)
  target_color_id = 0 if target_color == "red" else 1
  scene_plate_names = _build_scene_plates(seed_plate_name, target_color)

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
  # Background hue/sat/val shift is a scene/white-balance property — sample once and bake into the
  # shared background so it stays consistent across the T frames instead of flickering per frame.
  bg_base = A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.7)(image=bg_base)["image"]
  # Highlight clipping is a sensor property — decide on/off and its threshold once per sequence.
  desat_params = (random.uniform(150, 220), random.uniform(0.2, 0.6)) if random.random() < 0.5 else None

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
    # Bias only the seed plate (the one that can be the label) toward engageable yaw; distractors spin free.
    engageable = plate_idx == 0 and random.random() < SEED_ENGAGEABLE_PROB
    dyn = _sample_plate_dynamics(fx, engageable=engageable)
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

  # Motion blur is object-space (in project_plate, along each plate's own velocity); here keep only
  # the per-frame sensor noise.
  per_frame_aug = A.Compose([
    A.OneOf([
      A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
      A.ISONoise(p=0.5),
    ], p=0.3),
  ])

  occluders = _sample_occluders()  # foreground occluders, consistent across the T frames

  images = []
  final_projections = []  # (color, number, projected_kps, center_px) at frame T-1

  for t in range(T):
    img = bg_base.copy()
    id_buf = np.full((IMG_H, IMG_W), -1, dtype=np.int16)  # topmost (nearest) plate idx per pixel

    # Resolve each visible plate's jittered pose first, then composite far -> near so nearer plates
    # correctly occlude farther ones (plate_infos order is arbitrary, not depth-sorted).
    pending = []  # (z_t, plate_idx, info, pose, curr_center, velocity_px)
    for plate_idx, info in enumerate(plate_infos):
      if t in no_plate_frames_per_plate[plate_idx]:
        continue

      x_t, y_t, z_t, rx_t, ry_t, rz_t = plate_poses[plate_idx][t]
      x_t += random.gauss(0, 1)
      y_t += random.gauss(0, 0.5)
      z_t += random.gauss(0, 2)
      rx_t += random.gauss(0, 0.1)
      ry_t += random.gauss(0, 0.2)
      rz_t += random.gauss(0, 0.05)

      curr_center = _estimate_plate_center_px(x_t, y_t, z_t, rx_t, ry_t, rz_t, fx)
      # plate's own image-space velocity (smooth dynamics) -> object-space motion blur; works at T=1
      velocity_px = _plate_image_velocity(plate_dynamics[plate_idx][0], t - (T - 1), fx)
      pending.append((z_t, plate_idx, info, (x_t, y_t, z_t, rx_t, ry_t, rz_t), curr_center, velocity_px))

    pending.sort(key=lambda p: p[0], reverse=True)  # far first; nearer plates painted on top

    drawn = []  # (plate_idx, info, kps_t, curr_center, ry_t) at t == T-1
    for _z, plate_idx, info, pose, curr_center, velocity_px in pending:
      x_t, y_t, z_t, rx_t, ry_t, rz_t = pose
      ret = project_plate(info["plate_rgba"], info["plate_alpha"], info["kps_local"], img,
                          x_t, y_t, z_t, rx_t, ry_t, rz_t, info["plate_w"], info["plate_h"], fx, velocity_px=velocity_px)
      if ret is not None:
        kps_t, H_t, mask_t = ret
        id_buf[mask_t > 0] = plate_idx
        _apply_emissive_leds(img, info["plate_rgba"], info["plate_alpha"], info["color"], H_t, velocity_px=velocity_px)
        if t == T - 1: drawn.append((plate_idx, info, kps_t, curr_center, ry_t))

    # Foreground occluders in front of all plates (+ marked in id_buf), so the visibility gate below
    # treats covered screws as hidden.
    _draw_occluders(img, id_buf, occluders)

    # Target candidates at the final frame: engageable yaw AND not occluded by a nearer plate. Both
    # gates keep the plate in the image but drop it from the target set, yielding a negative sample.
    if t == T - 1:
      for plate_idx, info, kps_t, curr_center, ry_t in drawn:
        if abs(((ry_t + 180) % 360) - 180) > MAX_ENGAGE_YAW: continue
        quad = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        cv2.fillConvexPoly(quad, kps_t.reshape(-1, 1, 2).astype(np.int32), 1)
        area = int(quad.sum())
        if area == 0: continue  # quad fully off-frame
        if ((id_buf == plate_idx) & (quad > 0)).sum() / area < MIN_TARGET_VISIBLE: continue
        final_projections.append((info["color"], info["number"], kps_t, curr_center))

    if desat_params is not None:
      img = apply_highlight_desat(img, p=1.0, thresh=desat_params[0], desat_factor=desat_params[1])
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
