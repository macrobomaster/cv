"""Minimal ctypes binding to the AprilTag 3 C library (nixpkgs `apriltag`).

cv2.aruco's apriltag detection has coarse corners (noisy PnP) and is slow;
AprilRobotics' AprilTag 3 is the reference detector — better quad fits, subpixel
corners, multithreaded, and a `quad_decimate` speed knob. nixpkgs ships the C lib
(`libapriltag.so`) but no usable Python binding (the bundled compiled extension
is locked to one python ABI), so this wraps the C API directly via ctypes
(python-version-independent). Load path comes from $APRILTAG_LIB (set by the
flake) with fallbacks.

detect(gray) -> [(id:int, corners:(4,2) float32 px)] in cv2.aruco order
(TL,TR,BR,BL) — AprilTag's native order is a fixed [1,0,3,2] permutation of that
(verified orientation-independent), reordered here so this is a drop-in for
cv2.aruco and the downstream PnP object points are unchanged.
"""
import ctypes, os

import numpy as np

def _load() -> ctypes.CDLL:
  for cand in (os.environ.get("APRILTAG_LIB"), "libapriltag.so", "libapriltag.so.3"):
    if not cand: continue
    try: return ctypes.CDLL(cand)
    except OSError: continue
  raise OSError("libapriltag.so not found — set APRILTAG_LIB or add nixpkgs `apriltag` to the env")

_lib = _load()

# --- struct layouts (apriltag.h, v3.x — stable) ----------------------------
class _ImageU8(ctypes.Structure):
  _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32),
              ("stride", ctypes.c_int32), ("buf", ctypes.c_void_p)]

# only the leading user-configurable params — enough to set them by overlaying
# the real detector's memory (private fields follow; we never touch them).
class _Detector(ctypes.Structure):
  _fields_ = [("nthreads", ctypes.c_int), ("quad_decimate", ctypes.c_float),
              ("quad_sigma", ctypes.c_float), ("refine_edges", ctypes.c_bool),
              ("decode_sharpening", ctypes.c_double), ("debug", ctypes.c_bool)]

class _Detection(ctypes.Structure):
  _fields_ = [("family", ctypes.c_void_p), ("id", ctypes.c_int), ("hamming", ctypes.c_int),
              ("decision_margin", ctypes.c_float), ("H", ctypes.c_void_p),
              ("c", ctypes.c_double * 2), ("p", (ctypes.c_double * 2) * 4)]

class _ZArray(ctypes.Structure):
  _fields_ = [("el_sz", ctypes.c_size_t), ("size", ctypes.c_int),
              ("alloc", ctypes.c_int), ("data", ctypes.c_void_p)]

_lib.apriltag_detector_create.restype = ctypes.c_void_p
_lib.tag36h11_create.restype = ctypes.c_void_p
_lib.apriltag_detector_add_family_bits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_lib.apriltag_detector_detect.restype = ctypes.c_void_p
_lib.apriltag_detector_detect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ImageU8)]
_lib.apriltag_detections_destroy.argtypes = [ctypes.c_void_p]

class Detector:
  """36h11 detector. quad_decimate>1 detects quads on a downscaled image (faster,
  slightly less accurate); refine_edges + subpixel give clean corners."""
  def __init__(self, nthreads=4, quad_decimate=1.0, quad_sigma=0.0,
               refine_edges=True, decode_sharpening=0.25):
    self._td = _lib.apriltag_detector_create()
    self._fam = _lib.tag36h11_create()
    _lib.apriltag_detector_add_family_bits(self._td, self._fam, 2)
    d = _Detector.from_address(self._td)
    d.nthreads = int(nthreads); d.quad_decimate = float(quad_decimate)
    d.quad_sigma = float(quad_sigma); d.refine_edges = bool(refine_edges)
    d.decode_sharpening = float(decode_sharpening)

  def detect(self, gray:np.ndarray) -> list:
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape
    im = _ImageU8(w, h, w, gray.ctypes.data_as(ctypes.c_void_p))
    za_ptr = _lib.apriltag_detector_detect(self._td, ctypes.byref(im))
    out = []
    if za_ptr:
      za = _ZArray.from_address(za_ptr)
      for i in range(za.size):
        det_ptr = ctypes.cast(za.data + i * za.el_sz, ctypes.POINTER(ctypes.c_void_p))[0]
        det = _Detection.from_address(det_ptr)
        p = det.p  # AprilTag native order → cv2.aruco TL,TR,BR,BL via [1,0,3,2]
        corners = np.array([[p[k][0], p[k][1]] for k in (1, 0, 3, 2)], dtype=np.float32)
        out.append((int(det.id), corners))
      _lib.apriltag_detections_destroy(za_ptr)
    return out
