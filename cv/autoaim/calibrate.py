import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(1)

while len(objpoints) < 20:
  ret, frame = cap.read()
  if not ret:
    break

  # Convert the frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # find chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
  if ret:
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    cv2.drawChessboardCorners(frame, (9,6), corners2, ret)

  cv2.imshow('img', frame)
  key = cv2.waitKey(1)

  if ret and key == ord("s"):
    objpoints.append(objp)
    imgpoints.append(corners2)

  if key == ord("q"):
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("Distortion coefficient : \n")
print(dist)

h, w = gray.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

while True:
  ret, frame = cap.read()
  if not ret:
    break

  # undistort
  dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

  # crop the image
  x,y,w,h = roi
  dst = dst[y:y+h, x:x+w]

  cv2.imshow('calibresult', dst)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
