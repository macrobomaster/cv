import sys
import cv2
import numpy as np
from ..system.core import messaging

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = objp * 12.7 # scale to 1/2 inch squares but in mm

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

addr = sys.argv[1]
sub = messaging.Sub(["camera_feed"], addr=addr)

while len(objpoints) < 20:
  sub.update()
  camera_feed = sub["camera_feed"]
  if sub.updated["camera_feed"] and camera_feed is not None:
    frame = np.frombuffer(camera_feed["frame"], dtype=np.uint8).reshape(256, 512, 3).copy()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
      corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
      cv2.drawChessboardCorners(frame, (9,6), corners2, ret)

    # draw the number of object points
    cv2.putText(frame, f"Object Points: {len(objpoints)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('img', frame)
    key = cv2.waitKey(1)

    if ret and key == ord("s"):
      objpoints.append(objp)
      imgpoints.append(corners2)

    if key == ord("q"):
      exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print in a copyable format
print("Camera matrix : \n")
print(repr(mtx).replace("array(", "").replace(")", ""))
print("Distortion coefficient : \n")
print(repr(dist).replace("array(", "").replace(")", ""))
