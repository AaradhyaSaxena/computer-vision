import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from utils import *

# Load previously saved data
with np.load('data/parameters.npz') as X:
    nmtx,mtx, dist, _, _ = [X[i] for i in ('k_new','k','distortion_coeff','rvecs','tvecs')]

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((21*15,3), np.float32)
objp[:,:2] = np.mgrid[0:21,0:15].T.reshape(-1,2)

########   DRAW THE AXIS ON THE BOARD
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for fname in glob('*.png'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (21,15),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        a , rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, nmtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, nmtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()


# ############  RENDER A CUBE
# def drawCube(img, corners, imgpts):
#     imgpts = np.int32(imgpts).reshape(-1,2)

#     # draw ground floor in green
#     img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

#     # draw pillars in blue color
#     for i,j in zip(range(4),range(4,8)):
#         img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

#     # draw top layer in red color
#     img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

#     return img

# axis_cube = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
#                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


# for fname in glob('*.png'):
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (21,15),None)

#     if ret == True:
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

#         # Find the rotation and translation vectors.
#         a , rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

#         # project 3D points to image plane
#         imgpts, jac = cv2.projectPoints(axis_cube, rvecs, tvecs, nmtx, dist)

#         img = drawCube(img,corners2,imgpts)
#         cv2.imshow('img',img)
#         k = cv2.waitKey(0) & 0xff
#         if k == 's':
#             cv2.imwrite(fname[:6]+'.png', img)

# cv2.destroyAllWindows()

