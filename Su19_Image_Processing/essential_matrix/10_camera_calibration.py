# Here we calculate the value of intrinsic callibration constant of our camera

import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
# from sklearn import linear_model, datasets
from utils import *


_3d_points=[]
_2d_points=[]

grid_x = 21
grid_y = 15
grid = (grid_x, grid_y)
h = 480
w = 640

x,y=np.meshgrid(range(grid_x),range(grid_y))

world_points=np.hstack((x.reshape(grid_x*grid_y,1),y.reshape(grid_x*grid_y,1),np.zeros((grid_x*grid_y,1)))).astype(np.float32)

# p1 = "\images"
img_paths=glob('*.png') #get paths of all all images
for path in img_paths:
    im=cv2.imread(path)
    ret, corners = cv2.findChessboardCorners(im, grid)

    if ret: #add points only if checkerboard was correctly detected:
        _2d_points.append(corners) #append current 2D points
        _3d_points.append(world_points) #3D points are always the same

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1],im.shape[0]),None,None)

### Undistortion
nmtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

### Reprojection Error
tot_error = 0
for i in xrange(len(_3d_points)):
    _2d_points2, _ = cv2.projectPoints(_3d_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(_2d_points[i],_2d_points2, cv2.NORM_L2)/len(_2d_points2)
    tot_error += error
    err = tot_error/len(_3d_points)
print "total error: ", err

# np.savez('data/parameters',k_new = nmtx, k = mtx, rvecs =rvecs, tvecs = tvecs, distortion_coeff = dist, reprojection_error = err)
# l = np.load('data/parameters.npz')
# l.files

# p = returnP_fromK(mtx,rvecs,tvecs)
# print(p)

