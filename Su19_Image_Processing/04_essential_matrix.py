###################################
######## Essential matrix  ########
###################################

import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from utils import *
from utils_beta import *

_3d_points=[]
_2d_points=[]

grid_x = 21
grid_y = 15
grid = (grid_x, grid_y)

x,y=np.meshgrid(range(grid_x),range(grid_y))

world_points=np.hstack((x.reshape(grid_x*grid_y,1),y.reshape(grid_x*grid_y,1),np.zeros((grid_x*grid_y,1)))).astype(np.float32)

img_paths = ['opencv_frame_0.png','opencv_frame_1.png']
# img_paths=glob('*.png') #get paths of all all images
for path in img_paths:
    im=cv2.imread(path)
    ret, corners = cv2.findChessboardCorners(im, grid)
    
    if ret: #add points only if checkerboard was correctly detected:
        _2d_points.append(corners) #append current 2D points
        _3d_points.append(world_points) #3D points are always the same

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1],im.shape[0]),None,None)
# print(mtx)

im = np.array(_2d_points)
# homo_ob = np.array(_3d_points)
# print(homo_im.shape)

e = essential_matrix_cal(im)
print("essential_matrix:/n",e,"/n")

t = returnT_fromE(e)
print("translation:/n",t,"/n")
# tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]

r1,r2 = returnR_fromE(e)
print("rotation_matrix",r1,"/n",r2,"/n")
# u, v = returnUV_fromE(e)

# m1,m2,m3,m4 = returnP_fromE(e)

# print(m1,m2,m3,m4)
