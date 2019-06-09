import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from utils import *
from utils_beta import *
from utils_gamma import *

_3d_points=[]
_2d_points=[]

# img_paths=glob('*.png') #get paths of all all images
img_paths = ['opencv_frame_ref_0.png','opencv_frame_ref_1.png']

for path in img_paths:
    img =cv2.imread(path)
    corners = find_corners(img)
    _2d_points.append(corners)

im = np.array(_2d_points)









e = essential_matrix(im)



# img = cv2.imread("opencv_frame_ref_0.png",0)

# h = img.shape[1]
# w = img.shape[0]

# index = np.ones((3,640,480))
# index[:2,:,:] = np.mgrid[0:h,0:w]
# temp = index.reshape((3,-1)).T

# V = np.matmul(temp,e)
# V[:,0] = V[:,0]/V[:,2]
# V[:,1] = V[:,1]/V[:,2]
# V[:,2] = V[:,2]/V[:,2]

# # V = V.reshape((3,480,640))
# print(V.shape)

