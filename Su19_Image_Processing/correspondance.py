import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
# from sklearn import linear_model, datasets
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

homo_im = np.array(_2d_points)

e = essential_matrix(homo_im)
print("essential_matrix:\n",e,"\n")

t = returnT_fromE(e)
print("translation:\n",t,"\n")
# tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]

r1,r2 = returnR_fromE(e)
print("rotation_matrix",r1,"\n",r2,"\n")

# u, v = returnUV_fromE(e)

# m1,m2,m3,m4 = returnP_fromE(e)

# print(m1,m2,m3,m4)











