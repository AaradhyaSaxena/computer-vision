import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from numpy import linalg as LA
from sklearn import linear_model, datasets
from utils import * 
from utils_beta import *

grid_x=21
grid_y=15
grid=(21,15)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.imread('./image/a2.png',0)

obj_pt3x4_1 = return_objpoints(grid)
img_pt1, corners1 = return_imagepoints(image_path1,grid)
obj_pt3x4_2 = return_objpoints(grid)
img_pt2, corners2 = return_imagepoints(image_path2,grid)
   
P1 = projection_matrix4(img_pt1,obj_pt3x4_1)
P2 = projection_matrix4(img_pt2,obj_pt3x4_2)

# P1 = return_pcv(image_path1,corners1,obj_pt3x4_1)
# P2 = return_pcv(image_path2,corners2,obj_pt3x4_2)

F = return_fundamentalM(P1, P2)
F = F/F[-1,-1]

print(F)

w = errorFundamental(F,img_pt1,img_pt2)
print(w)
