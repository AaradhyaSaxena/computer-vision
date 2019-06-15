import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from utils_gamma import *

_3d_points=[]
_2d_points=[]

essentialMatrix=[]
tvecs=[]
rvecs=[]
index=[]
image=[]
X =[]


img_paths=glob('*.png')
for path in img_paths:
    img = cv2.imread(path,0)
    corners = find_corner_pts(img,500)
    _2d_points.append(corners)
    image.append(img)

homo_im = np.array(_2d_points)
image = np.array(image)


N = len(homo_im)

for i in range(N):
	j = i+1
	while(j<N):
		e = essential_matrix(homo_im[i],homo_im[j])
		essentialMatrix.append(e)
		tvecs.append(returnT_fromE(e))
		rvecs.append(returnR1_fromE(e))
		index.append((i,j))
		X.append(np.stack((image[i],image[j]),axis=-1))
		j = j+1

# np.savez('data/training_data', e = essentialMatrix, r = rvecs, t = tvecs, 
# 			corners = homo_im, index = index, X = X)
# l = np.load('data/training_data.npz')
# print(l.files)


#----------------predicted--------------------

model.load_weights('data/model_1')

# pre = np.load('data/prediction_model1.npz')
# r_pre = []
# t_pre = []
# e_pre =[]
# for i in range(len(pre['e'])):
# 	t_pre.append(returnT_fromE(pre['e'][i].reshape((3,3))))
# 	r_pre.append(returnR1_fromE(pre['e'][i].reshape((3,3))))
# 	e_pre.append(pre['e'])

# np.savez('data/prediction_model1', e_pre = e_pre, t_pre = t_pre, r_pre = r_pre)

#-----------------depth_estimation--------------











