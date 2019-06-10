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

img_paths=glob('*.png')
for path in img_paths:
    img =cv2.imread(path)
    corners = find_corners(img)
    _2d_points.append(corners)

homo_im = np.array(_2d_points)

N = len(homo_im)

################

# im1 = homo_im[0]
# im2 = homo_im[1]

# length = im1.shape[0]
# homo = np.ones((2,length,3))
# homo_im = np.ones((2,length,3))
# homo[:,:,0] = im1.T
# homo[:,:,1] = im2.T
# kk = np.load("data/parameters.npz")
# # k = kk['k_new']
# k = kk['k']
# homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
# homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T

# A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
# 	(homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
# 	homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
# 	(homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
# 	homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
# 	homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
	
# ata = np.matmul(A.T,A)
# u, s, vh = np.linalg.svd(ata, full_matrices=True)
# L = vh[-1]
# H = L.reshape(3, 3)

###################

for i in range(N):
	for j in range(N):
		while(j>i and j<N):
			e = essential_matrix(homo_im[i],homo_im[j])
			essentialMatrix.append(e)
			tvecs.append(returnT_fromE(e))
			rvecs.append(returnR1_fromE(e))

# np.savez('data/training_data', e = essentialMatrix, r = rvecs, t = tvecs, im = homo_im)
# l = np.load('data/training_data.npz')
# l.files



