import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
# from sklearn import linear_model, datasets
from utils import *
# from utils_beta import *
from utils_gamma import *


_3d_points=[]
_2d_points=[]

# img_paths=glob('*.png') #get paths of all all images
img_paths = ['data_5.png','data_6.png']

for path in img_paths:
    img =cv2.imread(path)
    corners = find_corners(img)
    _2d_points.append(corners)

homo_im = np.array(_2d_points)

#im shape = ()
def essential_matrix(im):

	length = im.shape[1]
	homo = np.ones((2,length,3))
	homo_im = np.ones((2,length,3))
	homo[:,:,:2] = im[:,:,:]
	kk = np.load("data/parameters.npz")
	k = kk['k_new']
	# k = kk['k']
	homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
	homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T
	
	A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
		(homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
		homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
		(homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
		homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
		homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
	
	ata = np.matmul(A.T,A)
	u, s, vh = np.linalg.svd(ata, full_matrices=True)
	L = vh[-1]
	H = L.reshape(3, 3)

	u1, s1, vh1 = np.linalg.svd(H,full_matrices=True)

	s2 = np.array([(s1[0]+s1[1])/2, (s1[0]+s1[1])/2, 0])
	left = np.matmul(u1,np.diag(s2))
	E = np.matmul(left,vh1)


	return E

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











