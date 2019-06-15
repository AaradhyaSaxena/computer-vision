import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from utils import *
from utils_beta import *

def find_corner_pts(img,n_pts=500):
	orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
				nlevels=8, fastThreshold=20,scaleFactor=1.2, 
				WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
				firstLevel=0,nfeatures= n_pts)
	kp = orb.detect(img,None)
	kp, des = orb.compute(img, kp)

	corn_arr = []
	for i in range(len(kp)):
		corn_arr.append(kp[i].pt)
	# corner = np.array(corn_arr)

	# corner_x = corner[:,0]
	# corner_y = corner[:,1]

	# return corner_x, corner_y
	return corn_arr


#shape of im2=im1=(500,2) i
def essential_matrix(im1, im2):

	length = im1.shape[0]
	homo = np.ones((2,length,3))
	homo_im = np.ones((2,length,3))
	# homo[:,:,0] = im1.T
	# homo[:,:,1] = im2.T
	homo[0,:,:2] = im1
	homo[1,:,:2] = im2
	
	kk = np.load("data/parameters.npz")
	# k = kk['k_new']
	k = kk['k']

	#shape of k=3x3 
	homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
	homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T
	
	xa=homo_im[0,:,0]
	xb=homo_im[1,:,0]
	ya=homo_im[0,:,1]
	yb=homo_im[1,:,1]
	c=np.ones((length,))
	A=[xb*xa,xb*ya,yb*xa,yb*ya,xb,yb,xa,ya,c] 
	A=np.array(A).T

	# A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
	# 	(homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
	# 	homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
	# 	(homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
	# 	homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
	# 	homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
	
	ata = np.matmul(A.T,A)
	u, s, vh = np.linalg.svd(ata, full_matrices=True)
	L = vh[-1]
	H = L.reshape(3, 3)

	u1, s1, vh1 = np.linalg.svd(H,full_matrices=True)

	s2 = np.array([(s1[0]+s1[1])/2, (s1[0]+s1[1])/2, 0])
	left = np.matmul(u1,np.diag(s2))
	E = np.matmul(left,vh1)


	return E

# takes in 3x3 essential matrix
def returnT_fromE(e):
	ete = np.matmul(e.T,e)
	u, s, vh = np.linalg.svd(ete, full_matrices=True)
	v = vh.T

	return v[:,-1]

def returnUV_fromE(e):
	ete = np.matmul(e.T,e)
	u, s, vh = np.linalg.svd(ete, full_matrices=True)
	v = vh.T
	# v = v/v[-1]

	return u, v

def returnR_fromE(e):

	t = returnT_fromE(e)
	tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]
	u, v = returnUV_fromE(e)
	z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
	w = np.array([[0,1,0],[-1,0,0],[0,0,1]])
	d = np.array([[1,0,0],[0,1,0],[0,0,0]])
	r1 = np.dot(u,np.dot(w.T,v.T))
	r2 = np.dot(u,np.dot(w,v.T))

	return r1,r2	

def returnR1_fromE(e):

	t = returnT_fromE(e)
	tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]
	u, v = returnUV_fromE(e)
	z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
	w = np.array([[0,1,0],[-1,0,0],[0,0,1]])
	d = np.array([[1,0,0],[0,1,0],[0,0,0]])
	r1 = np.dot(u,np.dot(w.T,v.T))
	r2 = np.dot(u,np.dot(w,v.T))

	return r1





















