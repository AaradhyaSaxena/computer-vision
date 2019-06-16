import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from utils import *

_3d_points=[]
points1=[]
points2=[]

essentialMatrix=[]
tvecs=[]
rvecs=[]
depth=[]
index=[]
image1=[]
image2=[]
X1 =[]
X2 =[]
##########################
##########################
##########################
##########################
# def return_depth(homo_pt1,homo_pt2,r1,t):

# 	length = homo_pt1.shape[1]
# 	homo1 = np.ones((length,3))
# 	homo2 = np.ones((length,3))
# 	homo1[:,:2] = homo_pt1[:,:]
# 	homo2[:,:2] = homo_pt2[:,:]
# 	kk = np.load("data/parameters.npz") # k = kk['k_new']
# 	k = kk['k']
# 	homo_im1 = np.ones((length,3))
# 	homo_im2 = np.ones((length,3))
# 	homo_im1[:,:] = np.matmul(inv(k),homo1[:,:].T).T
# 	homo_im2[:,:] = np.matmul(inv(k),homo2[:,:].T).T

# 	rot1 = np.matmul(np.matmul(homo_im2,r1),homo_im1.T)
# 	trans1 = np.matmul(homo_im2,t.reshape((3,1)))

# 	A = np.hstack((rot1,trans1))
# 	ata = np.matmul(A.T,A)
# 	u, s, vh = np.linalg.svd(ata, full_matrices=True)
# 	Depth = vh[-1].reshape(length+1,1)

# 	return Depth


# def find_correspondance(img1,img2,n_pts=20):

# 	orb = cv2.ORB()
# 	orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
# 					nlevels=8, fastThreshold=20,scaleFactor=1.2, 
# 					WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
# 					firstLevel=0,nfeatures=500)
# 	kp1 = orb.detect(img1,None)
# 	kp1, des1 = orb.compute(img1, kp1)
# 	kp2 = orb.detect(img2,None)
# 	kp2, des2 = orb.compute(img2, kp2)

# 	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 	matches = bf.match(des1,des2)
# 	matches = sorted(matches, key = lambda x:x.distance)

# 	good = []
# 	pts1 = []
# 	pts2 = []

# 	for m in matches:
# 		good.append([m])
# 		pts2.append(kp2[m.trainIdx].pt)
# 		pts1.append(kp1[m.queryIdx].pt)

# 	pts1 = np.int32(pts1)
# 	pts2 = np.int32(pts2)
	
# 	return pts1, pts2
##########################
##########################
##########################
##########################

img_paths=glob('*.png')
# img_paths = ['data_5.png','data_6.png']
N = len(img_paths)

for i in range(N):
	j = i+1
	while(j<N):
		img1 = cv2.imread(img_paths[i],0)
		img2 = cv2.imread(img_paths[j],0)
		pt1,pt2 = find_correspondance(img1,img2)
		points1.append(pt1[:20])
		points2.append(pt2[:20])
		image1.append(img1)
		image2.append(img2)
		j =j+1

homo_pt1 = np.array(points1)
homo_pt2 = np.array(points2)
image1 = np.array(image1)
image1 = image1[:,:,:,np.newaxis]
image2 = np.array(image2)
image2 = image2[:,:,:,np.newaxis]


Nn = len(homo_pt1)
for i in range(Nn):
	e = essential_matrix(homo_pt1[i],homo_pt2[i])
	essentialMatrix.append(e)
	tvecs.append(returnT_fromE(e))
	rvecs.append(returnR1_fromE(e))
	# depth.append(return_depth(homo_pt1[i],homo_pt2[i],returnR1_fromE(e),returnT_fromE(e)))
	index.append((i,j))

np.savez('data/training_data', e = essentialMatrix, r = rvecs, t = tvecs, 
			depth = depth, correspondance1 = homo_pt1, correspondance2 = homo_pt2, 
			index = index, X1 = image1, X2 = image2)
l = np.load('data/training_data.npz')
print(l.files)






#####--------------------------------------------

##########################
##########################
##########################
##########################
##########################
##########################
##########################
##########################
##########################
##########################
#############################
# #im shape = ()
# def essential_matrix(im):

# 	length = im.shape[1]
# 	homo = np.ones((2,length,3))
# 	homo_im = np.ones((2,length,3))
# 	homo[:,:,:2] = im[:,:,:]
# 	kk = np.load("data/parameters.npz")
# 	k = kk['k_new']
# 	# k = kk['k']
# 	homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
# 	homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T
	
# 	A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
# 		(homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
# 		homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
# 		(homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
# 		homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
# 		homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
	
# 	ata = np.matmul(A.T,A)
# 	u, s, vh = np.linalg.svd(ata, full_matrices=True)
# 	L = vh[-1]
# 	H = L.reshape(3, 3)

# 	u1, s1, vh1 = np.linalg.svd(H,full_matrices=True)

# 	s2 = np.array([(s1[0]+s1[1])/2, (s1[0]+s1[1])/2, 0])
# 	left = np.matmul(u1,np.diag(s2))
# 	E = np.matmul(left,vh1)

# 	return E

# def return_depth(im,r1,t):

# 	length = im.shape[1]
# 	homo = np.ones((2,length,3))
# 	homo_im = np.ones((2,length,3))
# 	homo_im[:,:,:2] = im[:,:,:]
# 	# homo[:,:,:2] = im[:,:,:]
# 	# kk = np.load("data/parameters.npz")
# 	# # k = kk['k_new']
# 	# k = kk['k']
# 	# homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
# 	# homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T

# 	rot1 = np.matmul(np.matmul(homo_im[1],r1),homo_im[0].T)
# 	trans1 = np.matmul(homo_im[1],t.reshape((3,1)))

# 	A = np.hstack((rot1,trans1))
# 	ata = np.matmul(A.T,A)
# 	u, s, vh = np.linalg.svd(ata, full_matrices=True)
# 	Depth = vh[-1].reshape(length+1,1)

# 	return Depth	
##----------------------------------------
# e = essential_matrix(homo_im)
# print("essential_matrix:\n",e,"\n")

# t = returnT_fromE(e)
# print("translation:\n",t,"\n")
# # tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]

# r1,r2 = returnR_fromE(e)
# print("rotation_matrix",r1,"\n",r2,"\n")

# depth = return_depth(homo_im,r1,t)
# print("depth:\n",depth,"\n")

# u, v = returnUV_fromE(e)

# m1,m2,m3,m4 = returnP_fromE(e)

# print(m1,m2,m3,m4)













