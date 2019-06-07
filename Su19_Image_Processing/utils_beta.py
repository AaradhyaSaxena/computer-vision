import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
# from sklearn import linear_model, datasets
from numpy import linalg as LA
from utils import *


def return_pcv(img_path, corners_, obj_p):
	objpoints = [] 
	imgpoints = []
	imgpoints.append(corners_)
	objpoints.append(obj_p)
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
	rx = [[1,0,0],[0,math.cos(rvecs[0][0]),math.sin(rvecs[0][0])],[0,(-1)*math.sin(rvecs[0][0]),math.cos(rvecs[0][0])]]
	ry = [[math.cos(rvecs[0][1]),0,math.sin(rvecs[0][1])],[0,1,0],[(-1)*math.sin(rvecs[0][1]),0,math.cos(rvecs[0][1])]]
	rz = [[math.cos(rvecs[0][2]),math.sin(rvecs[0][2]),0],[(-1)*math.sin(rvecs[0][2]),math.cos(rvecs[0][2]),0],[0,0,1]]
	R1 = np.matmul(rx,ry)
	R = np.matmul(R1,rz)
	m = np.ones((3,4))
	m[:,:3] = R
	m[:,[3]]= np.array(tvecs)
	cv_homography = np.matmul(mtx,m)
	pcv = cv_homography/cv_homography[-1,-1]

	return pcv

def return_fundamentalM(P1, P2):
	A1 = P1[:,:3]
	a1 = P1[:,3]

	A2 = P2[:,:3]
	a2 = P2[:,3]

	b = np.matmul(inv(A2),a2) - np.matmul(inv(A1),a1)
	Sb = [0,(-1)*b[2],b[1],b[2],0,(-1)*b[0],(-1)*b[1],b[0],0]
	Sb = np.array(Sb)
	Sb = Sb.reshape((3,3))

	F = np.matmul(np.matmul(inv(A1).T , Sb) ,inv(A2))

	return F

# 
def errorFundamental(F,img_pt1, img_pt2):
	x1 = homo_img(img_pt1)
	x2 = homo_img(img_pt2)

	w = np.matmul(np.kron(x2,x1).T, F.reshape((9,1)))
	print(w.shape)
	w1 = np.sum(w)

	return w1

# homo_im is numpy array
def essential_matrix(homo_im):

	length = homo_im.shape[1]
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
	H = H/H[-1,-1]

	return H

# homo_im is numpy array
def essential_matrix_cal(homo_im):
	A = np.hstack(((homo_im[1,:,0,0]*homo_im[0,:,0,0]).reshape((315,1)),
		(homo_im[1,:,0,0]*homo_im[0,:,0,1]).reshape((315,1)),
		homo_im[1,:,0,0].reshape((315,1)),(homo_im[1,:,0,1]*homo_im[0,:,0,0]).reshape((315,1)),
		(homo_im[1,:,0,1]*homo_im[0,:,0,1]).reshape((315,1)),
		homo_im[1,:,0,1].reshape((315,1)),homo_im[0,:,0,0].reshape((315,1)),
		homo_im[0,:,0,1].reshape((315,1)),np.ones((315,1))))
	
	ata = np.matmul(A.T,A)
	u, s, vh = np.linalg.svd(ata, full_matrices=True)
	L = vh[-1]
	H = L.reshape(3, 3)
	H = H/H[-1,-1]

	return H

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

def returnP_fromE(e):

	t = returnT_fromE(e)
	tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]
	u, v = returnUV_fromE(e)
	z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
	w = np.array([[0,1,0],[-1,0,0],[0,0,1]])
	d = np.array([[1,0,0],[0,1,0],[0,0,0]])
	r1 = np.dot(u,np.dot(w.T,v.T))
	r2 = np.dot(u,np.dot(w,v.T))

	m1 = np.dot(r1,np.array([[1,0,0,t[0]],[0,1,0,t[1]],[1,0,0,t[2]]]))
	m2 = np.dot(r1,np.array([[1,0,0,(-1)*t[0]],[0,1,0,(-1)*t[1]],[1,0,0,(-1)*t[2]]]))
	m3 = np.dot(r2,np.array([[1,0,0,t[0]],[0,1,0,t[1]],[1,0,0,t[2]]]))
	m4 = np.dot(r2,np.array([[1,0,0,(-1)*t[0]],[0,1,0,(-1)*t[1]],[1,0,0,(-1)*t[2]]]))

	return m1, m2, m3, m4











