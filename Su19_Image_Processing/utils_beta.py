import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from sklearn import linear_model, datasets
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

# homo_im and homo_ob are numpy array
def essential_matrix(homo_im,homo_ob):
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


















