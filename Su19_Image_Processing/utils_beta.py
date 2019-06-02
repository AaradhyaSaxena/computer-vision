import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from sklearn import linear_model, datasets
from numpy import linalg as LA


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

