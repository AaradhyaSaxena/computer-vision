import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from utils import *
from utils_beta import *

def find_corners(img):
	orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
				nlevels=8, fastThreshold=20,scaleFactor=1.2, 
				WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
				firstLevel=0,nfeatures=500)
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

