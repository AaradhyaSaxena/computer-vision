import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils_gamma import *

# # img = cv2.imread('image/a.png',0)
# # img = cv2.imread('image/a2.png',0)
# img = cv2.imread('opencv_frame_ref_1.png',0)

# # Initiate STAR detector
# # orb = cv2.ORB()
# orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
# 				nlevels=8, fastThreshold=20,scaleFactor=1.2, 
# 				WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
# 				firstLevel=0,nfeatures=500)

# # find the keypoints with ORB
# kp = orb.detect(img,None)
# # print(kp)
# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)

# # #draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img,kp,0,color=(0,255,0), flags=0)
# plt.imshow(img2),plt.show()


###########################
##########function#########
# def find_corners(img):
# 	orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
# 				nlevels=8, fastThreshold=20,scaleFactor=1.2, 
# 				WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
# 				firstLevel=0,nfeatures=500)
# 	kp = orb.detect(img,None)
# 	kp, des = orb.compute(img, kp)

# 	corn_arr = []
# 	for i in range(len(kp)):
# 		corn_arr.append(kp[i].pt)
# 	corner = np.array(corn_arr)

# 	corner_x = corner[:,0]
# 	corner_y = corner[:,1]

# 	return corner_x, corner_y
############################






