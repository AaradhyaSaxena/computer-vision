import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from utils_beta import *
from utils_gamma import *

# img1 = cv2.imread('opencv_frame_ref_1.png',0)  #queryimage # left image
# img2 = cv2.imread('opencv_frame_ref_2.png',0) #trainimage # right image

img1 = cv2.imread('data_3.png',0)  
img2 = cv2.imread('data_7.png',0)


#####################################
### THIS DOES NOT TRULY BELONG HERE #
#####################################

# _3d_points=[]
# _2d_points=[]
# img_paths = ['opencv_frame_3.png','opencv_frame_4.png']
# for path in img_paths:
#     img =cv2.imread(path)
#     corners = find_corners(img)
#     _2d_points.append(corners)
# homo_im = np.array(_2d_points)

# e = essential_matrix(homo_im)
# print("essential_matrix:\n",e,"\n")
#####################################
#####################################
# sift = cv2.SIFT()
orb = cv2.ORB()
orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
				nlevels=8, fastThreshold=20,scaleFactor=1.2, 
				WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
				firstLevel=0,nfeatures=500)

## find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
kp1 = orb.detect(img1,None)
kp1, des1 = orb.compute(img1, kp1)
kp2 = orb.detect(img2,None)
kp2, des2 = orb.compute(img2, kp2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None, flags=2)
# plt.imshow(img3),plt.show()

good = []
pts1 = []
pts2 = []

for m in matches:
	good.append([m])
	pts2.append(kp2[m.trainIdx].pt)
	pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

##########################################################
##########################################################
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,e)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,e)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
##########################################################
##########################################################

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

#########################
#### DEPTH ESTIMATION ###
#########################


# imgL = cv2.imread('opencv_frame_ref_1.png',0)
# imgR = cv2.imread('opencv_frame_ref_2.png',0)

# # stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity,'gray')
# plt.show()











