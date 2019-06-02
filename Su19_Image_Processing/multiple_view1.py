import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from numpy import linalg as LA
from sklearn import linear_model, datasets
from utils import * 
# from utils_beta import *

grid_x=21
grid_y=15
grid=(21,15)

objpoints1 = [] 
imgpoints1 = []

image_path1 = './image/a.png'
image_path2 = './image/a2.png'

img1 = cv2.imread('./image/a.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('./image/a2.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  


obj_pt3x4_1 = return_objpoints(grid)
img_pt1, corners1 = return_imagepoints(image_path1,grid)
    
obj_pt3x4_2 = return_objpoints(grid)
img_pt2, corners2 = return_imagepoints(image_path2,grid)
   
# P1 = projection_matrix4(img_pt1,obj_pt3x4_1)
# P2 = projection_matrix4(img_pt2,obj_pt3x4_2)

P1 = return_pcv(image_path1,corners1,obj_pt3x4_1)
P2 = return_pcv(image_path2,corners2,obj_pt3x4_2)

F = return_fundamentalM(P1, P2)

print(F)



















































# cap = cv2.VideoCapture(0)
# while(True):
#     try:
#         ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    #     obj_pt3x4 = return_objpoints(grid)
    #     ret,img_pt, corners = return_imagepoints(gray,grid)
    #     if(not ret):
    #         cv2.imshow('frame',gray)
    #         if cv2.waitKey(10) & 0xFF == ord('q'):
    #             break
    #         continue
 
    #     cv2.drawChessboardCorners(gray, grid, corners,True)
    
    #     P = projection_matrix4(img_pt,obj_pt3x4)
     
    #     proj_px = img_projection(P, np.array([[0,0,0],[0,0,10],[10,0,0],[0,10,0]]))
    #     proj_px = np.clip(proj_px/proj_px[2],0,1000)   
    #     proj_px=proj_px.astype("uint8")
    #     print(proj_px)



    #     cv2.line(gray,tuple(proj_px.T[0][:2]),tuple(proj_px.T[1][:2]),(0,255,0),5)
    #     cv2.line(gray,tuple(proj_px.T[0][:2]),tuple(proj_px.T[2][:2]),(0,255,0),5)
    #     cv2.line(gray,tuple(proj_px.T[0][:2]),tuple(proj_px.T[3][:2]),(0,255,0),5)

    #     cv2.imshow('frame',gray)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #            break

    # except KeyboardInterrupt:
    #     cap.release()
    #     cv2.destroyAllWindows()


# cap.release()
# cv2.destroyAllWindows()