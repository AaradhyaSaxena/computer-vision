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

grid_x=21
grid_y=15
grid=(21,15)

objpoints = [] 
imgpoints = []

 
cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        obj_pt3x4 = return_objpoints(grid)
        ret,img_pt, corners = return_imgGreypoints(gray,grid)
        if(not ret):
            cv2.imshow('frame',gray)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue
 
        cv2.drawChessboardCorners(gray, grid, corners,True)
    
        P = projection_matrix4(img_pt,obj_pt3x4)
     
        proj_px = img_projection(P, np.array([[0,0,0],[0,0,10],[10,0,0],[0,10,0]]))
        proj_px = np.clip(proj_px/proj_px[2],0,1000)   
        proj_px=proj_px.astype("uint8")
        print(proj_px)



        cv2.line(gray,tuple(proj_px.T[0][:2]),tuple(proj_px.T[1][:2]),(0,255,0),5)
        cv2.line(gray,tuple(proj_px.T[0][:2]),tuple(proj_px.T[2][:2]),(0,255,0),5)
        cv2.line(gray,tuple(proj_px.T[0][:2]),tuple(proj_px.T[3][:2]),(0,255,0),5)

        cv2.imshow('frame',gray)
        if cv2.waitKey(10) & 0xFF == ord('q'):
               break

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()


