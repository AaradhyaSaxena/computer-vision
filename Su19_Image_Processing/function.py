import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from sklearn import linear_model, datasets


# output = 3x3 projection_matrix, input image shape=(num_points,2); obj shape=(num_points,3)
def projection_matrix_direct(imgp1, objp):

    imgp = np.ones((315,3))
    imgp[:,0] = imgp1[:,0]
    imgp[:,1] = imgp1[:,1]
    imgp = imgp.T

    objp = (objp.T).reshape((3,315))
    obj = np.ones((3,315))
    obj[:2,:] = objp[:2,:]
    
    ab = np.matmul(obj,obj.T)
    abinv = inv(ab)
    abc = np.matmul(obj.T,abinv)
    camera_matrix = np.matmul(imgp,abc)
    
    return camera_matrix

# output = 3x3 projection_matrix, input image shape=(num_points,2); obj shape=(num_points,3) 
def projection_matrix3(img_p, obj_p):
    
    C = []

    for i in range(315):
        C.append(np.array([obj_p[i,0], obj_p[i,1],1,0,0,0, (-1)*obj_p[i,0]*img_p[i,0], 
                           (-1)*obj_p[i,1]*img_p[i,0], (-1)*img_p[i,0]]))
        C.append(np.array([0,0,0, obj_p[i,0], obj_p[i,1],1, 
                           (-1)*obj_p[i,0]*img_p[i,1], (-1)*obj_p[i,1]*img_p[i,1],(-1)*img_p[i,1]]))
    
    c = np.array(C)
    ctc = np.matmul(c.T,c)
    u, s, vh = np.linalg.svd(ctc, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)
    H = H/H[-1,-1]
    
    return H

# output = 3x4 projection_matrix, input image shape=(num_points,2); obj shape=(num_points,3) 
def projection_matrix4(img_p, obj_p):
    
    C = []

    for i in range(315):
        C.append(np.array([obj_p[i,0], obj_p[i,1],1,0,0,0, (-1)*obj_p[i,0]*img_p[i,0], 
                           (-1)*obj_p[i,1]*img_p[i,0], (-1)*img_p[i,0]]))
        C.append(np.array([0,0,0, obj_p[i,0], obj_p[i,1],1, 
                           (-1)*obj_p[i,0]*img_p[i,1], (-1)*obj_p[i,1]*img_p[i,1],(-1)*img_p[i,1]]))
    
    c = np.array(C)
    ctc = np.matmul(c.T,c)
    u, s, vh = np.linalg.svd(ctc, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)
    H = H/H[-1,-1]
    
    return H



#returns (num_corners,2)
def return_imagepoints(image_path="./a.png"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (21,15),None)
    if(ret==False):
        print("image/corner doesnt exist")
    else:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpt = np.ones((21*15,2))
        imgpt[:,0] = corners[:,0,0]
        imgpt[:,1] = corners[:,0,1]
        
        return imgpt
    

#returns (num_obj,3)
def return_objpoints():
    objp = np.zeros((21*15,3), np.float32)
    objp[:,:2] = np.mgrid[0:21,0:15].T.reshape(-1,2)

    return objp


def homo_img(img_p):
    img_homo = np.ones((3,315))
    img_homo[0,:] = img_p[:,0]
    img_homo[1,:] = img_p[:,1]
    
    return img_homo

def homo_obj3(obj_p):
    obj_homo3 = np.ones((3,315))
    obj_homo3[0,:] = obj_p[:,0]
    obj_homo3[1,:] = obj_p[:,1]
    
    return obj_homo3

def homo_obj4(obj_p):
    obj_homo4 = np.ones((4,315))
    obj_homo4[0,:] = obj_p[:,0]
    obj_homo4[1,:] = obj_p[:,1]
    obj_homo4[2,:] = obj_p[:,2]
    
    return obj_homo4    


img_p=return_imagepoints()
obj_p=return_objpoints()
P=projection_matrix3(img_p,obj_p)

print(P)


