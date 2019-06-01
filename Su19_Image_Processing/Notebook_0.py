
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from numpy import linalg as LA
from sklearn import linear_model, datasets
from function import *


# In[2]:


img_p, corners = return_imagepoints('./a.png',(21,15))
obj_p = return_objpoints((21,15))

obj_p4 = homo_obj4(obj_p)
obj_p3 = homo_obj3(obj_p)
img_p = homo_img(img_p)

print("obj_p3.shape", obj_p3.shape)
print("obj_p4.shape", obj_p4.shape)
print("img_p.shape", img_p.shape)


# In[3]:


# img_p[:2,:].T.shape
# obj_p[:2,:].T.shape
img_p[:2,:].shape
img_p[:,:5]


# In[4]:


obj_pt3 = obj_p3.T
obj_pt4 = obj_p4.T
img_pt = img_p.T
print("obj_pt.shape", obj_pt3.shape)
print("obj_pt.shape", obj_pt4.shape)
print("img_pt.shape", img_pt.shape)


# In[5]:


objpoints = [] 
imgpoints = []

imgpoints.append(corners)
objpoints.append(obj_p)

img = cv2.imread("./a.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

rx = [[1,0,0],[0,math.cos(rvecs[0][0]),math.sin(rvecs[0][0])],[0,(-1)*math.sin(rvecs[0][0]),math.cos(rvecs[0][0])]]

ry = [[math.cos(rvecs[0][1]),0,math.sin(rvecs[0][1])],[0,1,0],[(-1)*math.sin(rvecs[0][1]),0,math.cos(rvecs[0][1])]]

rz = [[math.cos(rvecs[0][2]),math.sin(rvecs[0][2]),0],[(-1)*math.sin(rvecs[0][2]),math.cos(rvecs[0][2]),0],[0,0,1]]

R1 = np.matmul(rx,ry)
R = np.matmul(R1,rz)
print("rotation_matrix",R)
print("tvecs",tvecs)

m = np.ones((3,4))
m[:,:3] = R
m[:,[3]]= np.array(tvecs)

cv_homography = np.matmul(mtx,m)

pcv = cv_homography/cv_homography[-1,-1]
pcv


# In[8]:


p = projection_matrix_direct(img_p,obj_pt3)
print(p,"\n")

p_ = projection_matrix3(img_pt,obj_pt3)
print(p_,"\n")

_p = projection_matrix4(img_pt,obj_pt4)
print(_p,"\n")

print(pcv)


# In[13]:


# proj_cv = img_projection(pcv, np.array([[10,20,100]]))
# print(proj_cv/proj_cv[2],'\n')

# _proj = img_projection(_p,np.array([[10,20,100]]))
# print(_proj/_proj[2],'\n')

p = img_projection(_p,obj_p[:5,:])
print(p/p[-1,-1],'\n')

# # print(p/p[2])

# print(img_p[:,:5],'\n')

# _proj = img_projection(_p,np.array([[1,2,1]]))
# print(_proj,'\n')

# print(_proj/_proj[2])


# In[12]:


img_p[:,:5]


# In[24]:


# a,b,c = RQ_decomposition(_p)
# print("camera_calibration_matrix",a)
# print("rotation_matrix",b)
# print("translation_matrix",c)


# NOTE: The (1,2) term is zero in a camera calibration matrix, which is not zero in our case.

# Manually decomposing the p-matrix by RQ decomposition.

# In[16]:


# # trying to get the camera matrix in proper form
# m = _p[:,:3]
# m1 = m[:,0]
# m2 = m[:,1]
# m3 = m[:,2]

# alpha1 = LA.norm(m1)
# alpha2 = LA.norm(m2)
# x0 = LA.norm(m3)
# y0 = LA.norm(m3)
# print(alpha1,alpha2,x0,y0)
# # kappa = [[alpha1,0,x0],[0,alpha2,y0],[0,0,1]]
# # kappa
# # print(_p[0])
# # print(_p[1])
# # print(_p[2])


# On manually calculating the RQ decomposition, in the k-matrix the value of all the variable comes out to be 136.9, this is because in all the columns of p-matrix, there is a dominting value (1.36) present.

# So basically we can't predict z co-ordinates by _p, because the 3rd row contains too large numbers. And we normalize by LA.norm(..,axis=0), we wouldn't get proper estimation of points.

# In[ ]:


# def return_objpoints1():
#     objp = np.ones((21*15,3), np.float32)
#     objp[:,:2] = np.mgrid[0:21,0:15].T.reshape(-1,2)

#     return objp1

# We observe that because in all our object images the points of z-axis are 0,
# in the camera calibration matrix the points associated to z-coordinate explode,
# as on increasing or decreasing those points doesn't really matter.
# To counter this we thought if we add a non-zero point to all Z co-ordinate, we might get a better result.
# But on adding a constant to all (Z), the value of x0,y0 increases rather than decreasing z co-efficients.


# In[47]:


# img_p, corners = return_imagepoints()
# obj_p = return_objpoints()

# obj_p4 = homo_obj4(obj_p)
# obj_p3 = homo_obj3(obj_p)
# img_p = homo_img(img_p)

# print("obj_p3.shape", obj_p3.shape)
# print("obj_p4.shape", obj_p4.shape)
# print("img_p.shape", img_p.shape)

# # img_p[:2,:].T.shape
# # obj_p[:2,:].T.shape
# img_p[:2,:].shape

# obj_pt3 = obj_p3.T
# obj_pt4 = obj_p4.T
# img_pt = img_p.T
# print("obj_pt.shape", obj_pt3.shape)
# print("obj_pt.shape", obj_pt4.shape)
# print("img_pt.shape", img_pt.shape)


# In[46]:


# img_p.shape
# # obj_p = obj_p4
# # obj_p4.shape


# --------------------------------------

# ---------------------------------------
