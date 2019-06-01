
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from sklearn import linear_model, datasets


# In[2]:


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((15*21,3), np.float32)
objp[:,:2] = np.mgrid[0:21,0:15].T.reshape(-1,2)


# In[3]:


objp.shape


# In[4]:


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('a.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (21,15),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (21,15), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(5000)
    
cv2.destroyAllWindows()


# In[7]:


print(corners.shape)


# In[8]:


imgpoints
imgp1 = np.array(imgpoints)
imgp1.reshape((315,2))

imagepoints = np.ones((315,3))
imagepoints[:,0] = imgp1[0,:,0,0]
imagepoints[:,1] = imgp1[0,:,0,1]


# In[9]:


imagepoints = imagepoints.T


# In[10]:


imagepoints[:,:5]


# In[11]:


objp = objp.T
print("objp.shape",objp.shape)
obj = np.zeros((4,315))
objl = np.ones((1,315))
obj[:2,:] = objp[:2,:]
obj[3,:] = objl
print("obj.shape",obj.shape)


# In[12]:


obj[:,:5]


# In[13]:


imgp = imagepoints


# In[14]:


testo = objp[:,:5]


# In[15]:


testi = imgp[:,:5]


# We now have the image and object points

# Arranging the matrix in "THAT" form

# In[16]:


# imgp[:,:6]
# obj[:,:6]
# imgp[0,0]


# In[15]:


A = []


# In[16]:


for i in range(315):
    A.append(np.array([obj[0,i], obj[1][i], obj[2][i],1,0,0,0,0, (-1)*obj[0,i]*imgp[0,i], (-1)*obj[1][i]*imgp[0,i], (-1)*obj[2][i]*imgp[0,i],(-1)*imgp[0,i]]))
    A.append(np.array([0,0,0,0, obj[0,i], obj[1][i], obj[2][i],1, (-1)*obj[0,i]*imgp[1,i], (-1)*obj[1][i]*imgp[1,i], (-1)*obj[2][i]*imgp[1,i],(-1)*imgp[1,i]]))


# In[17]:


A


# In[18]:


len(A)


# In[19]:


a = np.array(A)
# d.reshape((12,12))
a.shape


# In[20]:


# a[:6,:12]


# In[21]:


ata = np.matmul(a.T,a)
ata.shape


# In[22]:


u1, s1, vh1 = np.linalg.svd(ata, full_matrices=True)

print(u1.shape)
print(s1.shape)
print(vh1.shape)


# In[23]:


s1


# In[24]:


vh1


# In[25]:


# orig_candidate = vh1[:,11]


# In[26]:


candidate1 = vh1[10,:]


# In[27]:


p12 = candidate1
p12


# In[28]:


p12 = p12.reshape((3,4))
p12


# In[29]:


from numpy import linalg as LA

norm = LA.norm(p12,axis=0)
print(norm)
p = p12/norm
p


# In[30]:


obj.shape


# In[31]:


p.shape


# In[32]:


imagepred1 = np.matmul(p,obj)
imagepred1[:,:5]


# In[33]:


imagepoints[:,:5]


# In[34]:


error1 = imagepoints - imagepred1
error1[:,:5]


# high error

# In[35]:


ptp = np.matmul(p.T,p)
ptpinv = inv(ptp)
left = np.matmul(ptpinv,p.T)
left


# In[36]:


_norm = LA.norm(left)
leftN = left/_norm
leftN


# Predicting co-ordinates of objects

# In[37]:


predobj = np.matmul(leftN,imgp)
predobj[1,0:30]


# In[38]:


obj[1,0:30]


# Now trying the same things after excluding the Z

# In[18]:


objp = objp.T

print(objp.shape)
obj = np.zeros((3,315))
# objl = np.ones((1,315))
obj[:2,:] = objp[:2,:]
# obj[3,:] = objl
print(obj.shape)


# In[19]:


obj[:,:5]


# In[20]:


obj.shape


# In[27]:


C = []


# In[28]:


for i in range(315):
    C.append(np.array([obj[0,i], obj[1,i],1,0,0,0, (-1)*obj[0,i]*imgp[0,i], (-1)*obj[1,i]*imgp[0,i],(-1)*imgp[0,i]]))
    C.append(np.array([0,0,0, obj[0,i], obj[1,i],1, (-1)*obj[0,i]*imgp[1,i], (-1)*obj[1,i]*imgp[1,i],(-1)*imgp[1,i]]))


# In[29]:


C.shape


# In[30]:


len(C)


# In[31]:


C = np.array(C)
C.shape


# In[32]:


ctc = np.matmul(C.T,C)
# ctcinv = inv(ctc)
# ctcinv.shape


# In[33]:


print(ctc.shape)
# print(ctc)


# In[34]:


u, s, vh = np.linalg.svd(ctc, full_matrices=True)


# In[35]:


print(u.shape)
print(s.shape)
print(vh.shape)


# In[47]:


s
# vh


# In[49]:


L = vh[-1]
H = L.reshape(3, 3)
H


# In[ ]:


denormalised = np.dot( np.dot (np.linalg.inv(first_normalisation_matrix),H ), second_normalisation_matrix)
homography = denormalised / denormalised[-1, -1]
homography


# We need to normalize the matrix H

# In[41]:


from numpy import linalg as LA

norm0 = LA.norm(p,axis=0)
print(norm0)
p = p/norm0
norm1 = LA.norm(p,axis=1)
print(norm1)
p = p/norm1
p


# In[42]:


objp[:,:5]


# In[43]:


objpt = np.ones((3,315))
objpt[:2,:] = obj[:2,:]
objpt[:,:5]


# In[44]:


imgpre = np.matmul(p,objp)
imgpre


# In[45]:


imgp


# In[66]:


pinv = inv(p)

norm = LA.norm(pinv)
pinv = pinv/norm
print(norm)
pinv


# In[67]:


obj[:,:5]


# In[68]:


obj.shape


# In[69]:


check = np.matmul(pinv,imgp)
check


# ---------------------
