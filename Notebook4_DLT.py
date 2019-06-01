
# coding: utf-8

# In[1]:


import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
import cv2
import glob

import math
from scipy import linalg
from numpy.linalg import inv
from sklearn import linear_model, datasets


# In[2]:


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((15*21,2), np.float32)
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


# In[5]:


print(corners.shape)


# In[6]:


imgpoints
imgp1 = np.array(imgpoints)
imgp1.reshape((315,2))

imagepoints = np.ones((315,3))
imagepoints[:,0] = imgp1[0,:,0,0]
imagepoints[:,1] = imgp1[0,:,0,1]

imagepoints = imagepoints.T

print(imagepoints[:,:5])

objp = objp.T
print(objp.shape)
obj = np.ones((3,315))
obj[:2,:] = objp[:2,:]
print(obj.shape)


# In[7]:


real = obj[:2,:]
sensed = imagepoints[:2,:]
real[:2,:5]
sensed[:2,:5]


# In[8]:


def get_normalisation_matrix(flattened_corners):
#     end = timer()

    avg_x = flattened_corners[0].mean()
    avg_y = flattened_corners[1].mean()

    s_x = np.sqrt(2 / flattened_corners[0].std())
    s_y = np.sqrt(2 / flattened_corners[1].std())

#     end("get_normalization_matrix")
    return np.matrix([
        [s_x,   0,   -s_x * avg_x],
        [0,   s_y,   -s_y * avg_y],
        [0,     0,              1]
    ])


# In[9]:


rnm = get_normalisation_matrix(real)
rnm


# In[10]:


inm = get_normalisation_matrix(sensed)
inm


# In[11]:


real_ = np.ones((3,315))
real_[:2,:] = real
print(real_.shape)
sensed_ = np.ones((3,315))
sensed_[:2,:] = sensed
print(sensed_.shape)


# Getting normalized matrix of real and sensed co-ordinates

# In[12]:


real_obj = np.dot(rnm,real_)
sensed_im = np.dot(inm,sensed_)
# real_obj[:,:5]
# sensed_im[:,:5]


# In[13]:


print(real_obj.shape)
print(sensed_im.shape)
print(obj.shape)
# print(imgp.shape)


# In[14]:


obj = real_obj
imgp = sensed_im


# In[22]:


obj[:,:5]


# In[16]:


C = []

for i in range(315):
    C.append(np.array([obj[0,i], obj[1,i],1,0,0,0, (-1)*obj[0,i]*imgp[0,i], (-1)*obj[1,i]*imgp[0,i],(-1)*imgp[0,i]]))
    C.append(np.array([0,0,0, obj[0,i], obj[1,i],1, (-1)*obj[0,i]*imgp[1,i], (-1)*obj[1,i]*imgp[1,i],(-1)*imgp[1,i]]))


# In[17]:


C = np.array(C)
print(C.shape)

ctc = np.matmul(C.T,C)
print(ctc.shape)

u, s, vh = np.linalg.svd(ctc, full_matrices=True)
print(u.shape)
print(s.shape)
print(vh.shape)


# In[18]:


s
# vh


# In[19]:


L = vh[-1]
H = L.reshape(3, 3)
H


# In[20]:


denormalised = np.dot( np.dot (np.linalg.inv(rnm),H ), inm)
estimated_homography = denormalised / denormalised[-1, -1]
estimated_homography


# We have calculated the HOMOGRAPHY matrix

# ------------------------------

# Now we will optimize the estimated_homography matrix, but first we should check how accurate the current homography matrix is in transforming the real points

# In[119]:


homography = estimated_homography.tolist()

homo =[]
for i in range(3):
    for j in range(3):
        homo.append(homography[i][j])


# In[151]:


def cost(homography, real):
#     [sensed, real] = data

#     aX = []
#     aY = []
    Y = []

    for i in range(len(real[0])):
        x = sensed[0][i]
        y = sensed[1][i]

        w = homography[6] * x + homography[7] * y + homography[8]

        M = np.array([
            [homography[0], homography[1], homography[2]],
            [homography[3], homography[4], homography[5]]
        ])

        homog = np.transpose(np.array([x, y, 1]))
        [u, v] = (1/w) * np.dot(M, homog)

#         aX.append(u)
#         aY.append(v)
        Y.append(u)
        Y.append(v)

#     return aX,aY
    return np.array(Y)


# In[152]:


# Y = cost(homo,real[:2,:])
# Y
# sensed


# In[153]:


def jac(homography, real):
#     [sensed, real] = data

    J = []

    for i in range(len(real[0])):
        x = real[0][i]
        y = real[1][i]

        s_x = homography[0] * x + homography[1] * y + homography[2]
        s_y = homography[3] * x + homography[4] * y + homography[5]
        w = homography[6] * x + homography[7] * y + homography[8]

        J.append(
            np.array([
                x / w, y / w, 1/w,
                0, 0, 0,
                (-s_x * x) / (w*w), (-s_x * y) / (w*w), -s_x / (w*w)
            ])
        )

        J.append(
            np.array([
                0, 0, 0,
                x / w, y / w, 1 / w,
                (-s_y * x) / (w*w), (-s_y * y) / (w*w), -s_y / (w*w)
            ])
        )

    return np.array(J)


# In[154]:


def refine_homography(homography, real, sensed):
    return opt.root(
        cost,
        homo,
        jac=jac,
        args=[real, sensed],
        method='lm'
    ).x


# In[157]:


# def compute_homography(real, sensed):
#     end = timer()

#     real = data['real']

#     refined_homographies = []

#     for i in range(0, len(data['sensed'])):
#         sensed = data['sensed'][i]
#         estimated = estimate_homography(real, sensed)
#         end = timer()
#         refined = refine_homography(estimated, sensed, real)
#         refined = refined / refined[-1]
#         end("refine_homography")
#         refined_homographies.append(refined)

#     end("compute_homography")
#     return np.array(refined_homographies)
# #     return 0


# In[158]:


refined = refine_homography(homo, real, sensed)
refined = refined / refined[-1]


# ----------------------------------

# ----------------------------------

# Checkin by ro and si

# In[21]:


ro = np.array([[1,2,4],[1,3,5]])
si = np.array([[100,200,400],[120,360,600]])

si[1].std()
s_x = np.sqrt(2 / ro[1].std())
s_x


# In[22]:


first_normalisation_matrix = get_normalisation_matrix(ro)
first_normalisation_matrix


# In[23]:


second_normalisation_matrix = get_normalisation_matrix(si)
second_normalisation_matrix


# In[24]:


ro_ = np.array([[1,2,4],[1,3,5],[1,1,1]])
si_ = np.array([[100,200,400],[120,360,600],[1,1,1]])


# In[25]:


# fnm = np.ones((315,3))
# fnm[:,:2] = real
# print(fnm[:,:5])
# snm = np.ones((315,3))
# snm[:,:2] = sensed
# print(snm[:,:5])


# In[26]:


pr_1 = np.dot(first_normalisation_matrix,ro_)
pr_2 = np.dot(second_normalisation_matrix,si_)
pr_1
pr_2


# In[27]:


obj = pr_1
imgp = pr_2


# In[28]:


M = []

for i in range(3):
    M.append(np.array([obj[0,i], obj[1,i],1,0,0,0, (-1)*obj[0,i]*imgp[0,i], (-1)*obj[1,i]*imgp[0,i],(-1)*imgp[0,i]]))
    M.append(np.array([0,0,0, obj[0,i], obj[1,i],1, (-1)*obj[0,i]*imgp[1,i], (-1)*obj[1,i]*imgp[1,i],(-1)*imgp[1,i]]))


# In[29]:


M


# In[30]:


M = np.array(M)
print(M.shape)

mtm = np.matmul(M.T,M)
print(mtm.shape)

u, s, vh = np.linalg.svd(mtm, full_matrices=True)
print(u.shape)
print(s.shape)
print(vh.shape)


# In[31]:


L = vh[-1]
H = L.reshape(3, 3)
H


# In[35]:


im = np.dot(H,ro_)
im


# In[36]:


si_


# Not working so well ! , ........trying after denormalization of H

# In[37]:


denormalised = np.dot( np.dot (np.linalg.inv(first_normalisation_matrix),H ), second_normalisation_matrix)
estimated_homography = denormalised / denormalised[-1, -1]
estimated_homography


# Image_points = homography X object_points

# In[33]:


imdeno = np.dot(estimated_homography,ro_)
imdeno


# In[34]:


si_


# Poor result, Estimated homogaphy ain't good enough!

# In[ ]:


from steps.parser import parse_data
from steps.dlt import compute_homography
from steps.intrinsics import get_camera_intrinsics
from steps.extrinsics import get_camera_extrinsics
from steps.distortion import estimate_lens_distortion
from utils.timer import timer


def calibrate():
    data = parse_data()

    end = timer()
    homographies = compute_homography(data)
    end("Homography Estimation")
    print("homographies")
    print(homographies)

    end = timer()
    intrinsics = get_camera_intrinsics(homographies)
    end("Intrinsics")

    print("intrinsics")
    print(intrinsics)

    end = timer()
    extrinsics = get_camera_extrinsics(intrinsics, homographies)
    end("Extrinsics")

    print("extrinsics")
    print(extrinsics)

    end = timer()
    distortion = estimate_lens_distortion(
        intrinsics,
        extrinsics,
        data["real"],
        data["sensed"]
    )
    end("Distortion")

    return 0

# calibrate()


# In[138]:


# here the data is kept in different order,
# they have the x coord followed by y, in our case we have separate list for x and y

def cost(homography, data):
    [sensed, real] = data

    Y = []

    for i in range(0, sensed.size / 2):
        x = sensed[i][0]
        y = sensed[i][1]

        w = homography[6] * x + homography[7] * y + homography[8]

        M = np.array([
            [homography[0], homography[1], homography[2]],
            [homography[3], homography[4], homography[5]]
        ])

        homog = np.transpose(np.array([x, y, 1]))
        [u, v] = (1/w) * np.dot(M, homog)

        Y.append(u)
        Y.append(v)

    return np.array(Y)


# In[ ]:


def refine_homography(homography, sensed, real):
    return opt.root(
        cost,
        homography,
        jac=jac,
        args=[sensed, real],
        method='lm'
    ).x


# In[ ]:


def jac(homography, data):
    [sensed, real] = data

    J = []

    for i in range(0, sensed.size / 2):
        x = sensed[i][0]
        y = sensed[i][1]

        s_x = homography[0] * x + homography[1] * y + homography[2]
        s_y = homography[3] * x + homography[4] * y + homography[5]
        w = homography[6] * x + homography[7] * y + homography[8]

        J.append(
            np.array([
                x / w, y / w, 1/w,
                0, 0, 0,
                (-s_x * x) / (w*w), (-s_x * y) / (w*w), -s_x / (w*w)
            ])
        )

        J.append(
            np.array([
                0, 0, 0,
                x / w, y / w, 1 / w,
                (-s_y * x) / (w*w), (-s_y * y) / (w*w), -s_y / (w*w)
            ])
        )

    return np.array(J)


# In[83]:


def compute_homography(real, sensed):
    end = timer()

    real = data['real']

    refined_homographies = []

    for i in range(0, len(data['sensed'])):
        sensed = data['sensed'][i]
        estimated = estimate_homography(real, sensed)
        end = timer()
        refined = refine_homography(estimated, sensed, real)
        refined = refined / refined[-1]
        end("refine_homography")
        refined_homographies.append(estimated)

    end("compute_homography")
    return np.array(refined_homographies)
    return 0


# In[24]:


import numpy as np
from scipy import optimize as opt
from utils.timer import timer


def get_normalisation_matrix(flattened_corners):
    end = timer()

    avg_x = flattened_corners[:, 0].mean()
    avg_y = flattened_corners[:, 1].mean()

    s_x = np.sqrt(2 / flattened_corners[0].std())
    s_y = np.sqrt(2 / flattened_corners[1].std())

    end("get_normalization_matrix")
    return np.matrix([
        [s_x,   0,   -s_x * avg_x],
        [0,   s_y,   -s_y * avg_y],
        [0,     0,              1]
    ])


def estimate_homography(first, second):
    end = timer()

    first_normalisation_matrix = get_normalisation_matrix(first)
    second_normalisation_matrix = get_normalisation_matrix(second)

    M = []

    for j in range(0, int(first.size / 2)):
        homogeneous_first = np.array([
            first[j][0],
            first[j][1],
            1
        ])

        homogeneous_second = np.array([
            second[j][0],
            second[j][1],
            1
        ])

        pr_1 = np.dot(first_normalisation_matrix, homogeneous_first)

        pr_2 = np.dot(second_normalisation_matrix, homogeneous_second)

        M.append(np.array([
            pr_1.item(0), pr_1.item(1), 1,
            0, 0, 0,
            -pr_1.item(0)*pr_2.item(0), -pr_1.item(1)*pr_2.item(0), -pr_2.item(0)
        ]))

        M.append(np.array([
            0, 0, 0, pr_1.item(0), pr_1.item(1),
            1, -pr_1.item(0)*pr_2.item(1), -pr_1.item(1)*pr_2.item(1), -pr_2.item(1)
        ]))

    U, S, Vh = np.linalg.svd(np.array(M).reshape((512, 9)))

    L = Vh[-1]

    H = L.reshape(3, 3)

    denormalised = np.dot(
        np.dot(
            np.linalg.inv(first_normalisation_matrix),
            H
        ),
        second_normalisation_matrix
    )

    end("estimate_homography")
    return denormalised / denormalised[-1, -1]


def cost(homography, data):
    [sensed, real] = data

    Y = []

    for i in range(0, sensed.size / 2):
        x = sensed[i][0]
        y = sensed[i][1]

        w = homography[6] * x + homography[7] * y + homography[8]

        M = np.array([
            [homography[0], homography[1], homography[2]],
            [homography[3], homography[4], homography[5]]
        ])

        homog = np.transpose(np.array([x, y, 1]))
        [u, v] = (1/w) * np.dot(M, homog)

        Y.append(u)
        Y.append(v)

    return np.array(Y)


def jac(homography, data):
    [sensed, real] = data

    J = []

    for i in range(0, sensed.size / 2):
        x = sensed[i][0]
        y = sensed[i][1]

        s_x = homography[0] * x + homography[1] * y + homography[2]
        s_y = homography[3] * x + homography[4] * y + homography[5]
        w = homography[6] * x + homography[7] * y + homography[8]

        J.append(
            np.array([
                x / w, y / w, 1/w,
                0, 0, 0,
                (-s_x * x) / (w*w), (-s_x * y) / (w*w), -s_x / (w*w)
            ])
        )

        J.append(
            np.array([
                0, 0, 0,
                x / w, y / w, 1 / w,
                (-s_y * x) / (w*w), (-s_y * y) / (w*w), -s_y / (w*w)
            ])
        )

    return np.array(J)


def refine_homography(homography, sensed, real):
    return opt.root(
        cost,
        homography,
        jac=jac,
        args=[sensed, real],
        method='lm'
    ).x

