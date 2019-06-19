import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from utils import *
# from keras.models import Model
# from keras.layers import *


##---------------crop--------------------

# images = glob('*.png')
# image_counter = 0

# for fname in images:
#     img = cv2.imread(fname,0)
#     crop_img = img[32:32+416, 112:112+416]

#     img_name = "crop_{}.png".format(image_counter)
#     cv2.imwrite(img_name, crop_img)
#     image_counter = image_counter +1
#     # cv2.imshow("cropped", crop_img)
#     # cv2.waitKey(0)

##---------------save---------------------

# X1 =[]
# X2 =[]

# img_paths=glob('*.png')
# N = len(img_paths)

# for i in range(N):
# 	j = i+1
# 	while(j<N):
# 		img1 = cv2.imread(img_paths[i],0)
# 		img2 = cv2.imread(img_paths[j],0)
# 		X1.append(img1)
# 		X2.append(img2)
# 		j =j+1

# X1 = np.array(X1)
# X2 = np.array(X2)
# X1 = X1[:,:,:,np.newaxis]
# X2 = X2[:,:,:,np.newaxis]
# np.savez('data/unet', X1 = X1, X2 = X2)
# data = np.load('data/unet.npz')
# print(data.files)

##-------------------flow_to_depth------------------

essentialMatrix=[]
tvecs=[]
rvecs=[]
depth=[]
index=[]

im1 = np.load('data/unet.npz')['X1']
im2 = np.load('data/unet.npz')['X2']
flow = np.load('data/unet_out.npz')['y']

samples = im1.shape[0]
image_height = im1.shape[1]
image_width = im1.shape[2]
flow_height = flow.shape[1]
flow_width = flow.shape[2]
n = image_height * image_width

corr1 =[]
corr2 =[]

for i in range(samples):
	(iy, ix) = np.mgrid[0:image_height, 0:image_width]
	(fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
	fx = fx.astype(np.float64)
	fy = fy.astype(np.float64)
	fx += flow[i,:,:,0]
	fy += flow[i,:,:,1]
	fx = np.minimum(np.maximum(fx, 0), flow_width)
	fy = np.minimum(np.maximum(fy, 0), flow_height)
	points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
	xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
	corr1.append(points)
	corr2.append(xi)

corr1 = np.array(corr1)
corr2 = np.array(corr2)

Nn = len(corr1)
for i in range(Nn):
	e = essential_matrix(corr1[i],corr2[i])
	essentialMatrix.append(e)
	tvecs.append(returnT_fromE(e))
	rvecs.append(returnR1_fromE(e))
	
	# depth.append(return_depth(corr1[i],corr2[i],returnR1_fromE(e),returnT_fromE(e)))


np.savez('data/flow_to_depth', e = essentialMatrix, r = rvecs, t = tvecs, 
			depth = depth, corr1 = corr1, corr2 = corr2)
l = np.load('data/flow_to_depth.npz')
print(l.files)





