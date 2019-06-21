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
from optical_flow_toolkit import *

## (im1,im2,flow).shape=(416x416)
## num_corr= number of points taken to calculate depth
def return_corr_from_flow(im1, im2, flow):
    # samples = im1.shape[0]
    # image_height = im1.shape[1]
    # image_width = im1.shape[2]
    # flow_height = flow.shape[1] 
    # flow_width = flow.shape[2]
    # n = image_height * image_width
    samples = flow.shape[0]
    image_height = im1.shape[1]
    image_width = im1.shape[2]
    flow_height = flow.shape[1] 
    flow_width = flow.shape[2]
    n = image_height * image_width

    essentialMatrix=[]
    index=[]
    tvecs=[]
    rvecs=[]
    depth=[]

    corr1 =[]
    corr2 =[]

    for i in range(samples):

        (iy, ix) = np.mgrid[0:image_height, 0:image_width]
        (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
        fx = fx.astype(np.float64)
        fy = fy.astype(np.float64)
        fx += flow[i,:,:,0]
        fy += flow[i,:,:,1]
        #fx += flow[:,:,0]
        #fy += flow[:,:,1]
        #fx=np.clip(fx,0,flow_width)
        #fy=np.clip(fy,0,flow_height)
        fx = np.minimum(np.maximum(fx, 0), flow_width)
        fy = np.minimum(np.maximum(fy, 0), flow_height)
        fx_new=fx.reshape((fx.shape[0]*fx.shape[1],1))
        fy_new=fy.reshape((fy.shape[0]*fy.shape[1],1))
        ix_new=ix.reshape((ix.shape[0]*ix.shape[1],1))
        iy_new=iy.reshape((iy.shape[0]*iy.shape[1],1))

        points=np.concatenate([ix_new,iy_new],axis=-1)
        xi=np.concatenate([fx_new,fy_new],axis=-1)

        #points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
        #xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
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

    for i in range(samples):

        e = essentialMatrix[i]
        t = tvecs[i]
        r = rvecs[i]
        d = []
        q = 436
        for j in range(int(n/q)):
            print(i,j)
            selec1 = corr1[i,j*(q):(j+1)*q,:]
            selec2 = corr2[i,j*(q):(j+1)*q,:]
            d.append(return_depth(selec1,selec2,r,t))

        if(n%q!= 0):
            selec1 = corr1[i,(n/q)*q:(n/q)*q+(n%q),:]
            selec2 = corr2[i,(n/q)*q:(n/q)*q+(n%q),:]       
            d.append(return_depth(selec1,selec2,r,t))

        depth.append(d)

    depth = np.array(depth).T
    depth = depth.reshape((samples,image_width,image_height+1,1))
    depth = (depth +1)/2

    return depth,essentialMatrix,tvecs,rvecs,corr1,corr2

###-----------------------------------------------------------

f=read_flo_file("/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/flow/alley_1/frame_0001.flo")

im1 = cv2.imread("/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/alley_1/frame_0001.png",0)
im2 = cv2.imread("/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/alley_1/frame_0002.png",0)
f = f[np.newaxis,:,:,:]
im1 = im1[np.newaxis,:,:]
im2 = im2[np.newaxis,:,:]

essentialMatrix,tvecs,rvecs,depth,corr1,corr2 = return_corr_from_flow(im1, im2, f)

"""
Once you have a .flo file, you can create a color coding of it using
color_flow

Use colortest to visualize the encoding


To compile

cd imageLib
make
cd ..
make
./colortest 10 colors.png
"""


# flo_paths=glob("/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/flow/alley_1")

# cropped_flow =[]
# flo_counter=0
# for flo in flo_paths:
# 	flow = read_flow(flo)

# 	crop_flo = flow[10:10+416, 304:304+416,:]

# 	cropped_flow.append(crop_flo)
# 	flo_name = "flo_crop_%d.flo"%(flo_counter)
# 	cv2.imwrite(flo_name, crop_flo)
# 	flo_counter = flo_counter +1
#  	# cv2.imshow("cropped", crop_flo)
#  	# # cv2.waitKey(0)



# cropped_flow = np.array(cropped_flow)




































# image_counter=0
# images = glob('*.png')
# for fname in images:
# 	img = cv2.imread(fname,0)
# 	# crop_img = img[10:10+416, 304:304+416]

# 	# img_name = "crop_{}.png".format(image_counter)
# 	# cv2.imwrite(img_name, crop_img)
# 	# image_counter = image_counter +1
# 	# cv2.imshow("cropped", crop_img)
# 	# cv2.waitKey(0)

# np.savez('flow_data', flow = cropped_flow)
# fl = np.load('flow_data.npz')

# fl.files