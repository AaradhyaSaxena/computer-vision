import numpy as np
# from PIL import Image

from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from numpy import linalg as LA

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

## (im1,im2,flow).shape=(416x416)
## num_corr= number of points taken to calculate depth
def return_corr_from_flow(im1, im2, flow):
    samples = im1.shape[0]
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

    for i in range(samples):

        e = essentialMatrix[0]
        t = tvecs[0]
        r = rvecs[0]
        d = []
        q = 416
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

    depth = np.array(depth)
    depth = depth.reshape((samples,image_height,image_width+1,1))
    depth = (depth+1)/2 # pixel values: (-1,1)>>(0,1)

    return essentialMatrix,tvecs,rvecs,depth,corr1,corr2

def return_depth(homo_pt1,homo_pt2,r1,t,):

    # print(homo_pt1.shape)
    length = homo_pt1.shape[0]
    homo1 = np.ones((length,3))
    homo2 = np.ones((length,3))
    homo1[:,:2] = homo_pt1[:,:]
    homo2[:,:2] = homo_pt2[:,:]
    kk = np.load("data/parameters.npz") # k = kk['k_new']
    k = kk['k']
    homo_im1 = np.ones((length,3))
    homo_im2 = np.ones((length,3))
    homo_im1[:,:] = np.matmul(inv(k),homo1[:,:].T).T
    homo_im2[:,:] = np.matmul(inv(k),homo2[:,:].T).T

    # print(homo_im1.shape)
    a = np.matmul(homo_im2,r1)
    rot1 = np.matmul(a,homo_im1.T)
    trans1 = np.matmul(homo_im2,t.reshape((3,1)))
    
    A = np.hstack((rot1,trans1))
    ata = np.matmul(A.T,A)
    # print(homo_im1.shape)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    # print(homo_im1.shape)
    Depth = vh[-1].reshape(length+1,1)

    return Depth

##########################-------------------------


def flow_read(filename):
    """ Read optical flow from file, return (U,V) tuple. 
    
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]
    return u,v

def flow_write(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def cam_write(filename, M, N):
    """ Write intrinsic matrix M and extrinsic matrix N to file. """
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    M.astype('float64').tofile(f)
    N.astype('float64').tofile(f)
    f.close()

