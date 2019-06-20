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

TAG_FLOAT = 202021.25

# def read_flo(file):
# 	assert type(file) is str, "file is not str %r" % str(file)
# 	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
# 	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
# 	f = open(file,'rb')
# 	flo_number = np.fromfile(f, np.float32, count=1)[0]
# 	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
# 	w = np.fromfile(f, np.int32, count=1)
# 	h = np.fromfile(f, np.int32, count=1)
# 	#if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
# 	data = np.fromfile(f, np.float32, count=2*w*h)
# 	# Reshape data into 3D array (columns, rows, bands)
# 	flow = np.resize(data, (int(h), int(w), 2))
# 	f.close()
# 	return flow
#------------------------------------


flo_paths=glob('*.flo')
N = len(flo_paths)



images = glob('*.png')
for fname in images:
for fname in range(1):
	img = cv2.imread(fname,0)
	print(img.shape)
    # img = cv2.imread(fname,0)

    # crop_img = img[32:32+416, 112:112+416]

    # img_name = "crop_{}.png".format(image_counter)
    # cv2.imwrite(img_name, crop_img)
    # image_counter = image_counter +1
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)



