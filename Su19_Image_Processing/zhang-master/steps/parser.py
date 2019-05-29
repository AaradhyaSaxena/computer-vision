import numpy as np

# Here real data is the object co-ordinates, whereas the image points are the sensed co-ordinates
def parse_data(basepath="data/corners_", ext=".dat"):

    sensed = []
    for i in range(1, 6):
        sensed.append(np.loadtxt(basepath + str(i) + ext).reshape((256, 2)))

    return {
        'real': np.loadtxt(basepath + "real" + ext).reshape((256, 2)),
        'sensed': sensed
    }
