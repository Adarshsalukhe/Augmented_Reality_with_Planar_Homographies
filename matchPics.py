import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, ratio=0.8, sigma=0.15):
    # Convert images to grayscale
    if len(I1.shape) == 3:
        I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    else:
        I1_gray = I1.copy()
    
    if len(I2.shape) == 3:
        I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    else:
        I2_gray = I2.copy()
    
    # Detect features using corner_detection
    locs1 = corner_detection(I1_gray, sigma)
    locs2 = corner_detection(I2_gray, sigma)
    
    # Build descriptors using computeBrief
    desc1, locs1 = computeBrief(I1_gray, locs1)
    desc2, locs2 = computeBrief(I2_gray, locs2)
    
    # Match features using briefMatch
    matches = briefMatch(desc1, desc2, ratio)
    
    return matches, locs1, locs2

def matchPicsCached(I1, I2, ratio=0.7,
                    cachedLocs1=None, cachedDesc1=None,
                    cachedLocs2=None, cachedDesc2=None):
    # Convert Images to GrayScale
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Detect Features in Both Images
    locs1 = corner_detection(I1_gray) if cachedLocs1 is None else cachedLocs1
    locs2 = corner_detection(I2_gray) if cachedLocs2 is None else cachedLocs2

    # Obtain descriptors for the computed feature locations
    # Use cached data if possible to avoid recomputation
    desc1, locs1 = computeBrief(I1_gray, locs1) \
        if (cachedLocs1 is None and cachedDesc1 is None) \
        else (cachedDesc1, cachedLocs1)

    desc2, locs2 = computeBrief(I2_gray, locs2) \
        if (cachedLocs2 is None and cachedDesc2 is None) \
        else (cachedDesc2, cachedLocs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2, desc1, desc2

def convert_features(matches, locs1, locs2):

    # Create arrays to store the converted features
    x1 = np.zeros((matches.shape[0], 2), dtype=np.float32)
    x2 = np.zeros((matches.shape[0], 2), dtype=np.float32)
    
    # Convert coordinate format - matchPics returns (y,x) but computeH_ransac expects (x,y)
    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]
        
        # Swap y,x to x,y
        x1[i, 0] = locs1[idx1, 1]
        x1[i, 1] = locs1[idx1, 0]
        x2[i, 0] = locs2[idx2, 1]
        x2[i, 1] = locs2[idx2, 0]
    
    return x1, x2