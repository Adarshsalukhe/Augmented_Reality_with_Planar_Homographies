import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature

PATCHWIDTH = 9

def briefMatch(desc1, desc2, ratio=0.8):
	matches = skimage.feature.match_descriptors(desc1, desc2, 'hamming', cross_check=True, max_ratio=ratio)
	return matches

def plotMatches(im1, im2, matches, locs1, locs2, saveTo=None, showImg=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skimage.feature.plot_matched_features(
        image0=im1, 
        image1=im2, 
        keypoints0=locs1, 
        keypoints1=locs2, 
        matches=matches, 
        ax=ax, 
        keypoints_color='k', 
        matches_color='r', 
        only_matches=True, 
        alignment='horizontal'
    )
    if saveTo is not None:
        plt.savefig(saveTo)
        if not showImg:
            plt.close()
    if showImg:
        plt.show()
    return

def makeTestPattern(patchWidth, nbits):
	np.random.seed(0)
	compareX = patchWidth*patchWidth * np.random.random((nbits,1))
 
	compareX = np.floor(compareX).astype(int)
	np.random.seed(1)
	compareY = patchWidth*patchWidth * np.random.random((nbits,1))
	compareY = np.floor(compareY).astype(int)

	return (compareX, compareY)


def computePixel(img, idx1, idx2, width, center):
	halfWidth = width // 2
	col1 = idx1 % width - halfWidth
	row1 = idx1 // width - halfWidth
	col2 = idx2 % width - halfWidth
	row2 = idx2 // width - halfWidth
	return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0


def computeBrief(img, locs):

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    part1 = np.logical_and(halfWidth <= locs[:, 0], locs[:, 0] < m-halfWidth)
    part2 = np.logical_and(halfWidth <= locs[:, 1], locs[:, 1] < n-halfWidth)
    locs = locs[np.logical_and(part1, part2), :]

    zipped = np.column_stack((compareX, compareY))
    col1 = zipped[:, 0] % patchWidth - halfWidth
    row1 = zipped[:, 0] // patchWidth - halfWidth
    col2 = zipped[:, 1] % patchWidth - halfWidth
    row2 = zipped[:, 1] // patchWidth - halfWidth
    center0_row1 = np.add.outer(locs[:, 0], row1).astype(int)
    center1_col1 = np.add.outer(locs[:, 1], col1).astype(int)
    center0_row2 = np.add.outer(locs[:, 0], row2).astype(int)
    center1_col2 = np.add.outer(locs[:, 1], col2).astype(int)

    desc = np.zeros((locs.shape[0], zipped.shape[0]))
    desc[img[center0_row1, center1_col1] < img[center0_row2, center1_col2]] = 1

    return desc, locs


def corner_detection(im, sigma=0.15):
	# fast method
	result_img = skimage.feature.corner_fast(im, PATCHWIDTH, sigma)
	locs = skimage.feature.corner_peaks(result_img, min_distance=1)
	return locs

def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches,
                matches_color='r', only_matches=False):
    """Custom implementation of plot_matches for compatibility"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a composite image
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    height = max(h1, h2)
    width = w1 + w2
    
    composite = np.zeros((height, width), dtype=np.uint8)
    composite[:h1, :w1] = image1
    composite[:h2, w1:w1+w2] = image2
    
    ax.imshow(composite, cmap='gray')
    ax.axis('off')
    
    # Draw lines for matches
    for idx1, idx2 in matches:
        y1, x1 = keypoints1[idx1]
        y2, x2 = keypoints2[idx2]
        x2 += w1  # offset for second image
        
        ax.plot([x1, x2], [y1, y2], '-', color=matches_color, linewidth=1)
        ax.plot(x1, y1, '.', color=matches_color, markersize=5)
        ax.plot(x2, y2, '.', color=matches_color, markersize=5)
        
        
