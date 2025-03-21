import os
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matchPics import matchPics
from helper import plotMatches

resultsdir = "results/rotTest"

if __name__ == "__main__":
    os.makedirs(resultsdir, exist_ok=True)

    # Read the image and convert to grayscale, if necessary
    originalImg = cv2.imread("data/cv_cover.jpg")
    if originalImg is None:
        print("Error: Could not load image. Check file path.")
        exit(1)
        
    rotImg = originalImg.copy()

    # Histogram count for matches
    nMatches = []
    angles = [(i+1)*10 for i in range(36)]  # 10, 20, ..., 360 degrees

    # Store angles and counts for 0, 90, 180 degrees for report
    special_angles = [0, 90, 180]
    special_visualizations = []

    print("Testing BRIEF descriptor with rotations...")
    
    # Add 0 degree rotation (identity) at the beginning
    angles.insert(0, 0)
    
    for i, angle in enumerate(angles):
        # Rotate Image
        if angle == 0:
            rotImg = originalImg.copy()  # No rotation
        else:
            rotImg = scipy.ndimage.rotate(originalImg, angle, reshape=True)
  
        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(originalImg, rotImg)
  
        # Update histogram
        nMatches.append(len(matches))

        # Save all results
        saveTo = os.path.join(resultsdir, f"rot{angle}.png")
        plotMatches(originalImg, rotImg, matches, locs1, locs2, saveTo=saveTo, showImg=False)
        
        # Check if this is one of our special visualization angles
        if angle in special_angles or (i == 0 and 0 in special_angles):
            special_visualizations.append((angle, saveTo))
            
        print(f"Angle {angle}°: {len(matches)} matches")

    # Display histogram
    plt.figure(figsize=(12, 6))
    plt.bar(x=angles, height=nMatches, width=5)
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Number of Matches")
    plt.title("Histogram of Matches vs Rotation Angle")
    plt.xticks(np.arange(0, 361, 30))  # Show tick marks every 30 degrees
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save histogram
    histPath = os.path.join(resultsdir, "rotTest.png")
    plt.savefig(histPath)
    print(f"Saved histogram to {histPath}")
    
    # Display special visualizations
    for angle, path in special_visualizations:
        img = cv2.imread(path)
        cv2.imshow(f"Rotation {angle} degrees", img)
    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    