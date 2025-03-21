import numpy as np
import cv2
import os
from matchPics import matchPics

if __name__ == "__main__":
    # Create results directory and load images
    os.makedirs("results", exist_ok=True)
    cv_cover = cv2.imread('data/cv_cover.jpg')
    cv_desk = cv2.imread('data/cv_desk.png')
    hp_cover = cv2.imread('data/hp_cover.jpg')
    
    if any(img is None for img in [cv_cover, cv_desk, hp_cover]):
        print("Error: Could not load one or more images")
        exit(1)
    
    # Resize hp_cover to match cv_cover dimensions
    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    
    # Find feature matches
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, ratio=0.75)
    
    # Extract and convert matched points
    x1 = np.array([(int(locs1[m[0]][1]), int(locs1[m[0]][0])) for m in matches])
    x2 = np.array([(int(locs2[m[1]][1]), int(locs2[m[1]][0])) for m in matches])
    
    # Compute homography
    H, _ = cv2.findHomography(x1, x2, cv2.RANSAC, 5.0)
    
    # Create composite using warpPerspective directly
    h, w = cv_desk.shape[:2]
    warped_hp = cv2.warpPerspective(hp_cover, H, (w, h))
    
    # Create mask and apply it
    mask = cv2.warpPerspective(np.ones_like(hp_cover), H, (w, h))
    result = cv_desk.copy()
    result = result * (1 - mask) + warped_hp
    
    # Save result
    cv2.imwrite("results/harrypotterized.jpg", result)
    print("Result saved to results/harrypotterized.jpg")