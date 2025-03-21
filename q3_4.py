import os
import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches

def main():
    # Create root directory path based on current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Load input images with absolute paths
    cv_cover_path = os.path.join(parent_dir, 'data', 'cv_cover.jpg')
    cv_desk_path = os.path.join(parent_dir, 'data', 'cv_desk.png')
    
    cv_cover = cv2.imread(cv_cover_path)
    cv_desk = cv2.imread(cv_desk_path)
    
    # Check if images loaded correctly
    if cv_cover is None or cv_desk is None:
        print(f"Error: Could not load input images.")
        print(f"Tried to load from: {cv_cover_path} and {cv_desk_path}")
        return
    
    # Create results directory
    resultsdir = os.path.join(parent_dir, "results", "matchPics")
    os.makedirs(resultsdir, exist_ok=True)
    
    # Use stricter parameters to get fewer, higher-quality matches
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, ratio=0.68)
    
    # Print number of matches found
    print(f"Found {matches.shape[0]} matches between images")
    
    # Limit to a smaller number of matches for clearer visualization (optional)
    max_matches = 15  # Set to a reasonable number for visualization
    if matches.shape[0] > max_matches:
        matches = matches[:max_matches]
        print(f"Displaying {matches.shape[0]} match lines in visualization")
    
    # Save the visualization
    saveTo = os.path.join(resultsdir, "matchPics.png")
    plotMatches(cv_cover, cv_desk, matches, locs1, locs2, saveTo=saveTo, showImg=True)
    
    print(f"Saved match visualization to {saveTo}")

if __name__ == "__main__":
    main()