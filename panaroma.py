import numpy as np
import cv2
import os
from matchPics import matchPics, convert_features
from planarH import computeH_ransac

# Image paths
LEFT_IMAGE_PATH = "data/pano_left.jpg"
RIGHT_IMAGE_PATH = "data/pano_right.jpg"
RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "panorama.jpg")

def find_overlap(left_img, right_img):
    # Convert to grayscale for comparison
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    h, w = left_gray.shape
    best_offset = 0
    best_score = float('-inf')
    
    # Try different offsets for the right image - search more thoroughly
    # First pass with larger steps to find approximate region
    possible_offsets = []
    
    for offset in range(w//4, 3*w//4, 10):  # Try wider range with bigger steps
        # The overlapping region
        left_region = left_gray[:, w-offset:w]
        right_region = right_gray[:, 0:offset]
        
        # Only compare regions with the same shape
        min_h = min(left_region.shape[0], right_region.shape[0])
        min_w = min(left_region.shape[1], right_region.shape[1])
        
        if min_h > 0 and min_w > 0:
            # Take sample points for faster calculation
            sample_rows = np.linspace(0, min_h-1, 20).astype(int)
            sample_cols = np.linspace(0, min_w-1, 20).astype(int)
            
            left_samples = left_region[np.ix_(sample_rows, sample_cols)]
            right_samples = right_region[np.ix_(sample_rows, sample_cols)]
            
            # Use normalized cross-correlation for better matching
            left_norm = left_samples - np.mean(left_samples)
            right_norm = right_samples - np.mean(right_samples)
            
            left_std = np.std(left_samples)
            right_std = np.std(right_samples)
            
            if left_std > 0 and right_std > 0:
                correlation = np.mean((left_norm / left_std) * (right_norm / right_std))
                
                if correlation > 0.2:  # Only consider reasonable correlations
                    possible_offsets.append((offset, correlation))
    
    # If we found possible matches, refine with smaller steps
    if possible_offsets:
        # Sort by correlation score and take top 3
        possible_offsets.sort(key=lambda x: x[1], reverse=True)
        best_candidates = possible_offsets[:3]
        
        # Fine search around the best candidates
        for base_offset, _ in best_candidates:
            search_range = 20  # +/- 20 pixels
            for offset in range(max(10, base_offset - search_range), 
                                min(w-10, base_offset + search_range + 1)):
                # The overlapping region
                left_region = left_gray[:, w-offset:w]
                right_region = right_gray[:, 0:offset]
                
                min_h = min(left_region.shape[0], right_region.shape[0])
                min_w = min(left_region.shape[1], right_region.shape[1])
                
                if min_h > 0 and min_w > 0:
                    # Calculate pixel-wise absolute difference
                    left_region = left_region[:min_h, :min_w]
                    right_region = right_region[:min_h, :min_w]
                    
                    diff = np.abs(left_region.astype(np.float32) - right_region.astype(np.float32))
                    score = -np.mean(diff)  # Negative mean difference (higher is better)
                    
                    if score > best_score:
                        best_score = score
                        best_offset = offset
    
    # If no good match found, use a reasonable default
    if best_offset < w//5:
        best_offset = w//3
        print("Using default offset as no reliable match found")
    
    return best_offset

def create_better_panorama(left_img, right_img):
    """Create a panorama with advanced overlap detection and blending."""
    # Get dimensions
    h_left, w_left = left_img.shape[:2]
    h_right, w_right = right_img.shape[:2]
    
    # Find the overlapping region
    overlap = find_overlap(left_img, right_img)
    print(f"Detected overlap: {overlap} pixels")
    
    # If no significant overlap detected, use a default
    if overlap < 20:
        overlap = min(w_left // 3, w_right // 3)
        print(f"Using default overlap: {overlap} pixels")
    
    # Calculate the width of the panorama
    width = w_left + w_right - overlap
    height = max(h_left, h_right)
    
    # Create the panorama
    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add the left image
    panorama[0:h_left, 0:w_left] = left_img
    
    # Create a smoother mask for blending in the overlapping region
    mask = np.zeros((height, width))
    mask[:, :w_left-overlap] = 1
    
    # Use a cosine blending function for smoother transition
    for i in range(overlap):
        # Cosine blend weight (smoother than linear)
        ratio = i / overlap
        weight = 0.5 * (np.cos(np.pi * ratio) + 1)
        mask[:, w_left-overlap+i] = weight
    
    # Create right image canvas
    right_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    right_canvas[0:h_right, (w_left-overlap):(w_left-overlap)+w_right] = right_img
    
    # Blend the images
    for c in range(3):  # For each color channel
        panorama[:,:,c] = mask * panorama[:,:,c] + (1 - mask) * right_canvas[:,:,c]
    
    # Detect and remove any seam artifacts by applying a small blur just at the seam
    seam_region = panorama[:, w_left-overlap//2-5:w_left-overlap//2+5, :]
    if seam_region.size > 0:
        # Apply a vertical-only blur to smooth any horizontal seam
        kernel = np.ones((3, 1), np.float32) / 3
        for c in range(3):
            seam_blurred = cv2.filter2D(seam_region[:,:,c], -1, kernel)
            panorama[:, w_left-overlap//2-5:w_left-overlap//2+5, c] = seam_blurred
    
    return panorama

def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load images
    left_img = cv2.imread(LEFT_IMAGE_PATH)
    right_img = cv2.imread(RIGHT_IMAGE_PATH)
    
    if left_img is None or right_img is None:
        print("Error: Could not load input images")
        return
    
    print(f"Creating panorama from {LEFT_IMAGE_PATH} and {RIGHT_IMAGE_PATH}")
    
    # Try different configurations and preprocessing options
    panoramas = []
    
    # 1. Original order
    print("Trying with original image order...")
    pano1 = create_better_panorama(left_img, right_img)
    panoramas.append(pano1)
    
    # 2. Reversed order
    print("Trying with reversed image order...")
    pano2 = create_better_panorama(right_img, left_img)
    panoramas.append(pano2)
    
    # 3. Try with contrast enhanced images
    print("Trying with enhanced contrast...")
    left_enhanced = np.clip(left_img * 1.2, 0, 255).astype(np.uint8)
    right_enhanced = np.clip(right_img * 1.2, 0, 255).astype(np.uint8)
    pano3 = create_better_panorama(left_enhanced, right_enhanced)
    panoramas.append(pano3)
    
    # 4. Try with different overlap estimation
    print("Trying with fixed overlap...")
    h_left, w_left = left_img.shape[:2]
    fixed_overlap = w_left // 2  # Try with a larger fixed overlap
    
    # Create a custom panorama with fixed overlap
    h_left, w_left = left_img.shape[:2]
    h_right, w_right = right_img.shape[:2]
    width = w_left + w_right - fixed_overlap
    height = max(h_left, h_right)
    
    pano4 = np.zeros((height, width, 3), dtype=np.uint8)
    pano4[0:h_left, 0:w_left] = left_img
    
    # Create blend mask
    blend_mask = np.zeros((height, width))
    blend_mask[:, :w_left-fixed_overlap] = 1
    for i in range(fixed_overlap):
        # Cosine blend
        ratio = i / fixed_overlap
        weight = 0.5 * (np.cos(np.pi * ratio) + 1)
        blend_mask[:, w_left-fixed_overlap+i] = weight
    
    # Create right image canvas with fixed overlap
    right_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    right_canvas[0:h_right, (w_left-fixed_overlap):(w_left-fixed_overlap)+w_right] = right_img
    
    # Blend
    for c in range(3):
        pano4[:,:,c] = blend_mask * pano4[:,:,c] + (1 - blend_mask) * right_canvas[:,:,c]
    
    panoramas.append(pano4)
    
    # Select the best panorama based on quality metrics
    best_pano = None
    best_score = 0
    
    for i, pano in enumerate(panoramas):
        if pano is not None:
            # Evaluate panorama quality
            gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
            
            # Count non-black pixels
            non_black = np.sum(gray > 10)
            
            # Calculate variance in the middle region as a measure of duplication/artifacts
            mid_x = pano.shape[1] // 2
            mid_region = gray[:, mid_x-50:mid_x+50]
            if mid_region.size > 0:
                var_score = np.var(mid_region)
                # Higher variance is better (less flat/duplicate areas)
                # But extremely high variance might indicate seam problems
                if var_score > 100 and var_score < 5000:
                    total_score = non_black * var_score / 1000
                else:
                    total_score = non_black
            else:
                total_score = non_black
            
            print(f"Panorama {i+1} score: {total_score}")
            
            if total_score > best_score:
                best_score = total_score
                best_pano = pano
    
    if best_pano is not None:
        # Save and display the panorama
        cv2.imwrite(OUTPUT_PATH, best_pano)
        print(f"Saved panorama to {OUTPUT_PATH}")
        
        cv2.imshow("Panorama", best_pano)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to create panorama")

if __name__ == "__main__":
    main()