import numpy as np
import cv2

def computeH(x1, x2):

    # Number of point correspondences
    n = x1.shape[0]
    
    # Create A matrix for homography calculation
    A = np.zeros((2*n, 9))
    
    for i in range(n):
        x, y = x2[i]
        u, v = x1[i]
        
        # Fill A matrix according to DLT approach
        # x*h7 - h1 = 0, y*h7 - h2 = 0, etc.
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]
    
    # Solve the linear system Ah = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    
    # The solution is the last column of V (last row of Vt)
    h = Vt[-1]
    
    # Reshape to 3x3 matrix
    H2to1 = h.reshape(3, 3)
    
    return H2to1

def computeH_norm(_x1, _x2):

    x1 = np.array(_x1)
    x2 = np.array(_x2)

    # Compute the centroid (mean) of the points
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1norm = x1 - mean1
    x2norm = x2 - mean2

    # Compute the scale to normalize the points
    # We want the largest distance from origin to be sqrt(2)
    dist1 = np.sqrt(np.sum(x1norm**2, axis=1))
    dist2 = np.sqrt(np.sum(x2norm**2, axis=1))
    
    # Scale factor to make max distance = sqrt(2)
    norm1 = np.max(dist1) / np.sqrt(2)
    norm2 = np.max(dist2) / np.sqrt(2)

    # Normalize the points
    x1norm = x1norm / norm1
    x2norm = x2norm / norm2

    # Similarity transform matrices
    # T1 transforms x1 -> x1norm
    T1 = np.array([
        [1/norm1, 0, -mean1[0]/norm1],
        [0, 1/norm1, -mean1[1]/norm1],
        [0, 0, 1]
    ])
    
    # T2 transforms x2 -> x2norm
    T2 = np.array([
        [1/norm2, 0, -mean2[0]/norm2],
        [0, 1/norm2, -mean2[1]/norm2],
        [0, 0, 1]
    ])
    
    # T1_inv transforms x1norm -> x1
    T1_inv = np.array([
        [norm1, 0, mean1[0]],
        [0, norm1, mean1[1]],
        [0, 0, 1]
    ])

    # Compute homography between normalized points
    H_norm = computeH(x1norm, x2norm)

    # Denormalize the homography
    H2to1 = T1_inv @ H_norm @ T2

    return H2to1

def computeH_ransac(_x1, _x2, nSamples=5000, threshold=3.0):

    x1 = np.array(_x1)
    x2 = np.array(_x2)
    
    # Validate input
    nPoints = len(x1)
    if nPoints != len(x2) or nPoints < 4:
        print(f"Error: Need at least 4 matching points, got {nPoints}")
        return None, None
    
    # Initialize variables to track best result
    bestH2to1 = None
    bestInlierCount = 0
    inliers = np.zeros(nPoints, dtype=int)
    
    # Convert points to homogeneous coordinates
    x1_homog = np.column_stack((x1, np.ones(nPoints)))
    x2_homog = np.column_stack((x2, np.ones(nPoints)))
    
    # Precompute weights based on spatial distribution
    # This prioritizes well-distributed point samples
    distances = np.zeros((nPoints, nPoints))
    for i in range(nPoints):
        for j in range(i+1, nPoints):
            d = np.sqrt(np.sum((x1[i] - x1[j])**2))
            distances[i, j] = distances[j, i] = d
    
    weights = np.sum(distances, axis=1)
    weights = weights / np.sum(weights)
    
    # RANSAC iterations
    for i in range(nSamples):
        try:
            # Choose 4 points with preference to well-distributed points
            # Using weighted random sampling instead of uniform sampling
            indexes = np.random.choice(np.arange(nPoints), size=4, replace=False, p=weights)
            
            # Extract the 4 points
            sample_x1 = x1[indexes]
            sample_x2 = x2[indexes]
            
            # Compute normalized homography
            H = computeH_norm(sample_x1, sample_x2)
            
            # Skip degenerate homographies
            if np.abs(np.linalg.det(H)) < 1e-8:
                continue
                
            # Calculate projection error for all points
            x2_homog_T = x2_homog.T
            x1_predicted_homog = H @ x2_homog_T
            
            # Handle division by zero
            z = x1_predicted_homog[2, :]
            valid_z = np.abs(z) > 1e-10
            
            if not np.all(valid_z):
                continue
                
            # Normalize homogeneous coordinates
            x1_predicted_homog = x1_predicted_homog[:, valid_z]
            x1_predicted_homog = x1_predicted_homog / x1_predicted_homog[2, :]
            
            # Compute squared Euclidean distance
            x1_valid = x1_homog[valid_z]
            distances = np.sum((x1_predicted_homog[:2, :].T - x1_valid[:, :2])**2, axis=1)
            
            # Identify inliers using the threshold
            current_inliers = np.zeros(nPoints, dtype=int)
            current_inliers[valid_z] = (distances < threshold).astype(int)
            inlierCount = np.sum(current_inliers)
            
            # Update best model if we found more inliers
            if inlierCount > bestInlierCount:
                bestInlierCount = inlierCount
                bestH2to1 = H
                inliers = current_inliers
        except Exception as e:
            # Skip iterations with numerical errors
            continue
    
    # If we found a good model, refine it using all inliers
    if bestInlierCount > 8:  # Require at least 8 inliers for refinement
        # Multiple refinement iterations using inliers
        for _ in range(2):  # Two refinement passes
            try:
                inlier_indices = np.where(inliers == 1)[0]
                if len(inlier_indices) >= 4:
                    refined_x1 = x1[inlier_indices]
                    refined_x2 = x2[inlier_indices]
                    
                    # Compute refined homography
                    refined_H = computeH_norm(refined_x1, refined_x2)
                    
                    # Verify this refinement is valid
                    if np.abs(np.linalg.det(refined_H)) > 1e-8:
                        bestH2to1 = refined_H
                        
                        # Recompute inliers with the refined homography
                        x1_predicted_homog = bestH2to1 @ x2_homog.T
                        
                        # Handle division by zero
                        z = x1_predicted_homog[2, :]
                        valid_z = np.abs(z) > 1e-10
                        
                        if np.all(valid_z):
                            x1_predicted_homog = x1_predicted_homog / x1_predicted_homog[2, :]
                            distances = np.sum((x1_predicted_homog[:2, :].T - x1_homog[:, :2])**2, axis=1)
                            inliers = (distances < threshold).astype(int)
            except Exception as e:
                # If refinement fails, keep the current best H
                pass
    
    # Check if we have a valid result
    if bestH2to1 is None or bestInlierCount < 8:
        print(f"Warning: Poor homography fit with only {bestInlierCount} inliers")
    else:
        print(f"Good homography found with {bestInlierCount} inliers out of {nPoints} points")
    
    return bestH2to1, inliers

def compositeH(H2to1, template, img, alreadyInverted=False):
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # For warping the template to the image, we need to invert if not already inverted
    if alreadyInverted:
        H1to2 = H2to1
    else:
        H1to2 = np.linalg.inv(H2to1)
    
    # Create a mask of same size as template (ones where template has content)
    mask = np.ones_like(template)
    
    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H1to2, (width, height), 
                                     flags=cv2.INTER_LINEAR)
    
    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H1to2, (width, height), 
                                         flags=cv2.INTER_LINEAR)
    
    # Invert the mask to get the background region
    inv_warped_mask = 1.0 - warped_mask
    
    # Create composite by keeping background where mask is 0, and template where mask is 1
    composite_img = img * inv_warped_mask + warped_template
    
    return composite_img
