import numpy as np
import cv2
import os

# Import necessary functions
import concurrent.futures
from loadSaveVid import loadVid, saveVid
from matchPics import matchPics, convert_features
from planarH import computeH_ransac, compositeH

# File paths
arSourcePath = "data/ar_source.mov"
bookMovieSourcePath = "data/book.mov"
resultsdir = "results"

def main():
    os.makedirs(resultsdir, exist_ok=True)
    
    # Create separate folders for cropped and composite frames
    cropped_dir = os.path.join(resultsdir, "cropped_frames")
    composite_dir = os.path.join(resultsdir, "composite_frames")
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(composite_dir, exist_ok=True)
    
    # Load videos
    print("Loading videos...")
    arFrames = loadVid(arSourcePath)
    bookFrames = loadVid(bookMovieSourcePath)
    if len(arFrames) == 0 or len(bookFrames) == 0:
        print("Error: Could not load videos")
        return
    
    # Process all frames available in both videos
    numFrames = min(len(arFrames), len(bookFrames))
    print(f"Processing {numFrames} frames...")
    
    # Load book cover for cropping
    bookCover = cv2.imread('data/cv_cover.jpg')
    if bookCover is None:
        raise FileNotFoundError("Could not load 'data/cv_cover.jpg'")
    
    # Process frames in parallel
    print("Starting parallel processing of frames...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create parameter tuples for each frame
        tasks = [(i, arFrames[i], bookFrames[i], bookCover, cropped_dir, composite_dir) 
                 for i in range(numFrames)]
        
        # Map tasks to the process_frame_complete function
        for i, _ in enumerate(executor.map(process_frame_complete, tasks)):
            if i % 10 == 0:
                print(f"Processed frame {i}/{numFrames-1} ({(i/numFrames)*100:.1f}%)")
    
    # Collect all composite frames for video creation
    print("Collecting processed frames...")
    composite_frames = []
    for i in range(numFrames):
        frame_path = os.path.join(composite_dir, f"composite_{i:04d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is not None:
            composite_frames.append(frame)
    
    # Save composite video
    print("Saving final video...")
    videoOutPath = os.path.join(resultsdir, "ar.avi")  # Changed to MP4 format
    saveVid(videoOutPath, composite_frames, fourccString="mp4v", fps=30)
    print(f"Video saved to {videoOutPath}")
    print(f"All frames processed and saved to {cropped_dir} and {composite_dir}")
    print(f"Total frames processed: {len(composite_frames)}")

def process_frame_complete(args):
    i, arFrame, bookFrame, bookCover, cropped_dir, composite_dir = args
    
    try:
        # Process AR frame - apply highlighting and remove top/bottom black borders
        cropped_frame = apply_dynamic_highlighting(arFrame)
        cropped_path = os.path.join(cropped_dir, f"cropped_{i:04d}.jpg")
        cv2.imwrite(cropped_path, cropped_frame)
        
        # Process book frame - detect and remove black borders if present
        processed_book_frame = process_book_frame(bookFrame)
        
        # Crop cropped frame to match book cover dimensions for overlay
        arFrameCropped = cropFrameToCover(cropped_frame, bookCover)
        
        # Create composite with the processed and cropped cropped frame
        composite = overlayFrame(bookCover, processed_book_frame, arFrameCropped)
        
        # Save composite frame
        composite_path = os.path.join(composite_dir, f"composite_{i:04d}.jpg")
        cv2.imwrite(composite_path, composite)
        
    except Exception as e:
        print(f"Error processing frame {i}: {str(e)}")
    
    return i

def apply_dynamic_highlighting(frame):
    # Detect black borders (only top and bottom)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find non-black regions (threshold may need adjustment based on your video)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Detect horizontal borders only
    row_sums = np.sum(thresh, axis=1)
    non_black_rows = np.where(row_sums > 0)[0]
    
    if len(non_black_rows) > 0:
        y_start, y_end = non_black_rows[0], non_black_rows[-1]
        # Crop only top and bottom black borders
        frame = frame[y_start:y_end+1, :]
    
    # Apply the yellow highlighting effect on the sides as in the original
    height, width = frame.shape[:2]
    center_strip_width = int(width * 0.40)
    center_start = (width - center_strip_width) // 2 - int(width * 0.025)
    center_end = center_start + center_strip_width
    result = frame.copy()
    golden_yellow = np.array([32, 165, 240], dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[:, :center_start] = True
    mask[:, center_end:] = True
    blended = (result.astype(np.float32) * 0.05 +
               np.full_like(result, golden_yellow, dtype=np.float32) * 0.95)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    result[mask] = blended[mask]
    return result

def process_book_frame(frame):
    # Detect black borders
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find non-black regions (threshold may need adjustment based on your video)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Detect horizontal borders
    row_sums = np.sum(thresh, axis=1)
    non_black_rows = np.where(row_sums > 0)[0]
    
    # Detect vertical borders
    col_sums = np.sum(thresh, axis=0)
    non_black_cols = np.where(col_sums > 0)[0]
    
    if len(non_black_rows) > 0 and len(non_black_cols) > 0:
        y_start, y_end = non_black_rows[0], non_black_rows[-1]
        x_start, x_end = non_black_cols[0], non_black_cols[-1]
        content = frame[y_start:y_end+1, x_start:x_end+1]
        return content
    
    # If no black borders detected, return the original frame
    return frame

def overlayFrame(bookCover, bookMovFrame, arFrameCropped):
    # Use matchPics to find feature matches
    matches, locs1, locs2 = matchPics(bookCover, bookMovFrame, ratio=0.65)
    
    if matches.size == 0 or matches.shape[0] < 4:
        return bookMovFrame
    
    # Convert feature locations to the format expected by computeH_ransac
    x1, x2 = convert_features(matches, locs1, locs2)
    
    # Compute homography with RANSAC
    H, _ = computeH_ransac(x2, x1, nSamples=5000, threshold=3.0)
    if H is None:
        return bookMovFrame
    
    # Normalize homography matrix
    H /= H[2, 2]
    
    # Create composite image
    compositeImg = compositeH(H, arFrameCropped, bookMovFrame, alreadyInverted=True)
    return compositeImg

def cropFrameToCover(frame, cover):
    hFrame, wFrame = frame.shape[:2]
    hCover, wCover = cover.shape[:2]
    aspectFrame = wFrame / hFrame
    aspectCover = wCover / hCover
    
    if aspectFrame > aspectCover:
        # Frame is wider - crop sides
        newWidth = int(hFrame * aspectCover)
        left = (wFrame - newWidth) // 2
        frameCropped = frame[:, left:left + newWidth]
    else:
        # Frame is taller - crop top/bottom
        newHeight = int(wFrame / aspectCover)
        top = (hFrame - newHeight) // 2
        frameCropped = frame[top:top + newHeight, :]
    
    # Resize to match cover dimensions
    frameCropped = cv2.resize(frameCropped, (wCover, hCover), interpolation=cv2.INTER_CUBIC)
    return frameCropped

if __name__ == "__main__":
    main()