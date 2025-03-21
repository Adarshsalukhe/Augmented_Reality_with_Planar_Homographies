import numpy as np
import cv2

def saveVid(path, frames, fourccString="MJPG", fps=30):
    frameShape = (frames[0].shape[1], frames[0].shape[0])

    fourcc = cv2.VideoWriter_fourcc(*fourccString)
    writer = cv2.VideoWriter(path,
                             fourcc=fourcc, fps=fps,
                             frameSize=frameShape,
                             params=None)

    for fr in frames:
        writer.write(fr)

    writer.release()
    return
	

def loadVid(path):
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	cap = cv2.VideoCapture(path)

	# Check if camera opened successfully
	if (cap.isOpened()== False):
		print("Error opening video stream or file")

	i = 0
	# Read until video is completed
	while(cap.isOpened()):
		# Capture frame-by-frame
		i += 1
		ret, frame = cap.read()
		if ret == True:

			#Store the resulting frame
			if i == 1:
				frames = frame[np.newaxis, ...]
			else:
				frame = frame[np.newaxis, ...]
				frames = np.vstack([frames, frame])
				frames = np.squeeze(frames)

		else:
			break

	# When everything done, release the video capture object
	cap.release()

	return frames

vid = loadVid('data/ar_source.mov');
saveVid('results/ar_source.avi', vid)
print(vid.shape)
