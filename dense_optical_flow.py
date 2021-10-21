import cv2 as cv
import numpy as np
import pandas as pd
import sys

def DenseOpticalFlow(print_location=False, save_output=False, path_to_video="test_video.mp4"):
	""" Performs dense optical flow on a video and displays the video and the outputs of dense optical flow

	Parameters
	----------
	print_location : bool
		flag if you want to print the cumulative sum of changes of X and Y to console (prints every 1.3 seconds)
	save_output : bool
		flag if you want to save the cumulative sum of changes in X and Y as a csv file (saved in same folder as output_vectors.csv)
	path_to_video : str
		specified the path to the video file to process. Defaults to 'test_video.mp4'
	"""

	# The video feed is read in as a VideoCapture object
	cap = cv.VideoCapture(path_to_video)

	# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
	ret, first_frame = cap.read()

	# object used to create foreground mask
	backSub = cv.createBackgroundSubtractorMOG2()

	# create foreground mask using the first (assumed to be correct) image
	fgMask = backSub.apply(first_frame)

	# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
	prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

	# Creates an image filled with zero intensities with the same dimensions as the frame
	mask = np.zeros_like(first_frame)

	# Sets image saturation to maximum
	mask[..., 1] = 255

	position_vectors = {'X': [], 'Y': []}
	count = 0

	while(cap.isOpened()):
		
		# ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
		ret, frame = cap.read()
		
		# Opens a new window and displays the input frame
		# cv.imshow('Clean Input', frame)
		
		# Converts each frame to grayscale - we previously only converted the first frame to grayscale
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		
		# Calculates dense optical flow by Farneback method
		flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
										None,
										0.5, 3, 15, 3, 5, 1.2, 0)
		
		# Computes the magnitude and angle of the 2D vectors
		magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
		position_vectors['X'].append(np.mean(flow[..., 0]))
		position_vectors['Y'].append(np.mean(flow[..., 1]))
		

		# adds frame with cumulative movement arrow
		(h, w) = frame.shape[:2]
		start_X, start_Y = (w//2, h//2)
		start_point = (start_X, start_Y)
		end_point = (start_X + int(np.sum(position_vectors['X'])*10), start_Y + int(np.sum(position_vectors['Y'])*10))
		color = (0, 255, 0)
		thickness = 4

		image = cv.arrowedLine(frame, start_point, end_point, color, thickness)
		cv.imshow('Input', image) 


		count += 1
		# print current cumulative sum in X and Y to console (every 1.3 second)
		if print_location:
			if count % 39 == 0:
				print("X Position: {:.4f}, Y Position: {:.4f}".format(np.sum(position_vectors['X']), np.sum(position_vectors['X'])))
		
		# save flow output as a csv file
		if save_output:
			if count % 1000 == 0:
				# print('saving!')
				df = pd.DataFrame(position_vectors)
				df.to_csv("output_vectors.csv")
		
		# Sets image hue according to the optical flow direction
		mask[..., 0] = angle * 180 / np.pi / 2
		
		# Sets image value according to the optical flow magnitude (normalized)
		mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
		
		# Converts HSV to RGB (BGR) color representation
		rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
		
		# Opens a new window and displays the output frame
		cv.imshow("dense optical flow", rgb)
		
		# Updates previous frame
		prev_gray = gray
		
		# Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	# The following frees up resources and closes all windows
	cap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	DenseOpticalFlow(*sys.argv[1:])
