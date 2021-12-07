# Head Movement Measurement During Structural MRI

Run the dense optical flow by navigating to the file in terminal. It takes 3 optional arguments: 
  print_location : bool
		flag if you want to print the cumulative sum of changes of X and Y to console (prints every 1.3 seconds)
	save_output : bool
		flag if you want to save the cumulative sum of changes in X and Y as a csv file (saved in same folder as output_vectors.csv)
	path_to_video : str
		specified the path to the video file to process. Defaults to 'test_video.mp4'

run by typing in the command "python dense_optical_flow.py" and add additional arguments as needed. e.g. "python dense_optical_flow.py True False '/path/to/video_file.mp4'"

An example of how to train the deep learning model is in run_video_model.sbatch
