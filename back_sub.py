import numpy as np
import cv2 as cv
import pandas as pd

path = "/Users/sagar/Desktop/Capstone/capstone_dataset_20210922/sub-NC232/ses-20190430/func/sub-NC232_ses-20190430_task-TASK_acq-normal_run-01_bold_video_cropped.mp4"
cap = cv.VideoCapture(path)

#pick a background subtractor method (MOG2 or GMG)
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()

#initialize first frame
ret, init_frame = cap.read()
init = fgbg.apply(init_frame)

#Store background subtraction differences in a dictionary
diff_dict = {}
target_times = list(np.arange(0,1.3*324,1.3))

#Get fps and frame to calculate time
fps = int(cap.get(cv.CAP_PROP_FPS))
frame_count = 0
time = 0 

while True:
    print(time)
    ret, frame = cap.read()
    if frame is None:
        break
    
    #Apply background subtraction mask to each frame
    next = fgbg.apply(frame)

    #Calculate differences in each frame
    diff = next-init
    sum_diff = diff.sum()

    #Store values
    time = float(frame_count)/fps
    diff_dict[time] = sum_diff

    #Display t-1 frame, t frame, and difference between two frames
    # cv.imshow('Comparison', np.hstack((init, next, diff)))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    #Iterate
    init = next
    frame_count= frame_count + 1

#Save data to intermediate file
diff_pd = pd.DataFrame.from_dict(diff_dict,orient = "index")
diff_pd.to_csv('back_sub.csv')

cap.release()
cv.destroyAllWindows()