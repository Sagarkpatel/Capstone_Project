import numpy as np
import cv2 as cv
import pandas as pd
import os
import pickle


path = "/Users/sagar/Desktop/Capstone/capstone_dataset_20210922/sub-NC232/ses-20190430/func/sub-NC232_ses-20190430_task-TASK_acq-normal_run-01_bold_video_cropped.mp4"
# path = "/Users/sagar/Desktop/Capstone/capstone_dataset_20210922/sub-NC235/ses-20190606/func/sub-NC235_ses-20190606_task-TASK_acq-normal_run-01_bold_video_cropped.mp4"
# path = "/Users/sagar/Desktop/Capstone/capstone_dataset_20210922/sub-NC248/ses-20190811/func/sub-NC248_ses-20190811_task-TASK_acq-normal_run-01_bold_video_cropped.mp4"
cap = cv.VideoCapture(path)

#pick a background subtractor method (MOG2 or GMG)
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()

#initialize first frame
ret, init_frame = cap.read()
init = fgbg.apply(init_frame)

#Store background subtraction differences in a dictionary
# df = pd.DataFrame(columns=list(range(141050)))
df_total = pd.DataFrame(columns=["Difference"])
df_dense = pd.DataFrame(columns=list(range(141050)),dtype=float)
#Get fps and frame to calculate time
fps = int(cap.get(cv.CAP_PROP_FPS))
frame_count = 0
time = 0 

filename_model = "final_back_model.sav"
model =  pickle.load(open(filename_model, 'rb'))

x_lst = []
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    
    #Apply background subtraction mask to each frame
    next = fgbg.apply(frame)

    #Calculate differences in each frame
    diff = next-init

    #Store values (each pixel or aggregative)
    time = float(frame_count)/fps
    x = diff.sum()
    x_lst.append(x)
    if len(x_lst) > int(30*1.3):
        x = sum(x_lst[-int(30*1.3):]) / len(x_lst[-int(30*1.3):])
        y = model.predict(x.reshape(-1, 1))
        if y ==1:
            text = "Movement Detected"
            color = (255,255,255)
            coordinates = (0,frame.shape[0]-10)
            font = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            for i in range(len(diff)):
                for j in range(len(diff[i])):
                    if diff[i,j] > 0:
                        frame[i,j,1] = 255
            frame = cv.putText(frame, text, coordinates, font, fontScale, color, thickness, cv.LINE_AA)

    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    #Iterate
    init = next
    frame_count= frame_count + 1

