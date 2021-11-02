import numpy as np
import cv2 as cv
import pandas as pd
import os


def getFilePaths():
    #Use whatever directory data file was unzipped into
    par_directory = "/Users/sagar/Desktop/Capstone/capstone_dataset_20210922"

    #Dictionary to hold file paths for video data for later processing (.mp4 files)
    video_dict = {}

    #Drill down into each subjects data files
    for sub_file in os.listdir(par_directory):
        subject = sub_file
        ses_directory = par_directory+"/" + str(sub_file)
        if not sub_file.startswith('.'):
            for ses_file in os.listdir(ses_directory):
                if not ses_file.startswith('.'):
                    root_dir = ses_directory+ "/" + ses_file+ "/func"
                    for file in os.listdir(root_dir):
                        if not file.startswith('.'):
                            if file.endswith(".mp4"):
                                par_file_dir = root_dir + "/"+ file
                                video_dict[subject] = par_file_dir
    return video_dict

def getTargetTime(time):
        target_times = list(np.arange(0,1.3*324,1.3))
        return target_times[min(range(len(target_times)), key = lambda i: abs(target_times[i]-time))]


video_dict = getFilePaths()
for sub in video_dict:

    path = video_dict[sub]
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
        time_proc = getTargetTime(time)
        # df.loc[time] = diff.flatten()
        # df_total.loc[time_proc] = np.abs(diff).sum()
        df_dense.loc[time_proc] = list(diff.flatten())
        #Display t-1 frame, t frame, and difference between two frames
        #No need to display video

        # cv.imshow('Comparison', np.hstack((frame, init, diff)))
        # cv.imshow("Frame", frame)s
        # cv.imshow("Difference", diff)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        #Iterate
        init = next
        frame_count= frame_count + 1

    #Save data to intermediate file
    # print(f"Aggregating summed {sub}")
    # df_total_diff = df_total.groupby(df_total.index).mean()
    # print(f"Saving summed {sub}")
    # df_total_diff.to_csv(f"data/total/{sub}.csv")


    print(f"Aggregating dense {sub}")
    df_dense_avg = df_dense.groupby(df_dense.index).mean()
    print(f"Saving dense {sub}")
    df_dense_avg.to_csv(f"data/dense/{sub}.csv")


    #Done with subject
    cap.release()
    cv.destroyAllWindows()
    print(f"Finished with {sub}")