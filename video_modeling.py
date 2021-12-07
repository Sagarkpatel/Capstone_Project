# import packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms.functional as TF
import cv2 as cv
import time
import csv
import argparse

def load_data():
    # Load ground truth data and video filenames for all subjects
    data_dir = "/scratch/trn219/capstone/capstone_dataset_20210922"
    ground_truth_columns = ["X_Rotation","X_Translation","Y_Rotation",
                            "Y_Translation","Z_Rotation","Z_Translation"]

    # Dictionary to hold data for each subject
    data_dict = {}

    # Drill down into each subjects ground truth and video data files
    for sub in os.listdir(data_dir):
        ses_dir = data_dir + "/" + str(sub)
        if not sub.startswith('.'):  
            data_dict[sub] = {}
            for ses in os.listdir(ses_dir):
                if not ses.startswith('.'):
                    root_dir = ses_dir + "/" + ses + "/func"
                    for file in os.listdir(root_dir):
                        if not file.startswith('.'):
                            if file.endswith(".par"):
                                par_file_dir = root_dir + "/" + file
                                par_file = np.loadtxt(par_file_dir)
                                par_file = pd.DataFrame(par_file, 
                                                        columns = ground_truth_columns)
                                #Each entry is taken in 1.3 s intervals
                                par_file["time"] = par_file.index * 1.3
                                par_file = par_file.set_index("time")
                                data_dict[sub]["ground_truth"] = par_file
                            if file.endswith(".mp4"):
                                vid_file_dir = root_dir + "/" + file
                                data_dict[sub]["video"] = str(vid_file_dir)

    # split data into train, test, and val based on subject
    subjects = sorted(list(data_dict.keys()))
    
    # 19 train, 1 val, 1 test
    train_subs = subjects[:19]
    val_subs = [subjects[19]]
    test_subs = [subjects[20]]

    train_dict = {sub: data_dict[sub] for sub in train_subs}
    val_dict = {sub: data_dict[sub] for sub in val_subs}
    test_dict = {sub: data_dict[sub] for sub in test_subs}
    
    return train_dict, val_dict, test_dict

class VideoClipDataset(Dataset):
    def __init__(self, data_dict, optical_flow, back_sub):
        self.data_dict = data_dict
        data_list = []
        for subject, data in self.data_dict.items():
            labels = data['ground_truth']
            starts = list(labels.index)
            ends = [start + 1.3 for start in starts]
            video = data['video']

            for i in range(len(starts)):    
                label = torch.Tensor(labels.iloc[i])
                data_list.append({'subject': subject, 'video': video,
                                  'start': starts[i], 'end': ends[i], 
                                  'label': label})
        self.data = pd.DataFrame(data_list)
        self.optical_flow = optical_flow
        self.back_sub = back_sub

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get 1.3s slice of video frames
        clip = torchvision.io.read_video(self.data.iloc[idx]['video'], 
                                         start_pts=self.data.iloc[idx]['start'], 
                                         end_pts=self.data.iloc[idx]['end'],
                                         pts_unit="sec")
        vframes, aframes, fps = clip
        vframes = vframes.permute(0, 3, 1, 2)
        vframes = vframes / 255.0
        n_frames = vframes.shape[0]

        if self.optical_flow or self.back_sub:
            # perform background subtraction and dense optical flow on each frame
            cap = cv.VideoCapture(self.data.iloc[idx]['video'])
            cap.set(cv.CAP_PROP_POS_MSEC, self.data.iloc[idx]['start'] * 1000)
            fgbg = cv.createBackgroundSubtractorMOG2()
            ret, init_frame = cap.read()
            init = fgbg.apply(init_frame)
            init_gray = cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY)
            if self.back_sub:
                backsub_frames = [init - init]
            if self.optical_flow:
                optflow_frames = [cv.calcOpticalFlowFarneback(init_gray, init_gray,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0)]

            count = 0
            while count < n_frames-1:    
                ret, frame = cap.read()
            
                if self.optical_flow:
                    # Calculates dense optical flow by Farneback method
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    flow = cv.calcOpticalFlowFarneback(init_gray, gray,
                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    optflow_frames.append(flow)
                    init_gray = gray
            
                if self.back_sub:
                    # calculate difference between frame and previous frame
                    fbgb_frame = fgbg.apply(frame)
                    backsub = fbgb_frame - init
                    backsub_frames.append(backsub)
                    init = fbgb_frame

                count += 1
            
            cap.release()
            cv.destroyAllWindows()
        
            # concatenate background subtraction and optical flow outputs
            # to vframes along channel dimension
            if self.optical_flow:
                optflow = torch.Tensor(np.array(optflow_frames)).permute(0, 3, 1, 2)
                # normalize optical flow output so no absolute value is larger than 1
                optflow = optflow / torch.max(torch.abs(optflow))
                vframes = torch.cat((vframes, optflow), 1)
            if self.back_sub:
                backsub = torch.Tensor(np.array(backsub_frames)).unsqueeze(1)/255.0
                vframes = torch.cat((vframes, backsub), 1)
        
        vframes = vframes.permute(1, 0, 2, 3)

        # pad vframes_final with 0s if there are fewer than 40 frames
        if n_frames < 40:
            frames_to_add = 40 - n_frames
            pad = (0, 0, 0, 0, 0, frames_to_add)
            vframes = F.pad(vframes, pad, 'constant', 0.0)

        label = self.data.iloc[idx]['label']
        subject = self.data.iloc[idx]['subject']
        start = self.data.iloc[idx]['start']
        return {'label': label, 'subject': subject, 'start': start,
                'vframes': vframes}

def train_model(model, train_loader, criterion, optimizer, device,
                n_epochs=20, output_string=''):
    model.train()
    total_start = time.time()
    best_loss = np.inf
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch+1, n_epochs))
        print('-' * 10)
        
        running_loss = 0.0
        batch_start = time.time()
        for i, batch in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            vframes = batch['vframes'].to(device)
            labels = batch['label'].to(device)
            outputs = model(vframes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            batch_loss = loss.item()
            running_loss += batch_loss
            if i % 100 == 99:    # print every 100 mini-batches
                time_elapsed = time.time() - batch_start
                print('[%2d, %4d] loss: %.4f, time elapsed %.0f m %.0fs' %
                      (epoch + 1, i + 1, running_loss / (100*4), 
                       time_elapsed // 60, time_elapsed % 60))
                running_loss = 0.0
                batch_start = time.time()
            
            # save model
            if batch_loss < best_loss:
                if output_string == '':
                    model_path = 'models/best_model.pth'
                else:
                    model_path = f'models/best_model_{output_string}.pth'
                torch.save(model.state_dict(), model_path)
                best_loss = batch_loss
    print('Finished training, total time elapsed', time.time()-total_start)
    return model

def evaluate_model(model, val_loader, device):
    start_time = time.time()
    model.eval()
    results = {'subject': [], 'start': [], 
               'X_Rotation': [], 'X_Translation': [],
               'Y_Rotation': [], 'Y_Translation': [],
               'Z_Rotation': [], 'Z_Translation': []
              }
    #losses = {'subject': [], 'start': [], 'loss': []}
    #results = []

    for batch in val_loader:
        with torch.no_grad():
            vframes = batch['vframes'].to(device)
            labels = batch['label'].to(device)
            outputs = model(vframes)
            #loss = criterion(outputs, labels)
        
        preds = [[out.item() for out in output] for output in outputs]
        
        results['subject'].extend(batch['subject'])
        results['start'].extend(batch['start'].tolist())
        results['X_Rotation'].extend([pred[0] for pred in preds])
        results['X_Translation'].extend([pred[1] for pred in preds])
        results['Y_Rotation'].extend([pred[2] for pred in preds])
        results['Y_Translation'].extend([pred[3] for pred in preds])
        results['Z_Rotation'].extend([pred[4] for pred in preds])
        results['Z_Translation'].extend([pred[5] for pred in preds])
        
    print('Finished validation, time elapsed: ', time.time() - start_time)
    results_df = pd.DataFrame(results)
    return results_df

class WeightedMSELoss(nn.Module):
    def __init__(self, threshold_rot=0.05, alpha_rot=200., 
                 threshold_trans=1.0, alpha_trans=10.):
        super().__init__()
        self.threshold_rot = threshold_rot
        self.alpha_rot = alpha_rot
        self.threshold_trans = threshold_trans
        self.alpha_trans = alpha_trans

    def forward(self, y_pred, y_true):
        col_condition = torch.tile(torch.tensor([True, False]), 
                                   (y_true.size(dim=0), y_true.size(dim=1)//2))
        col_condition = col_condition.to(y_true.get_device())
        weights = torch.where(col_condition,
                              torch.where(torch.abs(y_true) >= self.threshold_rot, 
                                          self.alpha_rot*torch.abs(y_true),
                                          torch.ones_like(y_true)),
                              torch.where(torch.abs(y_true) >= self.threshold_trans,
                                          self.alpha_trans*torch.abs(y_true),
                                          torch.ones_like(y_true)))
        loss = torch.mean(torch.square(y_pred - y_true) * weights)
        return loss

def main(args):
    # load data into train, val, and test
    train_dict, val_dict, test_dict = load_data()

    # create train, val, and test datasets in torch
    train_dataset = VideoClipDataset(train_dict, args.optical_flow, args.back_sub)
    val_dataset = VideoClipDataset(val_dict, args.optical_flow, args.back_sub)
    test_dataset = VideoClipDataset(test_dict, args.optical_flow, args.back_sub)

    # create train, val, and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              num_workers=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=20)

    # initialize model, criterion, optimizer
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = torchvision.models.video.r3d_18(pretrained=False)
    
    # configure number of input channels depending on whether including
    # background subtraction and/or optical flow
    in_channels = model.stem[0].in_channels
    if args.optical_flow:
        in_channels += 2
    if args.back_sub:
        in_channels += 1
    model.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), 
                              stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=6, bias=True)
    model = model.to(device)
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss == 'WeightedMSE':
        criterion = WeightedMSELoss()
    elif args.loss == 'MAE':
        criterion = nn.L1Loss()
    else:
        raise ValueError('Loss must be MSE, WeightedMSE, or MAE')
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = criterion.to(device)

    # print config information
    print(f'Batch size: {args.batch_size}, loss: {args.loss}, LR: {args.lr}, momentum: {args.momentum}')
    print(f'Optical flow: {args.optical_flow}, Background Subtraction: {args.back_sub}')
    print(f'Output string: {args.output_string}')
    print('Optimizer: ', optimizer)

    # train the model on train set
    trained_model = train_model(model, train_loader, criterion, optimizer,
                                device, n_epochs=args.n_epochs, 
                                output_string=args.output_string)

    # save trained model
    model_path = f'models/trained_model_{args.output_string}.pth'
    torch.save(trained_model.state_dict(), model_path)

    # evaluate model on val set
    results_df = evaluate_model(trained_model, val_loader, device)

    # save evaluation results to csv
    if args.output_string == '':
        results_path = 'results.csv'
    else:
        results_path = f'results_{args.output_string}.csv'
    results_df.to_csv(results_path)

if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser(description='train model on eye tracker video')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
            help='Device to train model. Options are cpu and cuda.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--optical_flow', action='store_true', 
            help='Flag to include dense optical flow output in model input')
    parser.add_argument('--back_sub', action='store_true',
            help='Flag to include background subtraction output in model input')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--n_epochs', default=20, type=int, 
            help='Number of training epochs')
    parser.add_argument('--output_string',
            help='String to add to end of output file names')
    parser.add_argument('--loss', default='MSE', 
            choices=['MSE', 'WeightedMSE', 'MAE'], help='Loss function')
    args = parser.parse_args()
    
    main(args)
