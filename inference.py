import os
import time
import scipy.io
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import model.utils_pred as utils_pred
from utils import *
from model.Prediction import *

from pathlib import Path
from collections import OrderedDict
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim


# torch setup

def torch_setup(config):
    torch.manual_seed(2020)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    torch.backends.cudnn.enabled = True


# Loading dataset

def get_dataset_batch(config):
    dataset = utils_pred.DataLoader(config.data_dir, 
                         transforms.Compose([
                            transforms.ToTensor()
                            ]),
                         resize_height = config.h, 
                         resize_width = config.w, 
                         time_step = config.t_length - 1,
                         filter_vid_name = config.desired_folder)

    config.dataset_size = len(dataset)

    dataset_batch = data.DataLoader(dataset, 
                              batch_size = config.batch_size, 
                              shuffle = False, 
                              num_workers = config.num_workers_test,
                              drop_last = False)

    return dataset_batch


# load model

def get_model(config):
    model = torch.load(config.model_file)
    model.cuda()
    m_items = torch.load(config.model_keys_file)

    return model, m_items


# load ground truth

def get_gt(config):
    mat = scipy.io.loadmat(config.gt_file)
    gt = mat["gt"][0]
    if config.anomalous_data:
        return gt
    return []


# videos dict

def get_videos(config):
    videos = OrderedDict()
    
    videos_list = sorted(glob.glob(os.path.join(config.data_dir, '*')))
    filtered_videos_list = []
    for video in videos_list:
        video_name = video.split('/')[-1]
        if config.desired_folder and (video_name != config.desired_folder):
            continue
        filtered_videos_list.append(video)
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
    return videos, filtered_videos_list


# init data structures

def init_datastructures(config, videos_dict, videos_list, gt, ds):
    psnr_list, feature_dist_list, labels_list = {}, {}, []
    for video in videos_list:
        video_name = video.split('/')[-1]
        if video_name not in videos_dict:
            continue
        video_length = videos_dict[video_name]["length"]
        video_idx = int(video_name) - 1 # assuming video name is like "001", "002" etc
        if video_idx >= len(gt):
            anomaly_start = video_length
            anomaly_end = video_length
        else:
            anomaly_start = gt[video_idx][0].item()
            anomaly_end = gt[video_idx][1].item()
        
        print(f"Video id: {video_idx}, anomaly start: {anomaly_start}, anomaly end: {anomaly_end}")
        
        y_true = [0] * anomaly_start + [1] * (anomaly_end - anomaly_start) + [0] * (video_length - anomaly_end)
        labels_list = np.append(labels_list, y_true[4:])

        psnr_list[video_name] = []
        feature_dist_list[video_name] = []
    
    ds["psnr_list"] = psnr_list
    ds["feature_dist_list"] = feature_dist_list
    ds["labels_list"] = labels_list


# get difference image

def get_diff_img(og, recon, use_ssim=True):
    if use_ssim:
        _, diff = ssim(og, recon, full=True)
    else:
        diff = np.abs(og-recon)
    
    return (diff*255).astype(np.uint8)


# localize anomaly

def localize_anomaly(config, diff_img):
    thresh = cv2.threshold(diff_img, config.loc_thresh , 255, cv2.THRESH_BINARY_INV)[1]
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_list = []                   
    loc_coords = 0
    c = max(cnts, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return x, y, w, h


# evaluate

def evaluate(config, model, m_items, dataset_batch, 
            videos_dict, videos_list, ds, 
            save_diff=False, tqdm_pbar=None):
    label_length = 0
    video_num = 0
    loss_func_mse = nn.MSELoss(reduction="none")

    label_length += videos_dict[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

    model.eval()
    t1 = time.time()

    fourcc_type = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_type)
    
    vw = None
    out_path = Path(config.vid_dir) / "vis" / f"tmp{config.desired_folder}.mp4"

    if save_diff:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out_path), fourcc, 12, (256*3, 256), 0)
    
    for k, (batched_imgs) in enumerate(dataset_batch):
        if k == label_length - 4*(video_num+1):
            video_num += 1
            label_length += videos_dict[videos_list[video_num].split('/')[-1]]['length']

        imgs = Variable(batched_imgs).cuda()
        
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], 
                                                                                                                                        m_items_test, 
                                                                                                                                        False)
        output_img = outputs.cpu().data.numpy()
        output_img = output_img.squeeze(0)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        output_img = np.transpose(output_img, (1, 2, 0))
        
        # # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])

        if  point_sc < config.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)
        psnr_dict = {}

        ds["psnr_list"][videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
        ds["feature_dist_list"][videos_list[video_num].split('/')[-1]].append(mse_feas)
        
        if save_diff:
            og = ((batched_imgs[0,3*4:]+1)*127.5).numpy().astype(np.uint8)
            og = np.transpose(og, (1, 2, 0))
            og_gray = cv2.cvtColor(og, cv2.COLOR_RGB2GRAY)
            recon = ((output_img+1)*127.5).astype(np.uint8)
            recon_gray = cv2.cvtColor(recon, cv2.COLOR_RGB2GRAY)
            diff = get_diff_img(og_gray, recon_gray)
            try:
                x, y, w, h = localize_anomaly(config, diff)
                if (w*h)>=500:
                    cv2.rectangle(recon_gray,(x,y),(x+w,y+h),(255, 255, 255),2)
            except Exception as err:
                print(err)
            side_by_side = np.concatenate((og_gray, recon_gray, diff), axis=1)
            vw.write(side_by_side)


        if tqdm_pbar:
            tqdm_pbar.update()

    
    to_return = None
    if save_diff:
        vw.release()
        print("Saved video ...")
        os.system('ffmpeg -i {} -vcodec libx264 {} -y'.format(str(out_path), str(out_path).replace("tmp","")))
        to_return = str(out_path).replace("tmp","")
    
    t2 = time.time()
    print(f"Elapsed: {(t2 - t1)} secs")

    return to_return


# calc anomaly scores

def get_anomaly_scores(config, videos_list, ds, save=True, plot=True):

    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_score_total_list += score_sum(
                                        anomaly_score_list(ds["psnr_list"][video_name]), 
                                        anomaly_score_list_inv(ds["feature_dist_list"][video_name]),
                                        config.alpha
                                    )

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    if save:
        df = pd.DataFrame(data=anomaly_score_total_list)
        df.to_csv('file1.csv')
    
    if plot:
        plt.plot(anomaly_score_total_list)
        plt.savefig('graph.png')
        plt.show();
    
    return anomaly_score_total_list


# calc AUC

def calc_AUC(anomaly_score_total_list, ds):
    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-ds["labels_list"], 0))
    return accuracy
