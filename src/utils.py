import os
from tqdm import tqdm
from config import IMAGE_EXT
from typing import List
import numpy as np
import cv2
import pyautogui 

def get_screenshot_as_cvimg():
    img = pyautogui.screenshot()
    img = cv2.cvtColor(
        np.array(img),
        cv2.COLOR_RGB2BGR
    )
    return img

def get_video_frames(fp) -> List:
    frames = []
    frame_num = 0
    vidcap = cv2.VideoCapture(fp)
    s = True
    while s:
        s, frame = vidcap.read()
        frame_num += 1
        frames.append((frame_num, frame))
    return frames

def extract_video_frames_to(video_fp, store_dir):
    frames = get_video_frames(video_fp)
    video_name = video_fp.split("/")[-1].split(".")[0]
    save_path = lambda f_id: os.path.join(
        store_dir,
        f"{video_name}-{f_id}{IMAGE_EXT}"
    )
    for (frame_num, f) in tqdm(frames, desc=f"Storing frames to {store_dir}"):
        cv2.imwrite(save_path(frame_num), f)
