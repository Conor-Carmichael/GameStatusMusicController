from time import time
from typing import List
import streamlit as st
from PIL import Image
import cv2
import os
from config import ACTIVE, INACTIVE, VIDEOS_DIR, DATA_DIR, GAME_STATUSES, CLASSES, IMAGE_EXT

video_paths = [os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR) if f.endswith(".mov")  or f.endswith(".mp4") ]

def save_frame(image_obj , game_state:str, label:str, video_from:str, frame_num:int) -> None :
    save_path = os.path.join(
        DATA_DIR, 
        label, 
        f"{video_from}-{frame_num}-{game_state}{IMAGE_EXT}"
    )
    cv2.imwrite(save_path, image_obj)
    print("SAVED to ", save_path)

def get_video_frames(fp) -> List:
    frames = []
    frame_num = 0
    vidcap = cv2.VideoCapture(fp)
    s = True
    while s:
        s, frame = vidcap.read()
        # yield frame_num, frame
        frame_num += 1
        frames.append((frame_num, frame))
    print(len(frames))
    return frames

def iter_frames(frames_list):
    for num, frame in frames_list:
        yield num, frame


def show_frame(cv2_image, empty) -> Image:
    with empty.container():
        st.image(Image.fromarray(cv2_image), channels='BGR')
    return 

def set_label(label:str, image_label):
    image_label = label

def get_class_counts() -> dict:
    counts = {}
    for cls in CLASSES:
        dirs = list(filter(lambda tup: tup[1] == cls, GAME_STATUSES.items()))
        count = sum([len(os.listdir(os.path.join(DATA_DIR, d))) for d in dirs])
        counts[cls] = count
    return counts



#--------------------#
#    Page Design     #
#--------------------#

st.title("Video Frame Labeller")
st.header(
    "Frame Labels\n" + "\n".join([f"\t{k}:{v}" for k, v in GAME_STATUSES.items()])
)

image_container = st.empty()
video_to_label = st.sidebar.selectbox("Select video to label", video_paths)
with st.spinner("Loading video frames..."):
    frames = get_video_frames(video_to_label)



for frame_num, frame in iter_frames(frames):
    image_label = ""
    btn_act = st.button("Active", on_click=set_label(ACTIVE, image_label))
    btn_inact = st.button("Inactive", on_click=set_label(INACTIVE, image_label))
    print(frame_num, image_label)
    while image_label == "": 
        show_frame(frame, image_container)
    save_frame(
        frame, 
        game_state="UNKNOWN",
        label=image_label, 
        video_from=video_to_label, 
        frame_num=frame_num
    )
