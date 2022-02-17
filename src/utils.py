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

