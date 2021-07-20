import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
from obj_detection import Detection
import logging



class Driver:
    def __init__(self):
        parser = ConfigParser()
        parser.read("./properties.config")
        self.image_path = parser.get("detection_config", "image_path")
        self.yolo_weights = parser.get("detection_config", "yolo_weights")
        self.yolo_cfg = parser.get("detection_config", "yolo_cfg")

    def _get_input_images(self):
        all_files = []
        # path = r"moments"  # make sure to put the 'r' in front
        for filename in os.listdir(self.image_path):
            if filename.endswith(".jpg"):
                all_files.append("moments/" + filename)
            else:
                continue
        print(all_files)

        return all_files


if __name__ == "__main__":
    logging.info("Starting Object Detection")
    driver = Driver()
    images = driver._get_input_images()
    detection = Detection()
    # img_path = "./new_images/sample2.jpg"
    # detection.get_crop_hints(img_path, 16/9)
    for img in images:
        logging.info("Successfully Passed The Arguments To The Function")
        detection.get_crop_hints(img, 16/9)

