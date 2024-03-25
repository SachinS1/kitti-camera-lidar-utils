
import numpy as np
import cv2
import glob
import os
from utils import *


data_root = "/home/neo/Desktop/Sachin/kitti-camera-lidar-utils/"

images_dir = data_root + "images/"
labels_dir = data_root + "labels/"
pcl_dir = data_root + "point_clouds/"
calib_dir = data_root + "calib/"
print(os.listdir(images_dir))

for images in os.listdir(images_dir):
    img_path = images_dir+images
    label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    label_path = os.path.join(labels_dir, label_filename)
    # view_image(images_dir+images)
    view_image_with_labels(img_path, label_path)

cv2.destroyAllWindows()