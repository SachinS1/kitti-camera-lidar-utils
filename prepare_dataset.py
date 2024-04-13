
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from utils import *

class KittiData(Dataset):
    def __init__(self, root = '/home/neo/Desktop/Sachin/kitti-camera-lidar-utils/', set = 'train', type = 'velodyne_train'):
        
        self.type = type
        self.root = root
        self.data_path = self.root

        self.lidar_path = os.path.join(self.data_path, "point_clouds/")
        self.image_path = os.path.join(self.data_path, "images/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "labels/")

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()   

    def __getitem__(self, i):


        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        image_file = self.image_path + '/' + self.file_list[i] + '.png'

        if self.type == 'velodyne_train':

            calib = load_kitti_calib(calib_file)

            converted_labels = convert_labels_to_targets(label_file, calib["Tr_velo2cam"] )


            # target = get_target(label_file,calib['Tr_velo2cam'])
            #print(target)
            #print(self.file_list[i])
            
            ################################
            # load point cloud data
            a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

            img_rgb = generate_bev_image(lidar_file)    
            return img_rgb

        elif self.type == 'velodyne_test':
            NotImplemented

        else:
            raise ValueError('the type invalid')


    def __len__(self):
        return len(self.file_list)
    

kitti_data = KittiData()
kitti_data.__getitem__(2)