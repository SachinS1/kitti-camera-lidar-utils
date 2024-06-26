import numpy as np
import cv2
from easydict import EasyDict as edict

configs = edict()
configs.lim_x = [0, 50]
configs.lim_y = [-25, 25]
configs.lim_z = [-2.73, 1.27]
configs.bev_width = 608
configs.bev_height = 608

lim_x = [0.0, 50.0]
lim_y = [-25, 25.0]
lim_z = [-2.5, 1]

kitti_classes = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram' ]

def view_image(img_path):

    img = cv2.imread(img_path)

    cv2.imshow("Frame", img)

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def view_image_with_labels(img_path, label_path):
        img = cv2.imread(img_path)

        with open(label_path, "r") as file:
             lines = file.readlines()

        for line in lines:
            data = line.split()
            label = data[0] 
            x1, y1, x2, y2 = map(float, data[4:8])

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 150), 2)

            # cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

        cv2.imshow("Frame", img)

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

def generate_bev_image(pcl_path, vis=False):

    '''
    bev image generation is adapted from the Udacity Self Driving Engineer Nanodegree course.
    '''
    
    lidar_pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)

    # compute bev-map discretization by dividing x-range by the bev-image height
    bev_discret = (lim_x[1] - lim_x[0]) / 608
    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    

    mask = (
        (lidar_pcl[:, 0] > lim_x[0]) & (lidar_pcl[:, 0] < lim_x[1]) &
        (lidar_pcl[:, 1] > lim_y[0]) & (lidar_pcl[:, 1] < lim_y[1]) &
        (lidar_pcl[:, 2] > lim_z[0]) & (lidar_pcl[:, 2] < lim_z[1])
    )
    lidar_pcl = lidar_pcl[mask]

    lidar_pcl_cpy = np.copy(lidar_pcl)

    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))

    # transform all metrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]  
 
    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    idx_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    
    lidar_pcl_hei = lidar_pcl_cpy[idx_height]
    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _, idx_height_unique, counts = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True, return_counts=True)

    lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and 
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(np.abs(lim_z[1] - lim_z[0]))

    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity

    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]

    # # # only keep one point per grid cell
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = lidar_pcl_cpy[indices]

    # # # create the intensity map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:,3])-np.amin(lidar_pcl_int[:, 3]))
    

    # # # create the density map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, 609,609))
    RGB_Map[2, :, :] = density_map # r_map
    RGB_Map[1, :, :] = height_map  # g_map
    RGB_Map[0, :, :] = intensity_map # b_map
    img_rgb = RGB_Map.transpose(1, 2, 0) * 256

    img_rgb = img_rgb.astype(np.uint8)
   
    if vis:

        cv2.imshow('img_rgb', img_rgb)
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    return img_rgb

def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as r:
        lines = r.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}



def convert_labels_to_targets(label_file, lidar_2_cam_matrix):
    '''
    script to convert provided kitti labels to complexyolo required labels
    '''
    with open(label_file, 'r') as r:
        lines = r.readlines()

    total_objects = len(lines)

    for i in range(total_objects):
        label_info = lines[i].strip().split(' ')
        object_class = label_info[0].strip()

        if object_class in kitti_classes:
            
            pass



    # pass