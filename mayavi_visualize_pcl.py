import numpy as np
from mayavi import mlab

def read_kitti_bin(bin_path):
   
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def visualize_point_cloud(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:,3]
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(800, 600))
    mlab.points3d(x, y, z, intensity, mode="point",colormap='viridis', figure=fig)

    mlab.show()

if __name__ == "__main__":
    bin_path = "point_clouds/000000.bin"
    points = read_kitti_bin(bin_path)
    visualize_point_cloud(points)
