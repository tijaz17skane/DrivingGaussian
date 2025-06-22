import argparse
import random
import numpy as np
import json
import json  
import os
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def get_box_corners(center, dimensions, orientation):
    
    cx, cy, cz = center
    length, width, height = dimensions
    q = orientation

    dx  = length / 2.0
    dy = width / 2.0
    dz = height / 2.0

    corners = np.array(
        [
            [dx, dy, dz],
            [-dx, dy, dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
        ]
    )

    rotation = R.from_quat([q[1], q[2], q[3], q[0]])
    rotated_corners = rotation.apply(corners)

    world_corners = rotated_corners + center

    return world_corners

def undistort_nearest(cv_image, k, d,fisheye = True):

    if fisheye:
       mapx, mapy = cv2.fisheye.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)
    else:
       mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)

    cv_image_undistorted = cv2.remap(cv_image, mapx, mapy, cv2.INTER_NEAREST)

    return cv_image_undistorted

def extract_value_between(string, start_char, end_char):
    start_index = string.find(start_char) + len(start_char)
    end_index = string.find(end_char, start_index)
    if(end_char ==""):
        end_index = len(string)
    if start_index != -1 :
        return string[start_index:end_index]
    else:
        return None
    
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
import torch
if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="/mnt/data/tijaz/anotherDataset/v1.0-mini/extracted/fcbccedd61424f1b85dcbf8f897f9754")
    parser.add_argument("--meta_file", default="transform.json")

    args = parser.parse_args()
    root_path = args.root_path+"/"

    lidars_folder = root_path+"lidars/"
    os.makedirs(root_path+'colmap/sparse/lidar/', exist_ok=True)
    sensor_path = root_path+args.meta_file
    moving_gids = []
    with open(sensor_path, "r") as f:
        data = json.load(f)
    tmp_points= []
    tmp_rgb = []
    transform =np.array(data["transform"])
    with open(root_path+'colmap/sparse/lidar/points3D.txt','w') as j:
        i = 1
        frames = data["frames"]
        lidar_frames = data["lidar_frames"]
        
        for lidar_frame in tqdm(lidar_frames):
                l2w = lidar_frame["transform_matrix"]
                l2w = transform @ np.array(l2w)

                file_path = root_path + lidar_frame["file_path"]
            # points = np.fromfile(file=file_path, dtype=np.float32, count=-1).reshape([-1,5])[:,0:3]
                pcd_data = o3d.io.read_point_cloud(file_path)
                points = np.array(pcd_data.points)
              
                bbox3d = []
                anns = lidar_frame["anns"]
                for ann in anns:
                    # if ann["is_moving"]:
                        bbox3d.append(ann["bbox3d"])
                bbox3d = np.array(bbox3d)
                points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                            torch.from_numpy(bbox3d[np.newaxis, :]))
                mask = ~(points_in_boxes[0].sum(-1).bool()).numpy()
                points = points[mask]
                homogeneous_positions = np.hstack([points , np.ones((points.shape[0], 1))])
                transformed_positions = np.dot(l2w, homogeneous_positions.T).T[:, :3]
                align = lidar_frame["align"]
                found_dict = [d for d in frames if d.get('filename') in align]
                for frame in found_dict:
                    rgb=cv2.imread(root_path + frame["file_path"]
                        )
                    c2w =transform @ frame["transform_matrix"]
                    w2c = np.linalg.inv(c2w)
                    h=frame['h']
                    w=frame['w']
                    fl_x=frame['fl_x']
                    fl_y=frame['fl_y']
                    cx=frame['cx']
                    cy=frame['cy']
                    intrinsic_matrix=np.array([[fl_x,0,cx,0],
                                            [0,fl_y,cy,0],
                                            [0,0,1,0],
                                            [0,0,0,1]])
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed_positions))
                    pcd = pcd.voxel_down_sample(voxel_size=0.2)
                    transformed_positions = np.array(pcd.points)
                    for m in transformed_positions:
                        m_1= np.array([m[0],m[1],m[2],1])
                        uv_homogeneous = intrinsic_matrix @ w2c @ m_1
                        u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)
                        if 0 <= u < w and 0 < v < h and uv_homogeneous[2]>0:
                            rgb_point = rgb[v, u]
                            tmp_points.append(m)
                            tmp_rgb.append(rgb_point)
                            error = random.uniform(0,1)
                            j.write(f'{i} {m[0]:.3f} {m[1]:.3f} {m[2]:.3f} {rgb_point[2]} {rgb_point[1]} {rgb_point[0]} {error:.3f} 1 1 2 2 {random.randint(1,300)} {random.randint(1,2000)}\n')
                            i += 1
        
    # tmp_points = np.stack(tmp_points, axis=0)
    # tmp_rgb = np.stack(tmp_rgb, axis=0)/255
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(tmp_points)
    # pcd.colors = o3d.utility.Vector3dVector(tmp_rgb)
    # pcd = pcd.voxel_down_sample(voxel_size=0.1)
    # o3d.io.write_point_cloud(root_path+'colmap/sparse/lidar/lidar.ply', pcd)
        # tmp = np.array(pcd.points)
        # # tmp -=tmp.mean(axis=0)
        # for m in tmp:
        #     error = random.uniform(0,1)
        #     rgb_point =[125,125,125]
        #     j.write(f'{i} {m[0]:.3f} {m[1]:.3f} {m[2]:.3f} {rgb_point[2]} {rgb_point[1]} {rgb_point[0]} {error:.3f} 1 1 2 2 {random.randint(1,300)} {random.randint(1,2000)}\n')
        #     i += 1
