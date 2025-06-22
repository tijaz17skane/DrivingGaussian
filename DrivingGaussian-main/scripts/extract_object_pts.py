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
    
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
import torch
if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="")
    parser.add_argument("--meta_file", default="transform.json")

    args = parser.parse_args()
    root_path = args.root_path+"/"
    sensor_path = root_path+args.meta_file
    with open(sensor_path, "r") as f:
        data = json.load(f)
    frames = data["frames"]
    lidar_frames = data["lidar_frames"]
    objs_points = {}
    objs_colors = {}
    for lidar_frame in tqdm(lidar_frames):
            l2w = lidar_frame["transform_matrix"]
                # pcd
            file_path = root_path + lidar_frame["file_path"]
            # points = np.fromfile(file=file_path, dtype=np.float32, count=-1).reshape([-1,5])[:,0:3]
            pcd_data = o3d.io.read_point_cloud(file_path)
            points = np.array(pcd_data.points)
            
            bbox3d = []
            anns = lidar_frame["objects"]
            for ann in anns:
                if ann["is_moving"]:
                    if ann["instance_token"] not in objs_points:
                        objs_points[ann["instance_token"]] = []
                        objs_colors[ann["instance_token"]] = []
                    bbox3d = np.array(ann["bbox3d"])
                    points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                    torch.from_numpy(bbox3d[np.newaxis,np.newaxis, :]))
                    mask = (points_in_boxes[0].sum(-1).bool()).numpy()
                    points_masked = points[mask]
                    homogeneous_positions = np.hstack([points_masked , np.ones((points_masked.shape[0], 1))])
                    transformed_positions = np.dot(l2w, homogeneous_positions.T).T[:, :3]
                    align = lidar_frame["align"]
                    found_dict = [d for d in frames if d.get('filename') in align]
                    for frame in found_dict:
                        rgb=cv2.imread(root_path + frame["file_path"]
                            )
                        c2w = frame["transform_matrix"]
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
                        for m in transformed_positions:
                            m_1= np.array([m[0],m[1],m[2],1])
                            uv_homogeneous = intrinsic_matrix @ w2c @ m_1
                            u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)
                            if 0 <= u < w and 0 < v < h and uv_homogeneous[2]>0:
                                rgb_point = rgb[v, u]
                                objs_points[ann["instance_token"]].append(m)
                                objs_colors[ann["instance_token"]].append(rgb_point)
              
    save_path = root_path + f"aggregate_lidar/dynamic_objects/"
    if not os.path.exists(root_path + f"aggregate_lidar/"):
        os.mkdir(root_path + f"aggregate_lidar/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for key in objs_points:
        tmp_points = objs_points[key]
        if len(tmp_points) == 0:
            continue
        tmp_rgb = objs_colors[key]
        tmp_points = np.stack(tmp_points, axis=0)
        tmp_rgb = np.stack(tmp_rgb, axis=0)/255
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp_points)
        pcd.colors = o3d.utility.Vector3dVector(tmp_rgb)
        o3d.io.write_point_cloud(save_path + f"{key}.ply", pcd)
