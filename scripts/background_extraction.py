from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os.path as osp
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, transform_matrix,points_in_box
from functools import reduce
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors
from nuscenes.utils.data_classes import LidarPointCloud
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

import joblib
import plotly.graph_objs as go
import os
import open3d as o3d

def get_prospect(
                 nusc: 'NuScenes',
                 sample_tokens_list: List,
                 target_labels_dict: Dict
                ):
    
    all_prospect_in_scene=[]
    filtered_labels = np.zeros((0,), dtype=np.uint8)
    
    for sample_token in sample_tokens_list:
    
        sample = nusc.get('sample', sample_token)
        cam = sample['data']['LIDAR_TOP']
        ann_tokens = sample['anns']
        filtered_ann_tokens = []

        for ann_token in ann_tokens:
            annotation = nusc.get('sample_annotation', ann_token)
            if annotation['category_name'] in target_labels_dict:
                print(annotation['category_name'])
                filtered_ann_tokens.append(ann_token)

        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=filtered_ann_tokens)
        current_sd_record = nusc.get('sample_data', cam)
        pcl_path = osp.join(nusc.dataroot, current_sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        lidarseg_labels_filename = osp.join(nusc.dataroot, nusc.get('lidarseg', cam)['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

        filtered_points = []
        points = pc.points[:4, :]
        temp=[]
        for box in boxes:
            temp = points[:3, :]
            mask = points_in_box(box, temp)
            points_in_current_box = points[:, mask]
            filtered_points.append(points_in_current_box)

            points_label_in_current_box= points_label[mask]

            assert len(points_in_current_box.T) == len(points_label_in_current_box)
            filtered_labels = np.concatenate((filtered_labels,points_label_in_current_box))

        filtered_points = np.hstack(filtered_points)

        current_pose_rec = nusc.get('ego_pose', current_sd_record['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

        current_cs_rec = nusc.get('calibrated_sensor', current_sd_record['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),inverse=False)

        trans_matrix = reduce(np.dot, [global_from_car, car_from_current])
        filtered_points[:3, :] = trans_matrix.dot(np.vstack((filtered_points[:3, :], np.ones(filtered_points.shape[1]))))[:3, :]

        all_points = filtered_points.T[:,:3]
        all_prospect_in_scene.append(all_points)
    
    all_prospect_in_scene = np.vstack(all_prospect_in_scene)
    
    return all_prospect_in_scene , filtered_labels
    
def concatenate_scene_points(nusc: 'NuScenes',
                             sample_tokens_list: List,
                            ):
    
    # Initialize variables to store points and labels.
    all_points = np.zeros((0, 3))
    all_labels = np.zeros((0,), dtype=np.uint8)
    # Iterate through sample tokens.
    for sample_token in sample_tokens_list:
        # Load point cloud and labels.
        ref_sd_token = nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        ref_sd_record = nusc.get('sample_data', ref_sd_token)
        pcl_path = osp.join(nusc.dataroot, ref_sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        current_pose_rec = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

        current_cs_rec = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),inverse=False)

        trans_matrix = reduce(np.dot, [global_from_car, car_from_current])
        pc.transform(trans_matrix)

        points = pc.points.T[:,:3]

        lidarseg_labels_filename = osp.join(nusc.dataroot, nusc.get('lidarseg', ref_sd_token)['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
        assert len(points) == len(points_label)

        # Concatenate points
        all_points = np.vstack((all_points, points))
        # all_points = np.vstack((all_points, rotated_points))
        all_labels = np.concatenate((all_labels, points_label))
    
    return all_points,all_labels

def get_nsweeps_points(
                        nusc: 'NuScenes',
                        sample_tokens_list: List,
                        channel : str,
                        nsweeps: int = 5,
                        ):

    all_points_rem = np.zeros((0, 3))
    for sample_token in sample_tokens_list:
        sample_record =  nusc.get('sample', sample_token)
        ref_sd_token = sample_record['data']['LIDAR_TOP']
        ref_sd_record = nusc.get('sample_data', ref_sd_token)
        pcl_path = osp.join(nusc.dataroot, ref_sd_record['filename'])


        pc_key = LidarPointCloud.from_file(pcl_path)
        pc_rem,_ = LidarPointCloud.from_file_multisweep(nusc, sample_record, channel, channel,nsweeps = 5)

        # Get the coordinates of points in pc_key and pc_rem
        points_key = pc_key.points
        points_rem = pc_rem.points
        print(points_rem.shape)
        # indices_not_in_key = np.setdiff1d(np.arange(points_rem.shape[1]), np.arange(points_key.shape[1]))
        indices_not_in_key = np.where(~np.isin(points_rem.T, points_key.T).all(axis=1))[0]
        print(indices_not_in_key.shape)
        points_not_in_key = points_rem[:,indices_not_in_key]

        current_pose_rec = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

        current_cs_rec = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),inverse=False)

        trans_matrix = reduce(np.dot, [global_from_car, car_from_current])
        points_not_in_key[:3, :] = trans_matrix.dot(np.vstack((points_not_in_key[:3, :], np.ones(points_not_in_key.shape[1]))))[:3, :]

        points =  points_not_in_key.T[:,:3]
        all_points_rem = np.vstack((all_points_rem, points))
    return all_points_rem

def delete_target_points(points,labels,colors,target_labels_dict:dict):
    points_in = points.copy()
    labels_in = labels.copy()
    target_labels_list = list(target_labels_dict.values())
    for target_value in target_labels_list:
        colors_array = (colors[labels_in]*255).astype(int)
        indices = np.where(np.all(colors_array == target_value, axis=1))
        points_in =  np.delete(points_in, indices, axis=0)
        labels_in =  np.delete(labels_in, indices, axis=0)
    return points_in, labels_in

def delete_target_points_2(points,labels,colors,target_labels_dict:dict):
    points_in = points.copy()
    labels_in = labels.copy()
    target_labels_list = list(target_labels_dict.values())
    for target_value in target_labels_list:
        colors_array = (labels_in*255).astype(int)
        indices = np.where(np.all(colors_array == target_value, axis=1))
        points_in =  np.delete(points_in, indices, axis=0)
        labels_in =  np.delete(labels_in, indices, axis=0)
    return points_in, labels_in

def predict_labels_for_point_cloud(point_cloud, scene_token):
    
    input_dir = f"./model/KNN/{scene_token}"
    model_filename = os.path.join(input_dir, 'knn_model_all.joblib')

    # Load the trained KNN model
    loaded_knn_model = joblib.load(model_filename)

    # Load the scaler object
    scaler_filename = os.path.join(input_dir, 'scaler_model_all.joblib')
    loaded_scaler = joblib.load(scaler_filename)

    # Use the loaded scaler to transform new data
    X_new_scaled = loaded_scaler.transform(point_cloud)

    # Predict labels using the loaded KNN model
    predicted_labels = loaded_knn_model.predict(X_new_scaled)

    return predicted_labels.astype(int)

def get_difference_points(
        all_points: np.ndarray,
        all_colors: np.ndarray,
        all_prospect_in_scene: np.ndarray,
        distance_threshold=0.01
    ):
    pcd_prospect = o3d.geometry.PointCloud()
    pcd_prospect.points = o3d.utility.Vector3dVector(all_prospect_in_scene)
    
    kdtree = o3d.geometry.KDTreeFlann(pcd_prospect)
    
    keep_indices = []
    for i, point in enumerate(all_points):
        [k, idx, _] = kdtree.search_radius_vector_3d(point, distance_threshold)
        if k == 0:
            keep_indices.append(i)
    
    filtered_points = all_points[keep_indices]
    filtered_colors = all_colors[keep_indices] if all_colors is not None else None
    
    return filtered_points, filtered_colors

def get_difference_points_unlabels(
        all_points: np.ndarray,
        all_prospect_in_scene: np.ndarray,
        distance_threshold=0.01
    ):
    pcd_prospect = o3d.geometry.PointCloud()
    pcd_prospect.points = o3d.utility.Vector3dVector(all_prospect_in_scene)
    
    kdtree = o3d.geometry.KDTreeFlann(pcd_prospect)
    
    keep_indices = []
    for i, point in enumerate(all_points):
        [k, idx, _] = kdtree.search_radius_vector_3d(point, distance_threshold)
        if k == 0:
            keep_indices.append(i)
    
    filtered_points = all_points[keep_indices]
    
    return filtered_points
