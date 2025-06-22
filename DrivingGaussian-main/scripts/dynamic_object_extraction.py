from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
import numpy as np
from pyquaternion import Quaternion
import cv2
import os.path as osp
from nuscenes.utils.geometry_utils import view_points, transform_matrix,points_in_box
from functools import reduce
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors
from nuscenes.utils.data_classes import LidarPointCloud,  Box
from nuscenes.utils.color_map import get_colormap
import os
import open3d as o3d

target_moving_object = ['vehicle.moving','cycle.with_rider','pedestrian.moving']
movable_static_object = ['movable_object.barrier','movable_object.debris','movable_object.trafficcone',"static_object.bicycle_rack"]
target_moving_object_other = ['movable_object.pushable_pullable']

def get_all_static_object_instances(nusc, scene):
    first_sample_record = nusc.get('sample', scene['first_sample_token'])
    curr_sample_record = first_sample_record
    instance_static_status = {}

    while curr_sample_record['next'] != '':
        for annotation in curr_sample_record['anns']:
            ans = nusc.get('sample_annotation', annotation)
            instance_token = ans['instance_token']
            category_name = nusc.get('category', nusc.get('instance', instance_token)['category_token'])['name']

            if category_name in movable_static_object:
                instance_static_status[instance_token] = True
            elif category_name in target_moving_object_other:
                instance_static_status[instance_token] = False
            else:
                if instance_token not in instance_static_status:
                    instance_static_status[instance_token] = True
                for attribute_token in ans['attribute_tokens']:
                    attribute_name = nusc.get('attribute', attribute_token)['name']
                    if attribute_name in target_moving_object:
                        instance_static_status[instance_token] = False

        curr_sample_record = nusc.get('sample', curr_sample_record['next'])

    for annotation in curr_sample_record['anns']:
        ans = nusc.get('sample_annotation', annotation)
        instance_token = ans['instance_token']
        category_name = nusc.get('category', nusc.get('instance', instance_token)['category_token'])['name']

        if category_name in movable_static_object:
            instance_static_status[instance_token] = True
        elif category_name in target_moving_object_other:
            instance_static_status[instance_token] = False
        else:
            if instance_token not in instance_static_status:
                instance_static_status[instance_token] = True
            for attribute_token in ans['attribute_tokens']:
                attribute_name = nusc.get('attribute', attribute_token)['name']
                if attribute_name in target_moving_object:
                    instance_static_status[instance_token] = False

    static_instances = [token for token, is_static in instance_static_status.items() if is_static]

    return static_instances

def get_all_instances(nusc, scene):
    first_sample_record = nusc.get('sample', scene['first_sample_token'])
    curr_sample_record = first_sample_record
    instance_set = set()
    
    while curr_sample_record['next'] != '':
        # instance_token
        for annotation in curr_sample_record['anns']:
            ans = nusc.get('sample_annotation', annotation)
            instance_token = ans['instance_token']
            category_name = nusc.get('category', nusc.get('instance', instance_token)['category_token'])['name']
            if category_name in target_moving_object_other:
                instance_set.update([nusc.get('sample_annotation', annotation)['instance_token']])
            else:    
                for attribute_token in ans['attribute_tokens']:
                    if nusc.get('attribute', attribute_token)['name'] in target_moving_object:
                        print(nusc.get('attribute', attribute_token)['name'])
                        instance_set.update([nusc.get('sample_annotation', annotation)['instance_token']])
        # instances = [nusc.get('sample_annotation', annotation)['instance_token'] for annotation in curr_sample_record['anns']]
        # instance_set.update(instances)
        curr_sample_record = nusc.get('sample', curr_sample_record['next'])

    for annotation in curr_sample_record['anns']:
        ans = nusc.get('sample_annotation', annotation)
        for attribute_token in ans['attribute_tokens']:
            if nusc.get('attribute', attribute_token)['name'] in target_moving_object:
                print(nusc.get('attribute', attribute_token)['name'])
                instance_set.update([nusc.get('sample_annotation', annotation)['instance_token']])
    instance_list = list(instance_set)

    return instance_list

def get_selected_anntokens(nusc,curr_sample_record,instance_token):
    
    instance_annotations = []
    annotations = [nusc.get('sample_annotation', annotation) for annotation in curr_sample_record['anns']]
    instance_annotations += [ann['token'] for ann in annotations if ann['instance_token'] == instance_token]
    return instance_annotations

def my_get_boxes(nusc, sample_data_token: str,instance_token:str) :

        sd_record = nusc.get('sample_data', sample_data_token)
        curr_sample_record = nusc.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:

            selected_anntokens = get_selected_anntokens(nusc,curr_sample_record,instance_token)
            if(len(selected_anntokens)==0): return []
            boxes = list(map(nusc.get_box, selected_anntokens))

        else:
            prev_sample_record = nusc.get('sample', curr_sample_record['prev'])

            curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
            prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]

            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}
            
            if instance_token not in prev_inst_map: return []
            
            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] == instance_token:

                    prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                    # Interpolate center.
                    center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                 curr_ann_rec['translation'])]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

                    box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
                else:
                    # If not, simply grab the current annotation.
                    # box = self.get_box(curr_ann_rec['token'])
                    continue

                boxes.append(box)
        return boxes

def transform_boxes(nusc,sd_record,boxes):
    box_list = []
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    for box in boxes:
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        box_list.append(box)
    return box_list

def concatenate_prospect(nusc,sample_data_list,instance_token ):
    all_points_in_scene=[]
    for sd_record in sample_data_list:
        sample_record = nusc.get('sample', sd_record['sample_token'])
        assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

        boxes = my_get_boxes(nusc, sd_record['token'],instance_token)
        if len(boxes) == 0: continue
        boxes = transform_boxes(nusc,sd_record,boxes)
             
        pcl_path = osp.join(nusc.dataroot, sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        # pc,_ = LidarPointCloud.from_file_multisweep(nusc, sample_record, channel, channel,nsweeps = 1)
        points = pc.points[:4, :]
        
        temp=[]
        for box in boxes:
            temp = points[:3, :].copy()
            mask = points_in_box(box, temp)
            c = np.array(get_colormap()[box.name]) / 255.0
            print(box.name)

            points_in_current_box = points[:, mask]

            corners = box.corners(wlh_factor=1.0)
            p1 = corners[:, 0]
            p_x = corners[:, 4]
            p_y = corners[:, 1]
            p_z = corners[:, 3]
            v = points_in_current_box[:3, :] - p1.reshape((-1, 1))

            i = p_x - p1
            j = p_y - p1
            k = p_z - p1
            iv = np.dot(i, v) / np.linalg.norm(i)
            jv = np.dot(j, v) / np.linalg.norm(j)
            kv = np.dot(k, v) / np.linalg.norm(k)

            new_coordinates = np.vstack((iv, jv, kv)).T
            all_points_in_scene.append(new_coordinates)
    return np.vstack(all_points_in_scene),c

def concatenate_prospect_not_moving(nusc,sample_data_list,instance_token ):
    all_points_in_scene=[]
    for sd_record in sample_data_list:
        sample_record = nusc.get('sample', sd_record['sample_token'])
        assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

        boxes = my_get_boxes(nusc, sd_record['token'],instance_token)
        if len(boxes) == 0: continue
        boxes = transform_boxes(nusc,sd_record,boxes)
        
        pcl_path = osp.join(nusc.dataroot, sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        # pc,_ = LidarPointCloud.from_file_multisweep(nusc, sample_record, channel, channel,nsweeps = 1)
        points = pc.points[:4, :]
        
        temp=[]
        filtered_points = []
        for box in boxes:
            temp = points[:3, :].copy()
            mask = points_in_box(box, temp)
            c = np.array(get_colormap()[box.name]) / 255.0
            print(box.name)

            points_in_current_box = points[:, mask]
            filtered_points.append(points_in_current_box)
            
        filtered_points = np.hstack(filtered_points)
        current_pose_rec = nusc.get('ego_pose', sd_record['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

        current_cs_rec = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),inverse=False)

        trans_matrix = reduce(np.dot, [global_from_car, car_from_current])
        filtered_points[:3, :] = trans_matrix.dot(np.vstack((filtered_points[:3, :], np.ones(filtered_points.shape[1]))))[:3, :]
        filtered_points=filtered_points.T[:,:3]
        all_points_in_scene.append(filtered_points)
        
    return np.vstack(all_points_in_scene),c
