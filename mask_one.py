import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import os
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
#os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
from PIL import Image
import pdb
import shutil
import os
import sys
import torch
from concurrent.futures import ProcessPoolExecutor

track_class = {'car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}
nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/data/tijaz/anotherDataset/v1.0-mini', verbose=False)

def get_instance_dict():
    instance_dict = {}
    i = 1
    for instance in nusc.instance:
        instance_token = instance['token']
        if instance_token not in instance_dict.keys():
            instance_dict[instance_token] = i
            i = i + 1

    return instance_dict

def getID(instance_dict, instance_token):
    return instance_dict[instance_token]

def mkdir(path):

    folder = os.path.exists(path)

    if not folder: 
        os.makedirs(path)  

def get_dynamic_objects_token(sample_data_token):
    sd_rec = nusc.get('sample_data', sample_data_token)
    s_rec = nusc.get('sample', sd_rec['sample_token'])
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    objects_token = []
    for ann_rec in ann_recs:
        # Augment sample_annotation with token information. 
        ann_rec['sample_annotation_token'] = ann_rec['token']
        # ann_rec['sample_data_token'] = sample_data_token
        ann_rec['filename'] = sd_rec['filename']
        if len(ann_rec['attribute_tokens']) == 0:
            continue
        attribute_name = nusc.get('attribute', ann_rec['attribute_tokens'][0])['name']
        
        if attribute_name == 'vehicle.moving' or attribute_name == 'pedestrian.moving' or attribute_name == 'cycle.with_rider':
            objects_token.append(ann_rec['instance_token']) 
    return objects_token

def get_boxes(nusc, sample_data_token, selected_anntoken=None):
        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)
        curr_sample_record = nusc.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            boxes = list(map(nusc.get_box, curr_sample_record['anns']))

        else:
            prev_sample_record = nusc.get('sample', curr_sample_record['prev'])

            curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
            prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]
            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            t = max(t0, min(t1, t))

            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] in prev_inst_map and curr_ann_rec['token'] == selected_anntoken:
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
                    return box
                else:
                    continue

        return []         
    
def write_mask(part_num, data_path, name, sample_data_token, camera_name, number):
    sd_rec = nusc.get('sample_data', sample_data_token)
    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    s_rec = nusc.get('sample', sd_rec['sample_token'])
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    img = Image.open(os.path.join(nusc.dataroot, sd_rec['filename']))

    picture = img.convert("RGBA")
    pixel_data = picture.load()
    picture_dynamic = picture.copy()
    picture_static = picture.copy()
    picture_copy = picture.copy()
    static_data = picture_static.load()

    width = 1600
    height = 900
    
    all_mask = np.zeros((height, width)).astype('int32')
    for ann_rec in ann_recs:
        img = Image.open(os.path.join(nusc.dataroot, sd_rec['filename']))

        picture = img.convert("RGBA")
        picture_dynamic = picture.copy()
        dynamic_data = picture_dynamic.load()
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token
        ann_rec['filename'] = sd_rec['filename']
        if len(ann_rec['attribute_tokens']) == 0:
            continue
        attribute_name = nusc.get('attribute', ann_rec['attribute_tokens'][0])['name']
        if ann_rec['instance_token'] in get_dynamic_objects_token(sample_data_token):
            semseg_mask = np.zeros((height, width)).astype('int32')
            if sd_rec['is_key_frame']:
                box = nusc.get_box(ann_rec['token']) 
            else:
                box = get_boxes(nusc, sample_data_token, ann_rec['token'])
                
            if not box:
                continue   
            box.translate(-np.array(pose_rec['translation']))  
            box.rotate(Quaternion(pose_rec['rotation']).inverse)
     
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            corners_3d = box.corners()
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d = corners_3d[:, in_front]

            corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

            final_coords = post_process_coords(corner_coords)

            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords
            x1 = int(min_x)
            y1 = int(min_y)
            x2 = int(max_x)
            y2 = int(max_y)

            b = list(set(ann_rec['category_name'].split('.')).intersection(track_class))
            if len(b) == 0:
                continue
            else:
                semseg_mask[y1:y2, x1:x2] = 1
                print("one object\n")
                all_mask[y1:y2, x1:x2] = 1

            for i in range(900):
                for j in range(1600):
                    if semseg_mask[i, j] == 0:
                        dynamic_data[j, i] = (255, 255, 255, 0)
            image_data = []
            for row in all_mask:
                image_data.extend([0 if pixel == 0 else 255 for pixel in row])
            image = Image.new("1", (len(all_mask[0]), len(all_mask)))
            image.putdata(image_data)
            print("Writing dynamic mask for instance: ", ann_rec['instance_token'])

            picture = Image.fromarray(np.uint8(all_mask), mode="1")
            semseg_mask_tensor = torch.from_numpy(semseg_mask)
            
            dynamic_sparse_path = data_path + "_dynamic" + "_" + ann_rec['instance_token'] + "/sparse/"
            print("Dynamic sparse path: ", dynamic_sparse_path)
            if not os.path.exists(dynamic_sparse_path):

                if os.path.exists(data_path + "/sparse/"):
                    shutil.copytree(data_path + "/sparse/", dynamic_sparse_path)
            
            datapath = data_path + "_dynamic" + "_" + ann_rec['instance_token'] + "/images/"
            mkdir(datapath)
            print("Dynamic image path: ", datapath)
            datapath_mask = datapath + "mask_" + ann_rec['filename'].split('/')[-1].rsplit('.', 1)[0] + ".png"
            datapath_full = datapath + ann_rec['filename'].split('/')[-1].rsplit('.', 1)[0] + ".png"
            mask_path = datapath + ann_rec['filename'].split('/')[-1].rsplit('.', 1)[0] + ".pt"
            print("Mask path: ", mask_path)
            torch.save(semseg_mask_tensor, mask_path)
            picture_dynamic.save(datapath_mask, "PNG")
            picture_copy.save(datapath_full, "PNG")

        else:
            continue
        
    for i in range(900):
        for j in range(1600):
            if all_mask[i, j] == 1:
                static_data[j, i] = (255, 255, 255, 0)
    for i in range(1,part_num+1):
        datapath = data_path + "_static_part" + str(i) + "/images/"
        mkdir(datapath)
        image_basename = ann_rec['filename'].split('/')[-1].rsplit('.', 1)[0]
        datapath_img = datapath + image_basename + ".png"
        datapath_mask = datapath + "mask_" + image_basename + ".png"
        mask_path = datapath + image_basename + ".pt"
        picture_static.save(datapath_img, "PNG")
        # Create a full mask (all ones) for static
        static_mask = np.ones((height, width), dtype='int32')
        static_mask_tensor = torch.from_numpy(static_mask)
        # Save mask as .pt
        torch.save(static_mask_tensor, mask_path)
        # Optionally, save mask as PNG for visualization
        mask_img = Image.fromarray((static_mask * 255).astype(np.uint8))
        mask_img.save(datapath_mask, "PNG")
                
def get_all_tokens(scene_name):
    scene_record = next(scene for scene in nusc.scene if scene['name'] == scene_name)
    current_token = scene_record['first_sample_token']
    frame_token_list = []
    sample_record = nusc.get('sample', current_token)
    for sensor, sensor_data_token in sample_record['data'].items():
        if sensor not in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
            continue
        token = sample_record['data'][sensor]
        while token != '':
            data = nusc.get('sample_data', token)
            frame_token_list.append(token)
            token = data["next"]
            
    return frame_token_list

def mask_dynamic(data_path, part_num,scene_name=None):
    print("Start writing masks")
    print("Part number: ", part_num)
    print("Data path: ", data_path)
    print("Scene name: ", scene_name)
    frame_token_list = get_all_tokens(scene_name)
    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = []
        for i in frame_token_list:
            futures.append(executor.submit(write_mask, part_num, data_path, 0, i, 0, 0))
        for future in futures:
            future.result()  # Wait for all to finish
