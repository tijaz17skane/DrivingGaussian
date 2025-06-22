# import torch
import numpy as np
from PIL import Image
import os 
import sys
# import chamfer
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
from nuscenes.utils.data_classes import LidarPointCloud,  Box
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from functools import reduce
import json
import numpy as np
import open3d as o3d
import torch
# from torchvision.transforms import ToTensor
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from torchvision.transforms import Compose
import cv2
import os
# import torch.nn.functional as F
import matplotlib.pyplot as plt
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
from nuscenes.utils.data_classes import LidarPointCloud,  Box
import multiprocessing
from multiprocessing import Pool
import multiprocessing
from multiprocessing import set_start_method, get_context
import os
import sys
import pdb
import time
import yaml
import torch
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation

import open3d
import open3d as o3d
from copy import deepcopy
import os.path as osp
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap
import json
def rotate_box_vertices_around_z(bbox):
    """Rotate the vertices of a 3D bounding box around the z-axis."""
    center = bbox[:3]
    size = bbox[3:6]
    rz = bbox[6]+np.pi/2
    half_size = size / 2.0 
    vertices = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],
        [+half_size[0], -half_size[1], -half_size[2]],
        [+half_size[0], +half_size[1], -half_size[2]],
        [-half_size[0], +half_size[1], -half_size[2]],
        [-half_size[0], -half_size[1], +half_size[2]],
        [+half_size[0], -half_size[1], +half_size[2]],
        [+half_size[0], +half_size[1], +half_size[2]],
        [-half_size[0], +half_size[1], +half_size[2]],
    ])

    # Rotate vertices around the z-axis
    cos_theta = np.cos(rz)
    sin_theta = np.sin(rz)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1],
    ])
    rotated_vertices = np.dot(vertices, rotation_matrix.T) + center

    return rotated_vertices
def get_R_T(quat, trans):
    """
        Args:
            quat: eg.[w,x,y,z]
            trans: eg.[x',y',z']
        Return:
            RT
    """
    RT = np.eye(4)
    RT[:3,:3] = Quaternion(quat).rotation_matrix
    RT[:3,3] = np.array(trans)
    return RT
def apply_4x4(RT, xyz):

    ones = np.ones_like(xyz[...,0:1])
    xyz1 = np.concatenate([xyz, ones], -1)
    xyz2 = RT@xyz1.T
    return xyz2.T[...,0:3]
import argparse    
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/mnt/data/tijaz/anotherDataset/v1.0-mini', help='path to Nuscenes dataset')
parser.add_argument('--out_dir', type=str, default='/mnt/data/tijaz/anotherDataset/v1.0-mini/extracted', help='path to save extracted data')
args = parser.parse_args()
target_moving_object = ['vehicle.moving','cycle.with_rider','pedestrian.moving']#
nusc =NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=True)
import cv2
from PIL import Image
from pathlib import Path
def prospect_mask_img(nusc,sample_data_token,):

    data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token,)
    processed_coords_all = []
    anns = []
    for box in boxes:
    # 3D bounding boxes to 2D projection
        ann = nusc.get('sample_annotation', box.token)
        attr = ann["attribute_tokens"]
        if len(attr) == 0:
            continue
        if nusc.get('attribute', (attr[0]))['name'] in target_moving_object:
            corners = view_points(box.corners(),camera_intrinsic , normalize=True)[:2, :]
            corner_coords = corners.T.tolist()
            # Coord processing via post_process_coords fn
            processed_coords = post_process_coords(corner_coords)
            if processed_coords is not None:
                b_translation = ann['translation']
                b_rotation = ann['rotation']
                b_transform_matrix = get_R_T(b_rotation, b_translation)
                anns.append(
                    {
                        "type": nusc.get('attribute', (attr[0]))['name'],
                        "ann_token":ann["token"] ,
                        "instance_token": ann['instance_token'],
                        "transform_matrix": b_transform_matrix.tolist(),
                        # "is_moving": is_moving,
                    }
                )
                processed_coords_all.append(processed_coords)
    return anns,processed_coords_all
def mask_img(nusc,sample_data_token,img):
    th = 100
    img_shape = img.shape
    anns,processed_coords_all = prospect_mask_img(nusc,sample_data_token)
    new_anns = []
    inst2img = {}
    mask = np.ones((img_shape[0],img_shape[1]),dtype=np.uint8)*255
    for i,bbox in enumerate(processed_coords_all):
        min_y, min_x, max_y, max_x = [int(np.round(coord)) for coord in bbox]
        mask[min_x:max_x,min_y:max_y] = 0
        if max_x-min_x<th or max_y-min_y<th:
            continue
        ann = anns[i]
        ann["bbox"] = [min_x, min_y, max_x, max_y]
        inst2img[ann["instance_token"]] = img[min_x:max_x,min_y:max_y]
        new_anns.append(ann)
    return inst2img,new_anns,mask
class NuscDataExtractor:
    def __init__(self, num_workers,out_dir) -> None:
        self.num_workers = num_workers
        self.out_dir = Path(out_dir)
    def extract_one(self, scene_token) -> int:
        out_dir = self.out_dir
        segment_name = None
        segment_out_dir = None
        sensor_params = None
        camera_frames = []
        lidar_frames = []
        annotations = []
        scene = nusc.get('scene', scene_token)
        # sample_token = scene['first_sample_token']
        i = 0
        # while 1:
        #     sample = nusc.get('sample', sample_token)
        #     segment_name = sample['scene_token']
        #     segment_out_dir = out_dir / segment_name
        #     if i == 0:
        #         sensor_params = self.extract_sensor_params(sample)
        #     if sample['next'] == "":
        #         break
        #     sample_token = sample['next']
        #     i += 1
        segment_out_dir = out_dir / scene_token
        transform=self.get_world_center(scene,segment_out_dir)
        camera_frames=self.extact_frame_images(scene,segment_out_dir,)
        lidar_frames=self.extract_frame_lidars(scene,segment_out_dir,)
        # annotations.append(self.extract_frame_annotation(sample))
        camera_frames.sort(key=lambda frame: f"{frame['file_path']}")
        lidar_frames.sort(key=lambda frame: f"{frame['file_path']}")
        annjson = self.extract_frame_annotation(scene)
        meta = {"frames": camera_frames, "lidar_frames": lidar_frames}
        meta["transform"]=transform.tolist()
        with open(segment_out_dir / "transform.json", "w") as fout:
            json.dump(meta, fout, indent=4)
        with open(segment_out_dir / "annotation.json", "w") as fout:
            json.dump(annjson, fout, indent=4)
        return 

    def extact_frame_images(
        self, scene, segment_out_dir,
    ) : 
        first_sample_token = scene['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        
        frame_images = []

        for camera_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            sample_data_token = first_sample['data'][camera_name]
            while 1:
                sensor = nusc.get('sample_data', sample_data_token)
                filename = sensor['filename'].split("/")[-1].split(".")[0]
                timestamp=sensor['timestamp']
                save_path = segment_out_dir / "images" / camera_name / f"{timestamp}.jpg"
                mask_path = segment_out_dir / "masks" / camera_name / f"{timestamp}.png"
                if not save_path.parent.exists():
                    save_path.parent.mkdir(parents=True)
                if not mask_path.parent.exists():
                    mask_path.parent.mkdir(parents=True)
                Image.open(os.path.join(args.dataroot, sensor['filename'])).save(save_path)
                cam_cs_record = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
                cam_pose_record = nusc.get('ego_pose', sensor['ego_pose_token'])
                intrinsic = np.array(cam_cs_record['camera_intrinsic'])
                extrinsic =  get_R_T(cam_cs_record['rotation'], cam_cs_record['translation'])
                # distortion = camera_params["camera_D"]
                ego_pose = get_R_T(cam_pose_record['rotation'], cam_pose_record['translation'])
                camera2world = ego_pose @ extrinsic  # opencv camera coord
                inst2img,anns,mask = mask_img(nusc,sensor['token'],np.array(Image.open(save_path)))
                Image.fromarray(mask.astype(np.uint8)).save(mask_path)
                for instance_token,img_bbox in inst2img.items():
                    image_path = segment_out_dir / "objects"/instance_token / "images" / f"{timestamp}.jpg"
                    if not image_path.parent.exists():
                        image_path.parent.mkdir(parents=True)
                    Image.fromarray(img_bbox.astype(np.uint8)).save(image_path)
                frame_images.append(
                    {
                        "file_path": save_path.relative_to(segment_out_dir).as_posix(),
                        "filename": sensor['filename'].split("/")[-1],
                        "fl_x": intrinsic[0, 0],
                        "fl_y": intrinsic[1, 1],
                        "cx": intrinsic[0, 2],
                        "cy": intrinsic[1, 2],
                        "w": sensor["width"],
                        "h": sensor["height"],
                        "camera_model": "PINHOLE",
                        "camera": camera_name,
                        "timestamp": timestamp/1.0e6,
                        "anns":anns,
                        # "k1": distortion[0],
                        # "k2": distortion[1],
                        # "k3": distortion[4],
                        # "k4": 0.0,
                        # "p1": distortion[2],
                        # "p2": distortion[3],
                        "transform_matrix": camera2world.tolist(),
                    }
                )
                if sensor['next'] == "":
                    break
                sample_data_token = sensor['next']

        return frame_images
    def get_world_center(
        self, scene, segment_out_dir
    ) :
        first_sample = nusc.get('sample', scene['first_sample_token'])
        sample_data_token = first_sample['data']['LIDAR_TOP']
        all_points = []
        while 1:

            lidar_sensor = nusc.get('sample_data',sample_data_token)
            timestamp=lidar_sensor['timestamp']
            lidar_cs_record = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
            lidar_pose_record = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
            extrinsic = get_R_T(lidar_cs_record['rotation'], lidar_cs_record['translation'])
            ego_pose = get_R_T(lidar_pose_record['rotation'], lidar_pose_record['translation'])
            lidar2world = ego_pose @ extrinsic

            lidar_name = "LIDAR_TOP"
            lidar_points = np.fromfile(os.path.join(args.dataroot, lidar_sensor['filename']), dtype=np.float32).reshape(-1,5)[:,:3]
            lidar_points = np.dot(lidar2world, np.concatenate([lidar_points, np.ones((lidar_points.shape[0], 1))], axis=1).T).T[:,:3]
            all_points.append(lidar_points)
            if lidar_sensor['next'] == "":
                break
            sample_data_token = lidar_sensor['next']
        all_points = np.concatenate(all_points, axis=0)
        transform = np.eye(4)
        transform[:3,3]=-np.mean(all_points, axis=0)
        return transform

    def extract_frame_lidars(
        self, scene, segment_out_dir,
    ) :
        first_sample = nusc.get('sample', scene['first_sample_token'])
        sample_data_token = first_sample['data']['LIDAR_TOP']
        frame_lidars = []
        while 1:

            lidar_sensor = nusc.get('sample_data',sample_data_token)
            timestamp=lidar_sensor['timestamp']
            lidar_cs_record = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
            lidar_pose_record = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
            extrinsic = get_R_T(lidar_cs_record['rotation'], lidar_cs_record['translation'])
            ego_pose = get_R_T(lidar_pose_record['rotation'], lidar_pose_record['translation'])
            lidar2world = ego_pose @ extrinsic
            lidar_name = "LIDAR_TOP"
            lidar_points = np.fromfile(os.path.join(args.dataroot, lidar_sensor['filename']), dtype=np.float32).reshape(-1,5)[:,:3]
            filename = lidar_sensor['filename'].split("/")[-1].split(".")[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_points)
            save_path = segment_out_dir / "lidars" / lidar_name / f"{filename}.pcd"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            o3d.io.write_point_cloud(save_path.as_posix(), pcd)
            _, lidar_boxes, _ = nusc.get_sample_data(lidar_sensor['token'])
            anns = []
            for box in lidar_boxes:
                ann = nusc.get('sample_annotation', box.token)
                b_translation = ann['translation']
                b_rotation = ann['rotation']
                b_transform_matrix = get_R_T(b_rotation, b_translation)
                if len(ann['attribute_tokens']) == 0:
                    continue
                attribute_token = ann['attribute_tokens'][0]
                if ann['num_lidar_pts'] == 0:
                    continue
                attribute = nusc.get('attribute', attribute_token)
                if attribute['name'] in target_moving_object:
                #     is_moving = True
                # else:
                #     is_moving = False
                    bbox3d = np.concatenate([box.center, box.wlh, np.array([box.orientation.yaw_pitch_roll[0]])]).astype(np.float32)
                    bbox3d[ 3:6] = bbox3d[ 3:6] * 1.1
                    # bbox_corners = rotate_box_vertices_around_z(bbox3d)
                    bbox3d[ 6] += np.pi / 2.
                    bbox3d[ 2] -= box.wlh[ 2] / 2.
                    bbox3d[ 2] = bbox3d[ 2] - 0.1
                    # 
                    anns.append(
                        {
                            "type": attribute['name'],
                            "ann_token":ann["token"] ,
                            "instance_token": ann['instance_token'],
                            "transform_matrix": b_transform_matrix.tolist(),
                            # "corners": bbox_corners.tolist(),
                            "bbox3d": bbox3d.tolist(),
                            # "is_moving": is_moving,
                        }
                    )
            sample_rec = nusc.get('sample', lidar_sensor['sample_token'])
            align = []
            for camera_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                align.append(nusc.get('sample_data',sample_rec['data'][camera_name])["filename"].split("/")[-1])
            frame_lidars.append(
                {
                    "file_path": save_path.relative_to(segment_out_dir).as_posix(),
                    "lidar": lidar_name,
                    "timestamp": timestamp,
                    "transform_matrix": lidar2world.tolist(),
                    "anns": anns,
                    "align":align,
                    "key_frame": lidar_sensor['is_key_frame'],
                }
            )
            if lidar_sensor['next'] == "":
                break
            sample_data_token = lidar_sensor['next']

        return frame_lidars
    def extract_frame_annotation(self, scene):
        first_sample_token = scene['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        
        annjson = {"frames":[]}

        for camera_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            sample_data_token = first_sample['data'][camera_name]
            while 1:
                sensor = nusc.get('sample_data', sample_data_token)
                if sensor['is_key_frame'] == True:
                    timestamp=sensor['timestamp']
                    _, boxes, _ = nusc.get_sample_data(sample_data_token)
                    objects = []
                    for box in boxes:
                        ann = nusc.get('sample_annotation', box.token)
                        if len(ann['attribute_tokens']) == 0:
                            continue
                        attribute_token = ann['attribute_tokens'][0]
                        if ann['num_lidar_pts'] == 0:
                            continue
                        attribute = nusc.get('attribute', attribute_token)
                        if attribute['name'] in target_moving_object:
                            is_moving = True
                        else:
                            is_moving = False
                        translation = ann['translation']
                        rotation = ann['rotation']
                        size = ann['size']
                        objects.append(
                            {
                                "type": attribute['name'],
                                "gid":ann["instance_token"] ,
                                "translation": translation,
                                "rotation": rotation,
                                "size": size,
                                "is_moving": is_moving,
                            }
                        )
                    annjson["frames"].append(
                        {
                            "timestamp": timestamp/1.0e6,
        
                            "objects": objects,
                        }
                    )
                if sensor['next'] == "":
                    break
                sample_data_token = sensor['next']

        return annjson

if __name__ == "__main__":
    
    nusc_data_extractor = NuscDataExtractor(1, args.out_dir)
    scenes = nusc.scene
    print(len(scenes))
    import os
    for s in scenes:
        if s["token"]=="fcbccedd61424f1b85dcbf8f897f9754":
            scenes=[s]
            break
    nusc_data_extractor.extract_one(scenes[0]["token"])
