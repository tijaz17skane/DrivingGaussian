#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch
import open3d as o3d
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, state=None,  source_path_token=None,model_path_token = None,  f_name = None, token_name = None, args1= None, load_iteration=None, shuffle=True,other_path = False, path_name = None,
                 resolution_scales=[1.0], number= 0):
        flag = 0
        if other_path:
            self.model_path = "" # custom path for dataset
            flag = 1
        elif model_path_token:
            self.model_path = model_path_token
        else:
            self.model_path = args.model_path

        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        import pdb
        if source_path_token:
            scene_info = sceneLoadTypeCallbacks["Colmap"](source_path_token, args.images, args.eval)
        else:
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
                print("Found cameras_sphere.npz file, assuming DTU data set!")
                scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
            elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
                print("Found dataset.json file, assuming Nerfies data set!")
                scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
                print("Found calibration_full.json, assuming Neu3D data set!")
                scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 200)
            elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
                print("Found calibration_full.json, assuming Dynamic-360 data set!")
                scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
            else:
                assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            
            import pdb
            os.makedirs(self.model_path, exist_ok=True)
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        if number == 0:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                                args,args1 =args1)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                            args,args1 =args1)   
            
        if self.loaded_iter:
            
            if f_name:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
            
            elif flag == 1:
                load_path = "" # custom path for 3dgs
                self.gaussians.load_ply(load_path)
                print(load_path)
            
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))  
 
        elif f_name:
            children = []
            gaussians1 = GaussianModel(3)
            
            if len(f_name) > 10:
                file_path1 = model_path_token[:-3] + "/point_cloud.ply"
                file_path2 = model_path_token[:-5]+ model_path_token[-2:] + "point_cloud.ply"
            else:
                print(model_path_token + "\n" + str(f_name[:-2]) + "\n" + str(f_name[-1]) + "\n" + str(f_name))
                file_path1 = model_path_token[:-2] + "point_cloud.ply"
                file_path2 = "/output" + model_path_token[-1] + "point_cloud.ply"
                print(file_path1 + "\n")
                print(file_path2 + "\n")
            gaussians1.load_ply(file_path1) 
            gaussians2 = GaussianModel(3)
            gaussians2.load_ply(file_path2)
            children.append(gaussians1)   
            children.append(gaussians2) 
            self.gaussians._xyz = torch.cat([chi._xyz for chi in children], dim=0)
            self.gaussians._features_dc = torch.cat([chi._features_dc for chi in children],
                                            dim=0)
            self.gaussians._features_rest = torch.cat(
                    [chi._features_rest for chi in children], dim=0)
            self.gaussians._scaling = torch.cat([chi._scaling for chi in children], dim=0)
            self.gaussians._rotation = torch.cat([chi._rotation for chi in children], dim=0)
            self.gaussians._opacity = torch.cat([chi._opacity for chi in children], dim=0)
            self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
        elif state and token_name:
            self.gaussians.create_from_pcd(args, scene_info.point_cloud, self.cameras_extent)
        else:
            self.gaussians.load_ply(os.path.join(source_path_token,"sparse/origin/points3D.ply"))

        self.point_cloud = scene_info.point_cloud
        
    def save(self, iteration,model_path_token=None):
        if model_path_token:
            self.model_path = model_path_token
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getPointCloud(self):
        return self.point_cloud
