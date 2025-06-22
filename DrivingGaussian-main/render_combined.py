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

from scipy.spatial.transform import Rotation
import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np

def render_set(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpt_on_the_fly:
            view.load2gpu()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach().cuda(device=6), time_input.cuda(device=6))
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def interpolate_time(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'frames.mp4'), renderings, fps=30, quality=8)

def interpolate_all(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    frame = 1300
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'frames.mp4'), renderings, fps=30, quality=8)

def render_sets(dataset: ModelParams, object_tokens, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    other_path = True
    with torch.no_grad():
        children = []
        all_scenes = []
        for i in range(len(object_tokens)):    
            gaussians = GaussianModel(3)
            children.append(gaussians)
            scene = Scene(dataset[i], gaussians, load_iteration=iteration, shuffle=False ,other_path=other_path, path_name = object_tokens[i], number = i)
            all_scenes.append(scene)
            other_path = True
        dataset = dataset[0]
        gaussians._xyz = torch.cat([chi._xyz for chi in children], dim=0)
        gaussians._features_dc = torch.cat([chi._features_dc for chi in children],
                                           dim=0)
        gaussians._features_rest = torch.cat(
                [chi._features_rest for chi in children], dim=0)
        gaussians._scaling = torch.cat([chi._scaling for chi in children], dim=0)
        gaussians._rotation = torch.cat([chi._rotation for chi in children], dim=0)
        gaussians._opacity = torch.cat([chi._opacity for chi in children], dim=0)
        gaussians.save_ply("./combined_opacity.ply")
        
        deform = DeformModel(dataset.is_blender)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--rotation", default=False)
    args = get_combined_args(parser)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test ,args.mode)
