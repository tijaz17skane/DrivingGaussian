import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import shutil
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
print(os.path.abspath(os.getcwd()))
import torch

from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from nuscenes.utils.data_classes import Quaternion
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from mask_one import mask_dynamic
from arguments import ModelParams, PipelineParams, get_combined_args
from render_combined import render_sets
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# torch.backends.cudnn.enabled = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations,check_point,first_iter, state, nodes = None, token_name = None,source_path_token=None,model_path_token=None, f_name=None):
    
    gaussians = GaussianModel(dataset.sh_degree)
    
    if j == "dynamic":
        nodes[token_name]['gaussian'] = gaussians
    deform = DeformModel(dataset.is_blender)
    deform.train_setting(opt)
    print(source_path_token)
    scene = Scene(dataset, gaussians, state, source_path_token, model_path_token, f_name, token_name)
    tb_writer = prepare_output_and_logger(dataset,model_path_token,source_path_token)
    gaussians.training_setup(opt)
    if check_point:
        deform.load_weights(check_point)
        assert first_iter > 1
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(first_iter,opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        #if iteration < opt.warm_up:
        if iteration < 50:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image
        image_name = viewpoint_cam.image_name
        mask_path = os.path.join(source_path_token, image_name + ".pt")
        if os.path.exists(mask_path):
            print(mask_path)
            mask = torch.load(mask_path).cuda()

            image *= mask
            gt_image *= mask
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration,model_path_token=model_path_token)
                deform.save_weights(model_path_token, iteration)
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
    if j=="dynamic":
        nodes[token_name]['gaussians'] = gaussians
        
def prepare_output_and_logger(args, model_path_token =None,source_path_token=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    if model_path_token:
        os.makedirs(model_path_token, exist_ok=True)
        args.model_path = model_path_token
        args.source_path = source_path_token
        print("Output folder: {}".format(model_path_token))
        with open(os.path.join(model_path_token, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))
    else:
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        if model_path_token:
            tb_writer = SummaryWriter(model_path_token)
        else:
            tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr

def save_subsets(file_name, total_lines, lines_per_subset):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, total_lines,lines_per_subset):
        if i + lines_per_subset * 2 > total_lines:
            
            subset = lines[i:total_lines]
        else:
            subset = lines[i:i + lines_per_subset]        
             
        part_number = (i // lines_per_subset) + 1
        with open(f"{file_name.split('.')[0]}_part{part_number}.txt", 'a', encoding='utf-8') as subset_file:
            subset_file.writelines(subset)

def copy_files(src_dir, dst_dir):

    # Ensure destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # List all files and directories in the source directory
    for item in os.listdir(src_dir):
        if ".txt" or ".png" in item:
            continue
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        # Check if it's a file and not a directory
        if os.path.isfile(src_path):
            # Copy file
            shutil.copy2(src_path, dst_path)  # copy2 to preserve metadata if desired
            print(f"Copied {src_path} to {dst_path}")

def get_objects_from_scene(nusc, scene_name="scene-0103"):

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    def get_image_data(sample_data_token):
        """Load the image data."""
        sample_data = nusc.get('sample_data', sample_data_token)
        timestamp = sample_data['timestamp']
        img_file_path = os.path.join(nusc.dataroot, sample_data['filename'])
        img = plt.imread(img_file_path)
        return img, timestamp

    def get_pose_data(sample_data_token):
        """Load the pose data."""
        sample_data = nusc.get('sample_data', sample_data_token)
        pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        quaternion = Quaternion(pose["rotation"])
        rotation_matrix = quaternion.rotation_matrix

        transform_matrix = np.eye(4) 
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = pose["translation"]
        
        return transform_matrix

    def get_box_data(token , sample_data_token):
        box = nusc.get_box(token)
        sd_rec = nusc.get('sample_data', sample_data_token)
        s_rec = nusc.get('sample', sd_rec['sample_token'])
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
        box.translate(-np.array(pose_rec['translation']))  
        box.rotate(Quaternion(pose_rec['rotation']).inverse)
    
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        corners_3d = box.corners()
        return corners_3d

    for scene in nusc.scene:
        name = scene['name']
        if name != scene_name:
            continue
        else:
            scene_true = scene
    # scene = nusc.get('scene', nusc.scene[0]["token"])
    scene = scene_true
    first_sample_token = scene['first_sample_token']
    sample_token = first_sample_token

    imgs, poses, bboxes, instance_segm = [], [], [], []
    frame_ids, render_poses = [], []
    visible_objects, object_meta, render_objects = [], {}, []
    object_tokens = []
    camera_intrinsics = None
    track_id = 0
    while sample_token:
        sample = nusc.get('sample', sample_token)
        for cam_name in sample['data'].keys():
            if 'CAM' in cam_name:  # Ensure processing only camera data
                sample_data_token = sample['data'][cam_name]
                img = get_image_data(sample_data_token)
                pose = get_pose_data(sample_data_token)
                
                if camera_intrinsics is None:
                    sd_record = nusc.get('sample_data', sample_data_token)
                    cam = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                    camera_intrinsics = np.array([cam['camera_intrinsic']])
                
                imgs.append(img)
                poses.append(pose)
                frame_ids.append([sample['timestamp'], cam_name, 0])
                render_poses.append(pose)  # Simplification for demonstration

                # Mock-up visible objects and object meta
                for ann_token in sample['anns']:
                    ann_record = nusc.get('sample_annotation', ann_token)
                    corner = get_box_data(ann_record["token"], sample_data_token)
                    if len(ann_record['attribute_tokens']) == 0:
                        continue
                    attribute_name = nusc.get('attribute', ann_record['attribute_tokens'][0])['name']
                    if ann_record['instance_token'] not in object_meta:
                        if attribute_name == 'vehicle.moving' or attribute_name == "pedestrian.moving"  or attribute_name == 'cycle.with_rider':
                            object_meta[ann_record['instance_token']] = {
                                'category_name': ann_record['category_name'],
                                'attribute_name': attribute_name,
                                'timestamp': sample['timestamp'],
                                "instance_token": ann_record['instance_token'],
                                "pose": pose,
                                "corner": corner
                                
                            }
                            track_id += 1
                            
                            visible_objects.append(ann_record['instance_token'])
                            render_objects.append([sample['timestamp'], ann_record['instance_token'], pose, corner])
                            object_tokens.append(ann_record['instance_token'])
        sample_token = sample['next']
    
    # Assuming homogeneous camera intrinsics for simplification
    hwf = camera_intrinsics[0, :3]  # Height, Width, Focal length
    i_split = [[0], [1], [2]]  # Mock-up split for demonstration
    return imgs, poses, render_poses, hwf, i_split, visible_objects, object_meta, render_objects, object_tokens

def save_subsets_by_part_number(file_name, part_number, object_tokens, scene_name=None):
    #for j in ["images.txt", "cameras.txt"]:
    
    file_name_image = file_name + "images.txt" 

    with open(file_name_image, 'r') as file:
        content = file.read()
 
    new_content = content.replace('.jpg', '.png')
    
    with open(file_name_image, 'w') as file:
        file.write(new_content)
    print("Sorting file by timestamp")
    sort_file_by_timestamp(file_name_image, file_name_image)
    print("Sorted file by timestamp")
    with open(file_name_image, 'r', encoding='utf-8') as file:
        lines_image = [line for line in file if not line.strip().startswith('#')]
        total_lines_image = len(lines_image)
    file_name_camera = file_name + "cameras.txt"
    with open(file_name_camera, 'r', encoding='utf-8') as file:
        lines_camera = [line for line in file if not line.strip().startswith('#')]
        total_lines_camera = len(lines_camera)

    #print("Total lines in images.txt:", total_lines_image)
    #print("Total lines in cameras.txt:", total_lines_camera)
    parts = file_name.split(os.sep)
    base_path_parts = parts[:-3]  
    rest_path_parts = parts[-3:-1]  
    f_name = rest_path_parts[-1]
    base_path = os.sep.join(base_path_parts[:-1]) + os.sep 
    rest_path = os.sep.join(rest_path_parts)
    base_copy = base_path_parts[-1]
    import math
    lines_per_subset = math.ceil(total_lines_camera / part_number)
    
    #print("Lines per subset:", lines_per_subset)
    #print("Base path:", base_path)
    #print("Rest path:", rest_path)
    #print("Base copy:", base_copy)
    #print("Part number:", part_number)

    for j in ["dynamic", "static"]:
          
        if j == "static":
            print("If static HIT")  
            print("generate file")
            print(total_lines_camera)
            
            for i in range(0, total_lines_camera, lines_per_subset):
                print("Iteration over static objects")  
                end_index = min(i + lines_per_subset, total_lines_camera)
                subset_image = lines_image[2 * i : 2 * end_index]
                subset_camera = lines_camera[i : end_index]
                subset_dir_name = base_path + base_copy + "_" + j + "_part" + str((i // lines_per_subset) + 1)
                # copy_files(base_path, subset_dir_name)
                # except:
                # pass
                
                subset_image_dir_name = subset_dir_name + "/images"
                # if os.path.exists(subset_image_dir_name):
                #     shutil.rmtree(subset_image_dir_name)
                os.makedirs(subset_image_dir_name,exist_ok=True)
                os.makedirs(subset_dir_name + "/sparse",exist_ok=True)
                os.makedirs(subset_dir_name + "/sparse/origin",exist_ok=True)
                subset_file_name = subset_dir_name + os.sep + rest_path + "/images.txt"
                if os.path.exists(subset_file_name):
                    os.remove(subset_file_name)
                
                with open(subset_file_name, 'w', encoding='utf-8') as subset_file:
                    subset_file.writelines(subset_image)
                    
                subset_file_name = subset_dir_name + os.sep+ rest_path + "/cameras.txt"
                if os.path.exists(subset_file_name):
                    os.remove(subset_file_name)
                with open(subset_file_name, 'w', encoding='utf-8') as subset_file:
                    subset_file.writelines(subset_camera)
                print("Static object out")  
        elif j == "dynamic":
            #print("Else Dynamic HIT")  
            for k in object_tokens: 
                #print("Dynamic iteration HIT")  
                subset_dir_name = base_path + base_copy 
                subset_dir_name = subset_dir_name + "_" + j + "_" + k + os.sep
               
                # copy_files(base_path, subset_dir_name)
                """if os.path.exists(subset_dir_name):
                        shutil.rmtree(subset_dir_name)
                print(subset_dir_name)
                # try:
                shutil.copytree(base_path + base_copy + os.sep, subset_dir_name)"""
                if not os.path.exists(subset_dir_name + "sparse/origin"):
                    print("Creating sparse/origin directory")
                    shutil.copytree(base_path + base_copy + "/sparse/origin", subset_dir_name + "/sparse/origin")

                print("Dynamic iteration OUT")  
            
    mask_dynamic(base_path + base_copy, part_num = part_number, scene_name=scene_name)

def build_graph(visible_objects, object_meta):
    nodes = {}
    for i in range(len(visible_objects)):
        object_meta[visible_objects[i]]["track_id"] = i
        token = object_meta[visible_objects[i]]['instance_token'] 
        nodes[object_meta[visible_objects[i]]['instance_token'] ] = {}
        nodes[token]['timestamp'] = object_meta[visible_objects[i]]["timestamp"]      
        nodes[token]['pose'] = object_meta[visible_objects[i]]["pose"]
        nodes[token]['corner'] = object_meta[visible_objects[i]]['corner']  
        nodes[token]['gaussian'] = None
    return nodes

def replace_part_image(source_dir, target_base_dir):
    for i in range(1, 17):

        source_file = os.path.join(source_dir, f'image_part{i}.txt')
        
        target_dir = os.path.join(target_base_dir, f'0_part{i}', 'sparse', '0')
        
        target_file = os.path.join(target_dir, 'images.txt')
        
        if not os.path.exists(source_file):
            print(f"Source file {source_file} does not exist.")
            continue
        
        if not os.path.exists(target_dir):
            print(f"Target directory {target_dir} does not exist.")
            continue
        
        shutil.copyfile(source_file, target_file)
        print(f"Copied {source_file} to {target_file}")

def replace_dynamic_image(txt_image,base_dir):
    def get_image_filenames(images_dir):

        return [f for f in os.listdir(images_dir) if f.endswith('.png')]

    def process_images_file(txt_image,images_file, valid_filenames):

        with open(txt_image, 'r') as file:
            lines = file.readlines()

        new_lines = []
        skip_next = False
        for i in range(len(lines)):
            # Skip empty or whitespace-only lines
            if not lines[i].strip():
                continue
            filename = lines[i].strip().split()[-1]
            if filename not in valid_filenames:
                skip_next = True
                # print("skip")
            else:
                new_lines.append(lines[i])
                new_lines.append(lines[i+1])

        with open(images_file, 'w') as file:
            file.writelines(new_lines)

    for subdir in os.listdir(base_dir):
        if "dynamic_" in subdir:
            part_dir = os.path.join(base_dir, subdir)
            images_dir = os.path.join(part_dir, 'images')
            images_file = os.path.join(part_dir, 'sparse/origin/images.txt')
            
            if os.path.exists(images_dir) and os.path.exists(images_file):
                valid_filenames = get_image_filenames(images_dir)
                process_images_file(txt_image,images_file, valid_filenames)
                print(f"Processed {images_file}")
            else:
                print(f"Either {images_dir} or {images_file} does not exist in {part_dir}.")

    print("All files have been processed.")

def handle_camera(part_num=16):
    txt_camera = "/mnt/data/tijaz/anotherDataset/v1.0-mini/extracted/fcbccedd61424f1b85dcbf8f897f9754/colmap/sparse/origin/cameras.txt"
    txt_image = "/mnt/data/tijaz/anotherDataset/v1.0-mini/extracted/fcbccedd61424f1b85dcbf8f897f9754/colmap/sparse/origin/images.txt"
    # Open the file in read mode
    with open(txt_camera, 'r') as file:
        # Count the lines
        line_count = sum(1 for line in file)
    total_lines_camera = line_count
    total_lines_image = line_count * 2
    lines_per_subset = int(line_count/part_num) +1
    # save_subsets(txt_image, total_lines_image, lines_per_subset*2)
    dir_name = os.path.join(txt_image.split('.')[0][:-7], str(lines_per_subset))
    # replace_part_image(dir_name,txt_camera.rsplit('/', 4)[0])
    replace_dynamic_image(txt_image,txt_camera.rsplit('/', 4)[0])

def sort_file_by_timestamp(filename,filename2):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    paired_lines = [(lines[i], lines[i+1]) for i in range(4, len(lines), 2)]
    
    paired_lines.sort(key=lambda x: int(x[0].split('__')[-1].split('.')[0]))
    
    with open(filename2, 'w') as file:
        for pair in paired_lines:
            file.write(pair[0])
            file.write(pair[1])

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6011)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5_000, 6_000, 7_000] + list(range(10_000, 40_000, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 20_000, 30_000, 50_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--first_iter", type=int, default=1)
    parser.add_argument("--part_num", type=int, default=16)
    parser.add_argument("--lidar", type= int, default = 3)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    model_path_copy = args.model_path
    source_path_copy = args.source_path

    nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/data/tijaz/anotherDataset/v1.0-mini', verbose=True)
    
    # Print all available scene names for debugging
    print("Available scene names:")
    for scene in nusc.scene:
        print(scene['name'])
    # Select the first scene name (or change as needed)
    selected_scene_name = nusc.scene[0]['name'] if len(nusc.scene) > 0 else None
    # Always use scene-0103
    selected_scene_name = "scene-0103"
    print(f"Using scene: {selected_scene_name}")
    imgs, poses, render_poses, hwf, i_split, visible_objects, object_meta, render_objects, object_tokens = get_objects_from_scene(nusc, scene_name=selected_scene_name)
    nodes = build_graph(visible_objects, object_meta)
    print("Getting INTO save_subsets_by_part_number")
    save_subsets_by_part_number(args.source_path + "colmap/sparse/origin/", args.part_num, object_tokens, scene_name=selected_scene_name)
    print("Getting OUTOF  save_subsets_by_part_number")
    handle_camera(args.part_num)
    print("Object tokens:", object_tokens)
    print("Training Started.... ")
    f_name = "1"
    for j in ["static","dynamic"]:
    
        source_path_state = os.path.join(source_path_copy, f"colmap_{j}")
        print("source_path:", source_path_state)
        # source_path_state = source_path_copy 
        model_path_state = model_path_copy + "_" + j
        
        print("model_path:", model_path_state)
        # mask_dynamic(args.source_path)
        if j == "dynamic":
            print("Dynamic objects found:", object_tokens)
            for k in range(len(object_tokens)):
                token_name = object_tokens[k]
                source_path_token = source_path_state + "_" + token_name

                model_path_token = model_path_state + "_" + token_name 
                print("model_path_token:", model_path_token)
                if not os.path.exists(source_path_token):
                    continue
                print("source_path:", source_path_token) 
                print("Optimizing " + model_path_token)
                # Initialize system state (RNG)
                safe_state(args.quiet)
                
                args.iterations = 50000
                
                # Start GUI server, configure and run training
                # network_gui.init(args.ip, args.port)
                torch.autograd.set_detect_anomaly(args.detect_anomaly)
                training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,args.start_checkpoint,args.first_iter, state = j, nodes = nodes, token_name = token_name,source_path_token = source_path_token, model_path_token = model_path_token)
        else:
            for i in range(1, 5):
                source_path_token = source_path_state + "_part" + str(i)

                model_path_token = model_path_state + "_part" + str(i)
            
                print("Optimizing " + source_path_token)
                # Initialize system state (RNG)
                safe_state(args.quiet)

                # Start GUI server, configure and run training
                # network_gui.init(args.ip, args.port)
                torch.autograd.set_detect_anomaly(args.detect_anomaly)
                if i > 0:
                    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,args.start_checkpoint,args.first_iter, state = j,source_path_token = source_path_token, model_path_token = model_path_token)
                if i > 1:
                    f_name =  f_name + "_" + str(i)
                    source_path_global = source_path_state + "_part" + f_name
                    model_path_global = model_path_state + "_part" + f_name
                    #os.makedirs(source_path_global)
                    if os.path.exists(source_path_global):
                        shutil.rmtree(source_path_global)
                    #shutil.copytree(args.source_path + "_static", source_path_global)
                    shutil.copytree(args.source_path, source_path_global)
                    if i > 10:
                        file_path = source_path_state + "_part" + f_name[:-3] +"/sparse/origin/images.txt"
                    else:
                        file_path = source_path_state + "_part" + f_name[:-2] +"/sparse/origin/images.txt"

                    with open(source_path_token+"/sparse/origin/images.txt", 'r') as file1:
                        content1 = file1.read()
                    
                    with open(file_path, 'r') as file2:
                        content2 = file2.read()
                    
                    combined_content = content1 + '\n' + content2
                    
                    with open(source_path_global+"/sparse/origin/images.txt", 'w') as file_combined:
                        file_combined.write(combined_content)
                    torch.autograd.set_detect_anomaly(args.detect_anomaly)
                    #if i != 2:
                    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,args.start_checkpoint,args.first_iter, state = j,source_path_token = source_path_global, model_path_token = model_path_global, f_name = f_name)
            # All done
            print("\nTraining complete.")
            
    parsers = []  
    models = []      
    for i in range(len(object_tokens) + 1):
        parser = ArgumentParser(description="Testing script parameters")
        parsers.append(parser)
        
        model = ModelParams(parser,sentinel=True)
        models.append(model)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--rotation", default=False)
    args1 = get_combined_args(parser)
    safe_state(args1.quiet)
    model_extract = []
    for i in range(len(models)):
        model_extract.append(models[i].extract(args1))
    render_sets(model_extract, object_tokens, args1.iteration, pipeline.extract(args1), args1.skip_train, args1.skip_test, args1.mode)
