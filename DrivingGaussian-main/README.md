# DrivingGaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes

## Cloning the Repo

This repository contains the implementation associated with the paper "DrivingGaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes". Please be advised that the use of this code is strictly limited to non-commercial purposes.

In compliance with licensing agreements and open-source constraints, certain codes and modules within this repo have been modified or removed, e.g., changing the incremental static background from depth-based to time-based, 4D gaussian model to deform gaussian model, etc. These could lead to performance decreases and issues. If you have any further questions, please feel free to send your questions to [xy56.zhou@gmail.com].

The complete code will be released in our future work.

We welcome all kinds of exchanges and discussions that will help us improve and refine this project. Your insights and feedback are invaluable as we strive for excellence and clarity in our work.

## Dataset

In our paper, we use autonomous driving dataset from [Nuscenes](https://www.nuscenes.org/nuscenes) and [KITTI360](https://www.cvlibs.net/datasets/kitti-360/). 

## Run

### Environment

```shell
conda env create --file environment.yml
conda activate drivinggaussian
pip install imageio==2.28.0
pip install opencv-python
pip install imageio-ffmpeg
```

### Dataset Preprocess

Using nuScenes as an example:

1. Loading scene data including images, LiDAR, and annotations.

2. Decoupling static and dynamic parts; this process will be carried out in the background_extraction, dynamic_object_extraction, and train.

3. Convert dataset to COLMAP format using extract_nuscene.py and extract_object_pts.py

```shell
python extract_nuscene.py
python transform.py
python align.py
python extract_object_pts.py
```
4. kNN can be used for downsampling point clouds.

5. COLMAP can also be used to generate an initial point cloud.

### Train

```shell
python train_combine.py -s source_path -m outputs_path
```

### Render

```shell
python render_combine.py -m outputs_path
```

## Acknowledgements
The overall code and renderer are based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Deformable](https://github.com/ingra14m/Deformable-3D-Gaussians), and [4DGS](https://github.com/hustvl/4DGaussians). We sincerely thank the authors for their great work.

## BibTex

```
@inproceedings{zhou2024drivinggaussian,
  title={Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes},
  author={Zhou, Xiaoyu and Lin, Zhiwei and Shan, Xiaojun and Wang, Yongtao and Sun, Deqing and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21634--21643},
  year={2024}
}
```
