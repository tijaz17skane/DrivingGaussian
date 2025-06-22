import collections
import json
import os
import shutil
import random
from scipy.spatial.transform import Rotation
import numpy as np

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
from pyquaternion import Quaternion
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
        
if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/Nuscenes/fcbccedd61424f1b85dcbf8f897f9754/")
    parser.add_argument("--json_file", default="transform.json")
    args = parser.parse_args()

    input_path = args.input_path
    input_json = input_path + "/" + args.json_file
    point3D_txt = input_path + "/colmap/sparse/origin/points3D.txt"
    cameras_txt = input_path + "/colmap/sparse/origin/cameras.txt"
    images_txt = input_path + "/colmap/sparse/origin/images.txt"
    folder_path = input_path + "/colmap/sparse/origin"
    result_path = input_path + "/colmap/sparse/result"
    
    if not os.path.exists(result_path):   
        os.makedirs(result_path)
        
    if not os.path.exists(folder_path):    
        os.makedirs(folder_path)
    if os.path.exists(point3D_txt):  
        print("Existing")  
    else:  
 
        with open(point3D_txt, 'w') as file:  
            pass 
        print("Creating")
    
    with open(input_json, 'r') as file:
        data = json.load(file)

    frames = data['frames']
    transform = np.array(data["transform"])

    ## cameras id
    cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    cams_map = {cam: i + 1 for i, cam in enumerate(cams)}

    cameras_content={}
    images_content={}
    id= 0
    for obj in frames:  
        id +=1
        c2w = np.array(obj["transform_matrix"])
        c2w = transform @ c2w
        #c2w = c2w@np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        #T0=np.array([random.random()*0.05,random.random()*0.05,random.random()*0.05])
        # c2w[:3,3]-=T0

        # c2w[2, :] *= -1
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # c2w[0:3, 1:3] *= -1

        c2w = np.linalg.inv(c2w)

        quaternion = rotmat2qvec(c2w[:3,:3])
        
        cam_index = cams_map[obj["camera"]]
        T=(c2w[:3,3]).tolist()
        # print(f"T (x, y, z): {T}")
        filename = obj["file_path"]
        # copy imges
        file_path = input_path + filename
        
        slash_index = filename.find('/')

        if slash_index != -1: 
            result_string = filename[slash_index + 1:]
        else:
            result_string = filename

        # print(result_string)
        value = "images_colmap"
        
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

        image_paras = [quaternion[0],quaternion[1],quaternion[2],quaternion[3],T[0],T[1],T[2],cam_index,result_string]
        image_paras = [str(i) for i in image_paras]  
        
        images_content[id] = image_paras
        
        camera_model =obj["camera_model"]
        
        w= obj["w"]
        h= obj["h"]
        fx= obj["fl_x"]
        fy= obj["fl_y"]
        cx = obj["cx"]
        cy = obj["cy"]
        # k1 = obj["k1"]
        # k2 = obj["k2"]

        if(cam_index in cameras_content):
            continue
        else:
            paras=[]
            if(camera_model == "OPENCV_FISHEYE"):
                k3 = obj["k3"]
                k4 = obj["k4"]
                paras=[camera_model,w,h,fx,fy,cx,cy,k1,k2,k3,k4]
            elif(camera_model == "OPENCV"):
                p1 = obj["p1"]
                p2 = obj["p2"]   
                paras=[camera_model,w,h,fx,fy,cx,cy,k1,k2,p1,p2]
            else:
                paras=[camera_model,w,h,fx,fy,cx,cy]
            paras =[str(i) for i in paras]  
            cameras_content[cam_index] = paras


    with open(cameras_txt, 'w') as f:  
        for cam_index in cameras_content:
            f.write(str(cam_index)+" ")
            f.write(' '.join(cameras_content[cam_index]) + '\n')
            
    with open(images_txt, 'w') as f:  
        for image_index in images_content:
            f.write(str(image_index)+" ")
            f.write(' '.join(images_content[image_index]) + '\n\n')
