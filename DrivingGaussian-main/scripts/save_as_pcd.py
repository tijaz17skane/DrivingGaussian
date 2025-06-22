import open3d as o3d
import numpy as np

def save_as_pcd(filtered_points,filtered_labels,colors,filename):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    color = np.array(colors[filtered_labels])
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(filename, pcd)

def save_as_pcd_rgb(filtered_points,filtered_labels,filename):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    pcd.colors = o3d.utility.Vector3dVector(filtered_labels)  # Normalize to [0, 1]

    # Save the PointCloud to a PCD file
    o3d.io.write_point_cloud(filename, pcd)
