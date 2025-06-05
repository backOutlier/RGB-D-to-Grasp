from utils.loader import *
import tempfile
from config import parser
args = parser.parse_args()
from pathlib import Path


def merge_multiview_pointclouds(scene_dir, object_id, visualize_after_fusion=False, return_background=True):  
    import open3d as o3d

    npz_paths, intrinsics, extrinsics = load_intrinsics_and_poses(scene_dir)
    all_object_points = []
    all_background_points = []
    SCALE = args.scale
    for npz_path, (fx, fy, cx, cy), extrinsic in zip(npz_paths, intrinsics, extrinsics):
        object_pts, background_pts = extract_object_and_background_from_npz(
            npz_path, object_id, fx, fy, cx, cy, extrinsic
        )
        object_pts *= SCALE
        background_pts *= SCALE
        
        if object_pts.shape[0] > 0:
            all_object_points.append(object_pts)

        if return_background and background_pts.shape[0] > 0:
            all_background_points.append(background_pts)

    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_object_points, axis=0))
  
    if return_background:
        background_pcd = o3d.geometry.PointCloud()
        background_pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_background_points, axis=0))

    # if visualize_after_fusion:
    #     if return_background:
    #         object_pcd.paint_uniform_color([1, 0, 0])    
    #         background_pcd.paint_uniform_color([0.5, 0.5, 0.5])  
    #         o3d.visualization.draw_geometries([object_pcd, background_pcd],
    #                                           window_name=f"Scene View - object {object_id}")
    #     else:
    #         o3d.visualization.draw_geometries([object_pcd], window_name=f"Object PointCloud - {object_id}")

    return (object_pcd, background_pcd) if return_background else object_pcd


import open3d as o3d
import numpy as np
import os

def create_mesh_from_points(points, scene_folder, object_id, method='bpa', depth=9):
    """
    将目标点云转换为 Mesh，并保存为带 object id 和 scene 名的 obj 文件

    参数：
        - points: (N, 3) numpy array
        - scene_folder: 包含 scene 的完整路径
        - object_id: int
        - method: 'poisson' 或 'bpa'
        - depth: poisson 重建深度

    返回：
        - mesh: open3d TriangleMesh
        - mesh_path: 保存的 obj 路径
    """
    # 构造点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()

    # mesh 重建
    if method == 'poisson':
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    elif method == 'bpa':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # 生成保存路径
    scene_name = os.path.basename(scene_folder.rstrip("/"))
    filename = f"{scene_name}_object{object_id}.obj"

    # 指定目标保存路径
    save_dir = "/media/labpc2x2080ti/data/dataset/Gen_Score/Meshes"
    os.makedirs(save_dir, exist_ok=True)

    mesh_path = os.path.join(save_dir, filename)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    print(f"[✓] Mesh saved to: {mesh_path}")

    return mesh, mesh_path




