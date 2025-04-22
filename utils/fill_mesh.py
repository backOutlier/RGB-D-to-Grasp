
import open3d as o3d
import numpy as np

def sample_surface_and_inner_centers(mesh_o3d, voxel_size=0.005, 
                                     num_surface_points=1000, 
                                     max_inner_points=None):
    """
    从 mesh 中采样 grasp center，包括表面点和体素内部点。
    
    Args:
        mesh_o3d (o3d.geometry.TriangleMesh): Open3D 格式 mesh
        voxel_size (float): 内部体素大小
        num_surface_points (int): 采样的表面点数量
        max_inner_points (int or None): 返回的内部点上限（None 表示全部）

    Returns:
        np.ndarray: [N, 3] 的 grasp center 点（包含表面和内部）
    """
    # 表面点
    pcd_surface = mesh_o3d.sample_points_uniformly(number_of_points=num_surface_points)
    surface_points = np.asarray(pcd_surface.points)

    # 体内点
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_o3d, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    inner_points = np.array([
        voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels
    ])
    if max_inner_points is not None and len(inner_points) > max_inner_points:
        choice = np.random.choice(len(inner_points), max_inner_points, replace=False)
        inner_points = inner_points[choice]

    all_centers = np.concatenate([surface_points, inner_points], axis=0)
    return all_centers

def trimesh_to_open3d(tri_mesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    return mesh_o3d