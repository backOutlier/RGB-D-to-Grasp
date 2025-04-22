def visualize_mask_points_on_mesh(mesh_trimesh, valid_mask, color=[1, 0, 0]):
    import open3d as o3d
    import numpy as np

    # 转换为 Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
    mesh_o3d.compute_vertex_normals()

    # 初始化颜色（灰色）
    num_vertices = len(mesh_trimesh.vertices)
    colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (num_vertices, 1))

    # 把 mask 区域设为红色
    colors[valid_mask] = np.array(color)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([mesh_o3d])
