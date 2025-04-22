import open3d as o3d
import numpy as np

def sample_points_on_mesh(mesh, num_points=1000, visualize=True):
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    if visualize:
        pcd.paint_uniform_color([1, 0, 0])
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        o3d.visualization.draw_geometries([mesh, pcd])

    return np.asarray(pcd.points)

# points = sample_points_on_mesh("/media/labpc2x2080ti/data/dataset/mesh_bpa.obj", num_points=2000)
