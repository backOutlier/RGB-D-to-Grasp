import open3d as o3d
import numpy as np
from config import parser
args = parser.parse_args()


def sample_points_on_pointcloud(
    pcd: o3d.geometry.PointCloud,
    surface_sample_num=None,
    voxel_size=None,
    normal_filter=False,
    normal_z_thresh=args.normal_z_thresh,
    return_normals=False,
    visualize=False,
    inward=True
):
    import numpy as np
    import open3d as o3d
    
    # Step 1: Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd_down.orient_normals_consistent_tangent_plane(k=20)

    if inward:               # <—— 新增参数
        center = pcd_down.get_center()
        pcd_down.orient_normals_towards_camera_location(center)

    # Step 2: Convert to numpy
    all_points = np.asarray(pcd_down.points)
    all_normals = np.asarray(pcd_down.normals)

    # Step 3: Normal filter
    if normal_filter:
        mask = all_normals[:, 2] > normal_z_thresh
    else:
        mask = np.ones(len(all_points), dtype=bool)

    filtered_points = all_points[mask]
    filtered_normals = all_normals[mask]
    rejected_points = all_points[~mask]
    rejected_normals = all_normals[~mask]

    # Step 4: Optional resample to fixed count
    if len(filtered_points) > surface_sample_num:
        idx = np.random.choice(len(filtered_points), surface_sample_num, replace=False)
        filtered_points = filtered_points[idx]
        filtered_normals = filtered_normals[idx]

    # Step 5: Visualization if needed
    if visualize:
        vis_objects = []

        # Kept points (green)
        pcd_kept = o3d.geometry.PointCloud()
        pcd_kept.points = o3d.utility.Vector3dVector(filtered_points)
        pcd_kept.normals = o3d.utility.Vector3dVector(filtered_normals)  
        pcd_kept.paint_uniform_color([0, 1, 0])
        vis_objects.append(pcd_kept)

        # Rejected points (red)
        if len(rejected_points) > 0:
            pcd_rejected = o3d.geometry.PointCloud()
            pcd_rejected.points = o3d.utility.Vector3dVector(rejected_points)
            pcd_rejected.normals = o3d.utility.Vector3dVector(rejected_normals)
            pcd_rejected.paint_uniform_color([1, 0, 0])
            vis_objects.append(pcd_rejected)

        o3d.visualization.draw_geometries(vis_objects, point_show_normal=True)

    print(f"[DEBUG] Sampled {len(filtered_points)} valid points (kept), {len(rejected_points)} rejected.")

    if return_normals:
        return filtered_points, filtered_normals
    else:
        return filtered_points

    
# mesh = o3d.io.read_triangle_mesh("/media/labpc2x2080ti/data/dataset/graspnet/models/000/textured.obj")
# points = sample_points_on_mesh(mesh, surface_sample_num=1000)
