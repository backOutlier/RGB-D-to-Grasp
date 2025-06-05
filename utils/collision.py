import open3d as o3d
import numpy as np

def get_grasp_transform(p1,p2,view, angle, base_point, depth):
    """
    Given a grasp direction (view), in-plane rotation (angle),
    a base surface point, and forward depth, compute the grasp
    center and local coordinate frame (rotation matrix).

    Returns:
    - center: grasp center = base_point + depth * view
    - rotation: 3×3 matrix = [x_axis (width), y_axis (finger), z_axis (approach)]
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    z_axis = view / (np.linalg.norm(view) + 1e-9)  # grasp approach
    up = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.95 else np.array([1, 0, 0])
    
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis) #width direction
    y_axis = np.cross(z_axis, x_axis)
    
    R_base = np.stack([x_axis, y_axis, z_axis], axis=1)
    R_inplane = R.from_euler('z', angle, degrees=True).as_matrix()
    rotation = R_base @ R_inplane
    
    # x1 = np.dot(p1, x_axis)
    # x2 = np.dot(p2, x_axis)
    # x_mid = 0.5 * (x1 + x2)
    
    # center_init = base_point - depth * z_axis
    # center_proj=np.dot(center_init,x_axis)
    # delta = x_mid-center_proj
    # center=center_init+delta*x_axis
    # offset=center - base_point
    # return center, rotation, offset
        # ✅ 改动开始：使用接触点中点 + depth 方向位移作为 grasp center
    center_contact = 0.5 * (p1 + p2)
    center = center_contact - depth * z_axis
    # ✅ 改动结束

    offset = center - base_point
    return center, rotation, offset




def debug_grasp_box_position(grasp_box, ref_point=None, object_pcd=None, show_axes=True,p1=None, p2=None):
    """
    可视化 grasp 框（红）、参考点（蓝）、目标点云（绿），抓取框姿态使用其 rotation。
    - grasp_box: OrientedBoundingBox 或 TriangleMesh（需要 .R 属性）
    - ref_point: np.array, 抓取开口参考点（蓝球）
    - object_pcd: 目标点云（绿色）
    - show_axes: 是否显示 grasp 坐标系（对齐 grasp_box.R）
    """
    import open3d as o3d
    import numpy as np

    geometries = []

    # 添加抓取框
    if isinstance(grasp_box, o3d.geometry.OrientedBoundingBox):
        grasp_box.color = (1, 0, 0)
        geometries.append(grasp_box)
    elif isinstance(grasp_box, o3d.geometry.TriangleMesh):
        grasp_box.paint_uniform_color([1, 0, 0])
        geometries.append(grasp_box)

    # 添加参考点（蓝球）
    if ref_point is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere.translate(np.asarray(ref_point))
        sphere.paint_uniform_color([0, 0, 1])
        geometries.append(sphere)

    # 添加物体点云（绿）
    if object_pcd is not None:
        if isinstance(object_pcd, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(object_pcd)
            object_pcd = pcd
        object_pcd.paint_uniform_color([0.1, 0.8, 0.1])
        geometries.append(object_pcd)

    # 添加 grasp 坐标轴（方向对齐 grasp_box.R）
    if show_axes and hasattr(grasp_box, "center") and hasattr(grasp_box, "R"):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        axis.rotate(grasp_box.R, center=(0, 0, 0))         # ⬅️ 关键：应用姿态
        axis.translate(grasp_box.center)
        geometries.append(axis)
        # 添加接触点 p1（紫色）
    if p1 is not None:
        sphere_p1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        sphere_p1.translate(np.asarray(p1))
        sphere_p1.paint_uniform_color([0.6, 0.2, 1.0])  # 紫色
        geometries.append(sphere_p1)

    # 添加接触点 p2（青色）
    if p2 is not None:
        sphere_p2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        sphere_p2.translate(np.asarray(p2))
        sphere_p2.paint_uniform_color([0.2, 1.0, 1.0])  # 青色
        geometries.append(sphere_p2)

    # 可选：用线连起来
    if p1 is not None and p2 is not None:
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([p1, p2]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # 黄色线
        geometries.append(line)

    o3d.visualization.draw_geometries(geometries)





def build_grasp_box(center, rotation, width, height, thickness):
    """
    Construct an OrientedBoundingBox for a robotic grasp:
    - width: the gripper's opening width, aligned with the 2nd column of the rotation matrix (R[:, 1])
    - height: the gripper's forward approach length, aligned with the 1st column of the rotation matrix (R[:, 0])
    - thickness: the gripper's finger thickness, aligned with the 3rd column of the rotation matrix (R[:, 2])
    """
    forward_dir = rotation[:, 2]

    correction = forward_dir * (height / 2)#correct the right location of the center
    extent = np.array([width, thickness,height]) 
    box_center = center+ correction
    obb = o3d.geometry.OrientedBoundingBox(center=box_center, R=rotation, extent=extent)
    # print("Center:", box_center)
    return obb

def check_grasp_box_collision_with_background(obb, background_points):
    """
    Judege if the grasp box collides with the background points
    :param obb: the grasp box
    :param background_points: the background points
    :return: True if the grasp box collides with the background points, False otherwise
    """
    background_points = np.asarray(background_points, dtype=np.float64)
    if background_points.ndim != 2 or background_points.shape[1] != 3:
        raise ValueError(f"Invalid background_points shape: {background_points.shape}, expected (N, 3)")
    indices = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(background_points))
    return len(indices) > 0