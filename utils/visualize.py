import open3d as o3d
import numpy as np


def create_pi_shaped_gripper(p1, p2, lift_axis=np.array([0, 0, 1]), finger_length=1, color=[0, 1, 0]):
    """
    创建一个统一颜色的 Π 型夹抓（单个 grasp）
    """
    p1, p2 = np.array(p1), np.array(p2)
    lift_axis = lift_axis / np.linalg.norm(lift_axis)

    p1_tip = p1 + lift_axis * finger_length
    p2_tip = p2 + lift_axis * finger_length

    points = [p1, p1_tip, p2, p2_tip]
    lines = [[0, 1], [2, 3], [1, 3]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return line_set

def visualize_pi_grippers(mesh_o3d, grasps, background_points_world=None, finger_length=0.04):
    mesh = mesh_o3d
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    gripper_lines = []

    for grasp in grasps:
        p1 = np.array(grasp['p1'])
        p2 = np.array(grasp['p2'])
        view = np.array(grasp['view'])
        
        if np.linalg.norm(view) < 1e-6:
            continue
        view = view / np.linalg.norm(view)

        if np.linalg.norm(p1 - p2) < 1e-5:
            continue

        fc = grasp.get('fc', False)
        collision = grasp.get('collides_with_background', True)

        if fc and not collision:
            color = [0, 1, 0]  # 有效
        elif fc and collision:
            color = [0, 0, 1]  # 半有效
        else:
            color = [1, 0, 0]  # 无效

        gripper = create_pi_shaped_gripper(
            p1=p1, p2=p2, lift_axis=view, finger_length=finger_length, color=color
        )
        gripper_lines.append(gripper)

    geometries = [mesh] + gripper_lines

    # === 如果传了背景点云，就加进去
    if background_points_world is not None:
        background_pcd = o3d.geometry.PointCloud()
        background_pcd.points = o3d.utility.Vector3dVector(background_points_world)
        background_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色
        geometries.append(background_pcd)

    o3d.visualization.draw_geometries(geometries)



# # === 给定 p1, p2（你可以自己替换下面的坐标） ===
# p1 = np.array([0.0, 0.0, 0.0])
# p2 = np.array([0.05, 0.0, 0.0])  # 夹爪宽度为 5cm
# lift_axis = np.array([0, 0, 3])  # 向上提
# finger_length = 0.06  # 指长 6cm

# # === 创建夹抓并可视化 ===
# gripper = create_pi_shaped_gripper(p1, p2, lift_axis=lift_axis, finger_length=finger_length, color=[0, 1, 0])

# # 可视化
# o3d.visualization.draw_geometries([gripper])

