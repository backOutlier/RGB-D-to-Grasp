import open3d as o3d
import numpy as np
import random
from utils.collision import get_grasp_transform, build_grasp_box
from config import parser
args = parser.parse_args()
from scipy.spatial.transform import Rotation as R

def visualize_mask_points_with_grasp_frame(object_pcd, sub_points, center, view, angle):
    """
    可视化子点集（sub_points）与抓取姿态坐标系。

    Args:
        object_pcd: o3d.geometry.PointCloud，原始完整点云
        sub_points: (M, 3) ndarray，有效抓取区域的点集（例如 mask 后筛选得到的）
        center: (3,) ndarray，抓取中心点
        view: (3,) ndarray，抓取方向（approach vector）
        angle: float，夹爪在局部平面内的旋转角度（单位：度）
    """
    if sub_points is None or len(sub_points) < 2:
        print("无足够 sub_points，跳过可视化")
        return

    # 子点云
    sub_pcd = o3d.geometry.PointCloud()
    sub_pcd.points = o3d.utility.Vector3dVector(sub_points)
    sub_pcd.paint_uniform_color([1, 0, 0])  # 红色显示

    # 姿态坐标系（抓取方向）
    view_n = view / (np.linalg.norm(view) + 1e-9)
    up = np.array([0, 0, 1]) if abs(view_n[2]) < 0.95 else np.array([1, 0, 0])
    x_axis = np.cross(up, view_n); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(view_n, x_axis)
    R_base = np.stack([x_axis, y_axis, view_n], axis=1)
    R_inplane = R.from_euler("z", angle, degrees=True).as_matrix()
    R_final = R_base @ R_inplane

    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    grasp_frame.translate(center)
    grasp_frame.rotate(R_final, center=center)

    o3d.visualization.draw_geometries([object_pcd, sub_pcd, grasp_frame])




def create_graspnet_style_lineset(center, rotation, width,
                                   finger_length=0.06, arm_extension=0.02):
    """
    仿 GraspNet API 的抓取姿态线框生成方式
    """
    import numpy as np
    import open3d as o3d

    open_dir = rotation[:, 0]      # 张开方向
    thick_dir = rotation[:, 1]     # 厚度方向
    approach = rotation[:, 2]      # 前进方向

    half_width = width / 2

    # 左右两指尖中心
    p1 = center - half_width * open_dir
    p2 = center + half_width * open_dir

    # 向前延伸 finger（approach 方向）
    p1_tip = p1 + finger_length * approach
    p2_tip = p2 + finger_length * approach

    # 后端闭合和延伸
    back_center = 0.5 * (p1_tip + p2_tip)
    arm_tip = back_center + arm_extension * approach

    points = [p1, p2, p1_tip, p2_tip, back_center, arm_tip]
    lines = [
        [0, 2], [1, 3],    # 两指
        [2, 3],            # 前端闭合
        [4, 5],            # 后臂延伸
    ]
    colors = [[0, 0, 0] for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set



def visualize_random_grasps_graspnet_style(sub_candidates, object_pcd=None, num_each=5, background_points_world=None):
    """
    从 sub_candidates 中各随机抽取 num_each 个有效和无效抓取进行可视化。
    - 有效抓取: fc=True 且 collide=False → 绿色
    - 无效抓取: fc=False 或 collide=True → 红色
    """
    import open3d as o3d
    import random

    # 分类
    # valid = [g for g in sub_candidates if g.get("fc", False) and not g.get("collide", True)]
    valid = [g for g in sub_candidates if g.get("fc", False) and not g.get("collide", True) and abs(g.get("depth", -1) - 0.01) < 1e-6]

    invalid = [g for g in sub_candidates if not g.get("fc", False) or g.get("collide", True)]

    print(f"[INFO] valid number: {len(valid)}, invalid number: {len(invalid)}")

    # 随机抽样
    valid_sample = random.sample(valid, min(num_each, len(valid)))

    invalid_sample = random.sample(invalid, min(num_each, len(invalid)))
    print("[INFO] Valid sample scores:")
    for i, g in enumerate(valid_sample):
        print(f"  #{i+1}: score = {g.get('score', 'N/A')}")
    geometries = []

    # 背景物体（点云或 mesh）
    if object_pcd is not None:
        object_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(object_pcd)

    # 显示有效抓取（绿色）
    for g in valid_sample:
        grasp = create_graspnet_style_lineset(g["gripper_center"], g["rotation"], g["width"])
        grasp.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(grasp)

    # 显示无效抓取（红色）
    for g in invalid_sample:
        grasp = create_graspnet_style_lineset(g["gripper_center"], g["rotation"], g["width"])
        grasp.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(grasp)

    if background_points_world is not None:
        if isinstance(background_points_world, np.ndarray):
            bg_pcd = o3d.geometry.PointCloud()
            bg_pcd.points = o3d.utility.Vector3dVector(background_points_world)
            bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(bg_pcd)
        elif isinstance(background_points_world, o3d.geometry.PointCloud):
            background_points_world.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(background_points_world)
    # 显示
    o3d.visualization.draw_geometries(geometries)




def scale_grasp_scene(object_pcd, background_pcd=None, scale=1/100.0):
    """
    将点云从厘米缩放到米（或其他单位），默认 cm → m。
    - object_pcd: open3d.geometry.PointCloud
    - background_pcd: 可选背景点云
    - scale: 缩放因子，默认 1/100
    """
    import open3d as o3d

    assert isinstance(object_pcd, o3d.geometry.PointCloud), "object_pcd 必须是 Open3D 点云对象"
    object_pcd.scale(scale, center=(0, 0, 0))
    print(f"[INFO] Scaled object_pcd by factor {scale}")

    if background_pcd is not None:
        assert isinstance(background_pcd, o3d.geometry.PointCloud), "background_pcd 必须是 Open3D 点云对象"
        background_pcd.scale(scale, center=(0, 0, 0))
        print(f"[INFO] Scaled background_pcd by factor {scale}")

    return scale



def visualize_surface_and_invalid(surface_points, object_pcd, valid_mask,
                                  grasp_center=None, grasp_box=None,p1=None, p2=None):
    """
    可视化 grasp 有效区域（红）和无效区域（灰）以及 grasp 框/中心。

    参数：
    - surface_points: 被 valid_mask 选中的点（Nx3 numpy）
    - object_pcd: 原始 Open3D 点云
    - valid_mask: 与 object_pcd 一一对应的布尔掩码
    - grasp_center: 可选 grasp 中心点（蓝球）
    - grasp_box: 可选 grasp box（红框）
    """
    geometries = []

    # 1. surface points: 红色
    surf_pcd = o3d.geometry.PointCloud()
    surf_pcd.points = o3d.utility.Vector3dVector(surface_points)
    surf_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    geometries.append(surf_pcd)

    # 2. invalid points: 灰色
    object_np = np.asarray(object_pcd.points)
    invalid_points = object_np[~valid_mask]
    if len(invalid_points) > 0:
        invalid_pcd = o3d.geometry.PointCloud()
        invalid_pcd.points = o3d.utility.Vector3dVector(invalid_points)
        invalid_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色
        geometries.append(invalid_pcd)

    # 3. grasp_box
    if grasp_box is not None:
        grasp_box.color = (1.0, 0.0, 0.0)
        geometries.append(grasp_box)

    # 4. grasp center
    if grasp_center is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere.translate(np.asarray(grasp_center))
        sphere.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
        geometries.append(sphere)
        # 5. 接触点 p1（紫色）
    if p1 is not None:
        s1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        s1.translate(np.asarray(p1))
        s1.paint_uniform_color([0.6, 0.2, 1.0])  # 紫色
        geometries.append(s1)

    # 6. 接触点 p2（青色）
    if p2 is not None:
        s2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        s2.translate(np.asarray(p2))
        s2.paint_uniform_color([0.2, 1.0, 1.0])  # 青色
        geometries.append(s2)

    # 7. 可选连线
    if p1 is not None and p2 is not None:
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([p1, p2]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # 黄色线
        geometries.append(line)
    o3d.visualization.draw_geometries(geometries)










