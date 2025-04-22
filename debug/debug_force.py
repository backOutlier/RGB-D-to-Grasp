import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import sys
sys.path.append('/media/labpc2x2080ti/data/dataset/Gen_Score/')
from utils.gen_contact_points import *
from utils.calcuate import *
from utils.fill_mesh import sample_surface_and_inner_centers, trimesh_to_open3d
from utils.sample import *
    
# mesh_path = "/media/labpc2x2080ti/data/dataset/mesh_bpa.obj"  # 例如 "object_1.obj" 或 "/path/to/your/mesh.ply"
# mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
# mesh.compute_vertex_normals()
# o3d.io.write_triangle_mesh("test_box.obj", mesh)

# contact_candidates = sample_contact_candidates_from_mesh(mesh_path, num_samples=512)

# print(f"Total candidates found: {len(contact_candidates)}")
# print("Evaluating first 10 contact pairs:\n")

# for i, (p1, n1, p2, n2) in enumerate(contact_candidates[:1000]):
#     fc = check_force_closure(p1, n1, p2, n2)
#     print(f"Pair {i+1}: Force Closure = {fc}")


import open3d as o3d
import numpy as np

def visualize_grasp_lines(mesh_o3d, grasp_list, num_show=300):
    """
    用线段可视化多个 grasp（p1-p2），类似论文刺猬图
    
    Args:
        mesh_o3d: Open3D TriangleMesh
        grasp_list: list of dicts，每个包含 'p1', 'p2', 'fc' 字段
        num_show: 显示前 N 个
    """
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])
    
    lines = []
    points = []
    colors = []

    for i, g in enumerate(grasp_list[:num_show]):
        p1 = g["p1"]
        p2 = g["p2"]
        points.extend([p1, p2])
        lines.append([2*i, 2*i+1])
        
        if g.get("fc", False):
            colors.append([0, 1, 0])   # green for valid
        else:
            colors.append([1, 0, 0])   # red for invalid

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh_o3d, line_set])






import open3d as o3d
import numpy as np
import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np

def create_pi_shaped_gripper(p1, p2, n1, n2, finger_length=0.04, color=[1, 0, 0], lift_axis=np.array([0, 0, 1])):
    """
    创建 Π 型抓爪结构，夹指沿 lift_axis 延伸，而不是直接用 n1/n2
    """
    line_set = o3d.geometry.LineSet()
    points = []
    lines = []

    # 统一使用 lift_axis（例如视图方向或垂直方向）代替 n1/n2
    p1_tip = p1 + lift_axis * finger_length
    p2_tip = p2 + lift_axis * finger_length

    points.extend([p1, p1_tip, p2, p2_tip])
    lines.append([0, 1])  # 左指
    lines.append([2, 3])  # 右指
    lines.append([1, 3])  # 横梁

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set



def visualize_pi_grippers(mesh_o3d, valid_grasps, color_mode='angle', finger_length=0.04):
    mesh = mesh_o3d
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    gripper_lines = []

    for grasp in valid_grasps:
        p1 = np.array(grasp['p1'])
        p2 = np.array(grasp['p2'])
        n1 = np.array(grasp['n1'])
        n2 = np.array(grasp['n2'])
        view = np.array(grasp['view'])  # 就是你想让抓手“竖起来”的方向
        angle = grasp.get('angle', 0)
        fc = grasp.get('fc', False)

        if color_mode == 'angle':
            norm = angle / 90
            color = [1 - norm, norm, 0]
        elif color_mode == 'fc' and fc:
            color = [0, 1, 0]
        else:
            color = [1, 0, 0]

        gripper = create_pi_shaped_gripper(p1, p2, n1, n2, finger_length, color=color, lift_axis=view)
        gripper_lines.append(gripper)

    o3d.visualization.draw_geometries([mesh] + gripper_lines)


import trimesh

mesh_path = "/media/labpc2x2080ti/data/dataset/mesh_bpa.obj"
mesh_trimesh = trimesh.load(mesh_path, process=True)
mesh_o3d = trimesh_to_open3d(mesh_trimesh)
print(f"[DEBUG] mesh_o3d type = {type(mesh_o3d)}")

# sampling the surface centers
grasp_centers = sample_points_on_mesh(mesh_o3d, num_points=1000, visualize=True)
print(f"Generated {len(grasp_centers)} combined grasp centers.")
# visualize_grasp_centers(mesh_o3d, grasp_centers)


candidates = []
for center in grasp_centers:
    print(f"\n=== Grasp Center @ {np.round(center, 3)} ===")
    
    sub_candidates = sample_contact_candidates_from_grasp_center(
        mesh_trimesh=mesh_trimesh,
        center=center,
        num_views=300,
        num_angles=12,
        num_depths=4,
        depth_range=(0.02, 0.06),
        mu=0.8
    )

    # === 统计抓取结果 ===
    valid_grasps = [g for g in sub_candidates if g.get("fc", False) and "p1" in g and "p2" in g]
    invalid_grasps = [g for g in sub_candidates if not g.get("fc", False) and "p1" in g and "p2" in g]

    print(f"  → Valid FC grasps:   {len(valid_grasps)} / {len(sub_candidates)}")
    print(f"  → Invalid FC grasps: {len(invalid_grasps)}")

    # === 可视化部分抓取 ===
    print(f"  → Visualizing 24 valid grasps (0~24):")
    visualize_pi_grippers(mesh_o3d, valid_grasps[1:24], color_mode='fc')

    print(f"  → Visualizing 24 invalid grasps (0~24):")
    visualize_pi_grippers(mesh_o3d, invalid_grasps[1:24], color_mode='fc')

    # === 累计结果
    candidates += sub_candidates
    
print(f"Total candidates found: {len(candidates)}")




# # # 从 mesh 表面采样中心点
# num_grasp_centers = 10  # 自己调整数量
# pcd = mesh.sample_points_uniformly(number_of_points=num_grasp_centers)
# grasp_centers = np.asarray(pcd.points)


# for idx, center in enumerate(grasp_centers):
#     print(f"\n=== Grasp Center #{idx+1} @ {np.round(center, 3)} ===")
#     candidates = sample_contact_candidates_from_mesh_obj(
#         center=center,
#         num_views=300,
#         num_angles=12,
#         num_depths=4,
#         depth_range=(0.00, 0.12),
#         mu=0.8
#     )
#     fc_count = sum(1 for c in candidates if c["fc"])
#     print(f"FC success: {fc_count} / {len(candidates)}")
