import open3d as o3d
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('/media/labpc2x2080ti/data/dataset/Gen_Score/')
from utils.calcuate import check_force_closure
from utils.visualize import *



def sample_grasp_centers_from_mesh_interior(mesh_obj, num_samples=10000, return_ratio=0.01):
    """
    从三角网格 mesh 的体积内部采样 grasp 中心点。
    
    Args:
        mesh_obj (trimesh.Trimesh): 已填充的三角网格
        num_samples (int): 在 AABB 中总共采样的候选点数
        return_ratio (float): 从有效内部点中返回的比例（比如 0.01 表示返回 1%）
    
    Returns:
        np.ndarray: shape [M, 3] 的 grasp 中心点坐标（在 mesh 内部）
    """
    bounds = mesh_obj.bounds  # [min_xyz, max_xyz]
    pts = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_samples, 3))
    mask = mesh_obj.contains(pts)
    internal_pts = pts[mask]

    if len(internal_pts) == 0:
        raise ValueError("No internal points found. Check if your mesh is filled.")

    # 随机返回其中一部分
    keep_n = max(1, int(len(internal_pts) * return_ratio))
    sampled_pts = internal_pts[np.random.choice(len(internal_pts), keep_n, replace=False)]
    return sampled_pts




# ✅ 2. 从 grasp center 生成接触点对
def generate_contact_from_grasp_center(mesh_trimesh, center, view, angle, depth):
    view = view / np.linalg.norm(view)

    # 构造 grasp 坐标系
    up = np.array([0, 0, 1]) if abs(view[2]) < 0.95 else np.array([1, 0, 0])
    x_axis = np.cross(up, view)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(view, x_axis)
    R_base = np.stack([x_axis, y_axis, view], axis=1)

    # 应用 in-plane 旋转 angle（绕 view 方向旋转）
    R_inplane = R.from_euler('z', angle, degrees=True).as_matrix()
    R_final = R_base @ R_inplane

    # 张开方向（左右）就是旋转后的 x 轴
    width_dir = R_final[:, 0] 
    verts = mesh_trimesh.vertices
    normals = mesh_trimesh.vertex_normals
    rel = verts - center
    depth_along_view = rel @ view
    valid_mask = (depth_along_view >= 0) & (depth_along_view <= depth)
    # visualize_mask_points_on_mesh(mesh_trimesh, valid_mask, color=[0, 1, 0])    
    if not np.any(valid_mask):
        return None, None, None, None

    inside_points = verts[valid_mask]
    inside_normals = normals[valid_mask]
    width_proj = (inside_points - center) @ width_dir

    i_left = np.argmin(width_proj)
    i_right = np.argmax(width_proj)

    p1 = inside_points[i_left]
    p2 = inside_points[i_right]
    n1 = inside_normals[i_left]
    n2 = inside_normals[i_right]

    return p1, n1, p2, n2



# ✅ 3. 半球视角 + 多角度 + 多深度生成抓取对
def sample_contact_candidates_from_grasp_center(
    mesh_trimesh,
    center,
    num_views=300,
    num_angles=12,
    num_depths=4,
    depth_range=(0.02, 0.06),
    mu=0.8
):
    # 采样视角（半球均匀分布）
    phi = np.linspace(0, np.pi / 2, int(np.sqrt(num_views)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_views) * 2))
    views = []
    for p in phi:
        for t in theta:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)
            z = np.cos(p)
            views.append([x, y, z])
    views = np.array(views)
    views = np.unique(views, axis=0)

    # print(f"views:{views}")
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    depths = np.linspace(depth_range[0], depth_range[1], num_depths)

    candidates = []

    for view in views:
        for angle in angles:
            for depth in depths:
                try:
                    p1, n1, p2, n2 = generate_contact_from_grasp_center(mesh_trimesh,center, view, angle, depth)


                    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(n1).any() or np.isnan(n2).any():
                        continue

                    fc = check_force_closure(p1, n1, p2, n2, mu=mu)
                    # === 计算 width: p1-p2 向 view 的垂直距离（乘以比例因子）
                    grasp_axis = p2 - p1
                    grasp_axis_norm = grasp_axis / (np.linalg.norm(grasp_axis) + 1e-6)
                    view_norm = view / (np.linalg.norm(view) + 1e-6)

                    # 投影后保留垂直分量
                    dot = np.dot(grasp_axis_norm, view_norm)
                    proj_along_view = dot * view_norm
                    orthogonal_component = grasp_axis_norm - proj_along_view

                    grasp_width = np.linalg.norm(orthogonal_component * np.linalg.norm(grasp_axis))
                    width = 1.5 * grasp_width # 🔥 最终 width（可调比例）

                    candidates.append({
                        "center": center.copy(),
                        "p1": p1.copy(), "n1": n1.copy(),
                        "p2": p2.copy(), "n2": n2.copy(),
                        "view": view.copy(),
                        "angle": angle,
                        "depth": depth,
                        "width": width,
                        "fc": fc
                    })
                except Exception as e:
                    print(f"[ERROR] Failed grasp @ center={np.round(center, 3)}, angle={angle}, depth={depth}")
                    print(f"Reason: {e}")
                    continue

    return candidates
