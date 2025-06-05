import open3d as o3d
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import sys
sys.path.append('/media/labpc2x2080ti/data/dataset/Gen_Score/')
from utils.calcuate import check_force_closure_with_score
from utils.visualize import *
from utils.collision import *
from config import parser
args = parser.parse_args()
import time
from numba import njit
from utils.debug_grasp_box import *

@njit(fastmath=True)
def _contact_kernel(points, normals,
                    center,
                    approach_dir,  # ✅ 改名更清晰：抓取方向
                    thick_dir,     # ✅ 改名更清晰：夹爪厚度方向
                    width_dir,     # ✅ 抓取开合方向，用于投影
                    depth, half_T,
                    last_depth,
                    grasp_max_width):
    """
    扫描 points，找到 width_dir 上最远的两点（且分布在抓取中心两侧）。
    返回 (idx_min, idx_max, grasp_width)
    若无可行抓取 → idx_min == -1
    """
    p_min =  1e9
    p_max = -1e9
    i_min = -1
    i_max = -1

    for idx in range(points.shape[0]):
        dx = points[idx, 0] - center[0]
        dy = points[idx, 1] - center[1]
        dz = points[idx, 2] - center[2]

        # --- ✅ depth 条件 ---
        d_view = dx*approach_dir[0] + dy*approach_dir[1] + dz*approach_dir[2]
        if d_view < -depth or (last_depth >= 0.0 and d_view > last_depth):
            continue

        # --- ✅ thickness 条件 ---
        d_thick = dx*thick_dir[0] + dy*thick_dir[1] + dz*thick_dir[2]
        if d_thick < -half_T or d_thick > half_T:
            continue

        # --- ✅ 宽度方向投影（抓取开合方向） ---
        d_proj = dx*width_dir[0] + dy*width_dir[1] + dz*width_dir[2]

        # ✅ 强制选两侧点（异号）
        if d_proj < 0 and d_proj < p_min:
            p_min, i_min = d_proj, idx
        if d_proj > 0 and d_proj > p_max:
            p_max, i_max = d_proj, idx

    if i_min == -1 or i_max == -1:
        return -1, -1, -1.0

    grasp_width = p_max - p_min
    # if grasp_width > grasp_max_width:
    #     return -1, -1, -1.0

    return i_min, i_max, grasp_width
from scipy.spatial.transform import Rotation as R
import numpy as np

def generate_contact_from_pcd_center(
    points: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    view: np.ndarray,
    angle: float,
    depth: float,
    thickness: float,
    last_depth: float = None,
    last_grasp_width: float = None,
    last_p1: np.ndarray = None,
    last_p2: np.ndarray = None,
    last_n1: np.ndarray = None,
    last_n2: np.ndarray = None,
    grasp_max_width: float = 0.12,
):
    """
    生成两指抓取的接触点，基于局部坐标系 + 抓取限制条件
    """
    # ---------- ✅ 1) 构造抓取坐标系 ----------
    z_axis = view / (np.linalg.norm(view) + 1e-9)  # approach_dir
    up = np.array([0, 0, 1], dtype=view.dtype) if abs(z_axis[2]) < 0.95 else np.array([1, 0, 0], dtype=view.dtype)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-9
    y_axis = np.cross(z_axis, x_axis)

    R_base = np.stack([x_axis, y_axis, z_axis], axis=1)
    R_inplane = R.from_euler("z", angle, degrees=True).as_matrix().astype(view.dtype)
    R_final = R_base @ R_inplane

    # ✅ 三个方向显式命名
    width_dir    = R_final[:, 0]  # 开合方向（红）
    thick_dir    = R_final[:, 1]  # 厚度方向（绿）
    approach_dir = R_final[:, 2]  # 抓取前进方向（蓝）

    # ---------- ✅ 2) 调用内核 ----------
    half_T = 0.5 * thickness
    ld = last_depth if last_depth is not None else -1.0

    idx1, idx2, grasp_width = _contact_kernel(
        points, normals,
        center.astype(points.dtype),
        approach_dir,
        thick_dir,
        width_dir,
        depth, half_T,
        ld, grasp_max_width
    )

    # ---------- ✅ 3) 返回抓取信息 ----------
    if idx1 == -1:
        return None, None, None, None, None

    p1, p2 = points[idx1], points[idx2]
    n1, n2 = normals[idx1], normals[idx2]
    return p1, n1, p2, n2, grasp_width

def fibonacci_hemisphere(num_views):
    i = np.arange(num_views)
    phi = np.arccos(1 - i / (num_views - 1))  # elevation from z=1 to z=0
    theta = 2 * np.pi * i / ((1 + 5**0.5) / 2)  # golden angle

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def sample_contact_candidates_from_grasp_center(
    object_pcd,
    object_points: np.ndarray,
    background_points_world,
    center,
    num_views=None,
    num_angles=None,
    num_depths=None,
    depth_range=None,
):
    # Capture the views from half sphere
    views = fibonacci_hemisphere(num_views)

    # print(f"views:{views}")
    angles = np.linspace(0, 180, num_angles, endpoint=False)
    depths = np.linspace(depth_range[0], depth_range[1], num_depths)
    points = np.asarray(object_pcd.points)
    normals = np.asarray(object_pcd.normals)
    candidates = []
    combinations = [(v, a, d) for v in views for a in angles for d in depths]
    generate_contact_points_time = 0
    force_closure_time = 0
    collides_check_time = 0
    previous_angle = None 
    mask_cache = {} 
    
    for (v_idx, (view, angle, depth)) in enumerate(combinations):
        try:
            if previous_angle is None or angle != previous_angle:
                last_p1 = last_p2 = last_n1 = last_n2 = None
                last_depth = last_grasp_width = None
                previous_angle = angle

            key = (depth, angle)

            if key not in mask_cache:
                rel = points - center
                view_n = view / (np.linalg.norm(view) + 1e-9)

                # 构造姿态旋转矩阵
                up = np.array([0, 0, 1], dtype=np.float32) if abs(view_n[2]) < 0.95 else np.array([1, 0, 0], dtype=np.float32)
                x_axis = np.cross(up, view_n)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(view_n, x_axis)

                R_base = np.stack([x_axis, y_axis, view_n], axis=1)
                R_inplane = R.from_euler("z", angle, degrees=True).as_matrix()
                R_final = R_base @ R_inplane

                # 投影向量（opening / thickness / approach）
                open_dir = R_final[:, 0]
                thick_dir = R_final[:, 1]
                approach_dir = R_final[:, 2]

                # 投影深度和厚度
                depth_v = rel @ approach_dir
                thick_v = rel @ thick_dir

                half_T = 0.5 * args.gripper_thickness
                valid = ((depth_v >= -depth) &
                        (thick_v >= -half_T) & (thick_v <= half_T))

                mask_cache[key] = None if valid.sum() < 2 else (points[valid], normals[valid])

            sub_pts, sub_nrm = mask_cache[key]
            # visualize_mask_points_with_grasp_frame(sub_points=sub_pts, object_pcd=object_pcd, center=center, view=view, angle=angle)
            # t0 = time.perf_counter()
            p1, n1, p2, n2, grasp_width = generate_contact_from_pcd_center(sub_pts, sub_nrm, center, view, angle, depth,thickness=args.gripper_thickness,last_depth=last_depth,last_grasp_width=last_grasp_width,last_p1=last_p1,last_p2=last_p2,last_n1=last_n1,last_n2=last_n2)
            # t1 = time.perf_counter()
            # generate_contact_points_time = generate_contact_points_time + t1 - t0
            last_depth = depth
            last_grasp_width = grasp_width
            last_p1 = p1
            last_p2 = p2
            last_n1 = n1
            last_n2 = n2

            # generate_contact_points_time = generate_contact_points_time+ t1 - t0
            if any(x is None for x in [p1, n1, p2, n2]) or view is None:
                print(f"[DEBUG] Skipping invalid grasp @ center={np.round(center, 3)}, angle={angle}, depth={depth}")
                continue
            if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(n1).any() or np.isnan(n2).any():
                print(f"[DEBUG] Skipping NaN grasp @ center={np.round(center, 3)}, angle={angle}, depth={depth}")
                continue

            # t3 = time.perf_counter()
            fc, score = check_force_closure_with_score(p1, n1, p2, n2)
            # t4 = time.perf_counter()
            # force_closure_time = force_closure_time + t4 - t3


            point_center, rotation, offset = get_grasp_transform(p1, p2, view, angle, base_point=center, depth=depth)
            # print(f"current point:{point_center},orginal point:{center},p1:{p1},p2:{p2}")
            # grasp_width = np.linalg.norm(orthogonal_component * np.linalg.norm(grasp_axis))
            # print_pointcloud_aabb(object_pcd)

            if point_center is None or rotation is None:
                collides_with_background = True
            else:
                grasp_box = build_grasp_box(point_center, rotation, width=grasp_width, height=args.gripper_height, thickness=args.gripper_thickness)
               
                # debug_grasp_box_position(
                #     grasp_box=grasp_box,
                #     ref_point=center,
                #     object_pcd=object_pcd,# 可为 PointCloud 或 ndarray
                #     p1=p1,
                #     p2=p2 , 
                # )
              
                # t5 = time.perf_counter()
                collides_with_background = check_grasp_box_collision_with_background(grasp_box, background_points_world)
                # t6 = time.perf_counter()
                # collides_check_time = collides_check_time + t6 - t5
                # grasp_box.color = (1, 0, 0) if collides_with_background else (0, 1, 0)
                # grasp_boxes.append(grasp_box)

            candidates.append({
                "center": center,
                "view": view,
                # "p1": p1,
                # "n1": n1,
                # "p2": p2,
                # "n2": n2,
                "angle": angle,
                "depth": depth,
                "offset": offset,
                "width": grasp_width,
                "fc": fc,
                "score": score,
                "collide": collides_with_background,
                "gripper_center": point_center,
                "rotation": rotation,
            })



        except Exception as e:
            print(f"[ERROR] Failed grasp @ center={np.round(center, 3) if center is not None else 'None'}, angle={angle}, depth={depth}")
            print(f"Reason: {e}")
            continue

    # print(f"[DEBUG] generate_contact_points_time: {generate_contact_points_time:.4f} seconds")
    # print(f"[DEBUG] force_closure_time: {force_closure_time:.4f} seconds")
    # print(f"[DEBUG] collides_check_time: {collides_check_time:.4f} seconds")
    return candidates
 