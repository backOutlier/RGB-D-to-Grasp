import open3d as o3d
import numpy as np
import sys
sys.path.append('/media/labpc2x2080ti/data/dataset/Gen_Score/')
from utils.gen_contact_points import sample_contact_candidates_from_grasp_center
from config import parser
args = parser.parse_args()
from main import GraspCandidateGenerator

# ---------- 1. 创建并缩放立方体 ----------
L = 0.03  # 立方体边长：0.03 m
mesh_box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
mesh_box.translate(-mesh_box.get_center())     # 把几何中心移到原点
mesh_box.scale(L, center=(0, 0, 0))            # 缩放到目标尺寸

# ---------- 2. 立方体表面均匀采样 ----------
N = 4096
pcd = mesh_box.sample_points_uniformly(number_of_points=N,
                                       use_triangle_normal=True)
pcd.paint_uniform_color([0.3, 0.6, 0.9])

# ---------- 3. 抽取顶面 (z ≈ +L/2) 点并随机选 1 个 ----------
eps = 1e-6
pts = np.asarray(pcd.points)
top_mask = np.abs(pts[:, 2] - L / 2) < eps
top_pts = pts[top_mask]

if len(top_pts) < 10:
    raise ValueError(f"顶面点不足 10 个，仅有 {len(top_pts)} 个")

rng = np.random.default_rng()
idx = rng.choice(len(top_pts), size=1, replace=True)
sampled_top = top_pts[idx]             # shape (1, 3)

# 转为 PointCloud 方便可视化
top_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_top))
top_pcd.paint_uniform_color([1, 0, 0])
center = sampled_top[0]

# 小球半径
r = 0.01
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
sphere.translate(center)                    # ① 先对齐到顶面点
sphere.compute_vertex_normals()

# —— 关键：再远离立方体 —— #
D = 0.05                                    # 额外偏移 5 cm
offset_dir = np.array([1.0, 0.0, 0.0])      # 向 +X 方向
sphere.translate(D * offset_dir)            # ② 再移出去

# 采样背景点
sphere_pcd = sphere.sample_points_uniformly(number_of_points=2048, use_triangle_normal=True)
background_points_np = np.asarray(sphere_pcd.points, dtype=np.float32)

# 查看效果
o3d.visualization.draw_geometries([pcd, top_pcd, sphere_pcd])
grasp_object = GraspCandidateGenerator(
    object_points_world=pcd,
    background_points_world=background_points_np
)

grasp_object.generate_grasp_candidates(
        num_points=args.num_points, num_views=args.num_views, num_angles=args.num_angles, num_depths=args.num_depths,
        depth_range=args.depth_range,  visualize_each=args.visualize_grasp, visualize_num=args.visualize_num, num_threads=args.num_threads
    )