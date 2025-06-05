from scipy.spatial import ConvexHull
import sys
sys.path.append('/media/labpc2x2080ti/data/dataset/Gen_Score/')
from utils.gen_contact_points import *
from utils.calcuate import *
from utils.fill_mesh import sample_surface_and_inner_centers, trimesh_to_open3d
from utils.sample import *
from utils.visualize import *
import open3d as o3d
import numpy as np
import trimesh
from typing import List
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from multiprocessing import Pool

def scale_point_cloud(points, scale=0.8):
    """
    缩小点云：以质心为中心，向内缩放一定比例
    :param points: (N, 3) np.ndarray
    :param scale: 缩放比例（0.8 表示缩小 20%）
    :return: 缩放后的点云 (N, 3)
    """
    center = np.mean(points, axis=0, keepdims=True)
    return center + scale * (points - center)

def build_object_pcd_from_numpy(object_points_np: np.ndarray, object_normals_np: np.ndarray):
    
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(object_points_np)

    pcd.normals = o3d.utility.Vector3dVector(object_normals_np)

    return pcd

def get_object_size_oriented(pcd):
    obb = pcd.get_oriented_bounding_box()
    extent = obb.extent  # 长宽高（主方向）
    print(f"[DEBUG] Object size (Oriented Bounding Box): {extent}")
    

def process_center_worker(arg):
    center, object_pcd_np, object_normal_np, background_points_world, num_views, num_angles, num_depths, depth_range = arg

    try:
        object_pcd = build_object_pcd_from_numpy(object_pcd_np, object_normal_np)
        get_object_size_oriented(object_pcd)
        object_points = np.asarray(object_pcd.points)  # (N, 3) numpy array
        object_points = scale_point_cloud(object_points, scale=0.8)
    
        sub_candidates = sample_contact_candidates_from_grasp_center(
            object_pcd=object_pcd,
            object_points=object_points,
            background_points_world=background_points_world,
            center=center,
            num_views=num_views,
            num_angles=num_angles,
            num_depths=num_depths,
            depth_range=depth_range,
        )
        # print_grasp_candidate_distribution(sub_candidates)
        # visualize_random_grasps_graspnet_style(sub_candidates, object_pcd, num_each=3,background_points_world= background_points_world)
        return sub_candidates

    except Exception as e:
        print(f"[ERROR @ center {np.round(center, 3)}] {e}")
        return []

class GraspCandidateGenerator:
    def __init__(self, object_points_world: np.ndarray, background_points_world: np.ndarray):
        self.object_pcd = object_points_world
        self.object_pcd.estimate_normals()
        self.background_points_world = self.extract_convex_hull_points(background_points_world)
        # self.background_points_world = background_points_world
        self.background_points_world_show = self.background_points_world
        print(f"[DEBUG] object_pcd has {len(self.object_pcd.points)} points.")

    def extract_convex_hull_points(self, background_points, voxel_size=0.001, num_samples=5000):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(background_points)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        hull, _ = pcd.compute_convex_hull()

       
        dense = hull.sample_points_uniformly(number_of_points=num_samples)
        return np.asarray(dense.points)


    def generate_grasp_candidates(
        self,
        num_points=None,
        num_views=None,
        num_angles=None,
        num_depths=None,
        depth_range=None,
        visualize_each=False,  
        visualize_num=None,
        num_threads=None,
    ) -> List[dict]:
        if visualize_each:
            print("[WARNING] Visualization is disabled in multiprocessing mode.")

        grasp_centers=sample_points_on_pointcloud(self.object_pcd, surface_sample_num=args.num_points,voxel_size=args.voxel_size)
        print(f"Generated {len(grasp_centers)} combined grasp centers.")
        object_points_np = np.asarray(self.object_pcd.points)
        object_normals_np = np.asarray(self.object_pcd.normals)
        args_list = []
        for center in grasp_centers:
            args_list.append((
                    center,  # Ensure center is a copy to avoid shared state issues
                    object_points_np,
                    object_normals_np, 
                    self.background_points_world,
                    num_views,
                    num_angles,
                    num_depths,
                    depth_range,
                ))
        all_candidates = []

        with Pool(processes=8) as pool:
            results = list(tqdm(pool.imap_unordered(process_center_worker, args_list), total=len(args_list), desc="Processing grasp centers"))

        
        for res in results:
            all_candidates.extend(res)


        print(f"Total candidates found: {len(all_candidates)}")
        return all_candidates



from collections import Counter
import numpy as np

def print_grasp_candidate_distribution(sub_candidates):
    """
    打印 sub_candidates 中各项字段的分布统计：
    - fc (force closure)
    - collide (碰撞)
    - depth 分布
    - score 分布，包括 score > 0.9 的数量（排除 -1）
    """
    fc_list = [g.get("fc", False) for g in sub_candidates]
    collide_list = [g.get("collide", False) for g in sub_candidates]
    depth_list = [g.get("depth", -1) for g in sub_candidates]
    score_list = [g.get("score", None) for g in sub_candidates if "score" in g]

    print("=== Grasp Candidate Distribution ===")
    print(f"Total grasps: {len(sub_candidates)}")
    print(f"  FC=True      : {sum(fc_list)}")
    print(f"  FC=False     : {len(fc_list) - sum(fc_list)}")
    print(f"  Collide=True : {sum(collide_list)}")
    print(f"  Collide=False: {len(collide_list) - sum(collide_list)}")

    # Depth 分布（用 Counter）
    depth_counter = Counter(depth_list)
    print(f"  Depth values and counts:")
    for d, c in sorted(depth_counter.items()):
        print(f"    depth = {d:.5f} → {c} grasps")
    
    # Score 统计（如果存在）
    if score_list:
        score_array = np.array(score_list)
        valid_scores = score_array[score_array != -1]  # ✅ 排除 -1
        high_score_count = np.sum(valid_scores < 0.8)  #

        print(f"  Score: min={np.min(score_array):.4f}, max={np.max(score_array):.4f}, mean={np.mean(score_array):.4f}")
        print(f"  Score < 0.8 (excluding -1): {high_score_count}")
    else:
        print("  Score: no score field found.")