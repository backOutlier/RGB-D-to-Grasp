from tools.generate_object_grasp import GraspCandidateGenerator
import os
import json
from utils.extract import *
import multiprocessing
from functools import partial
from utils.sample import *

from config import parser
args = parser.parse_args()



def process_object(object_id, base_path, scene_folder):
    print(f"\n[Object {object_id}] Processing...")

    # fuse point clouds from multiple views(36 views in MetaGraspNet)
    object_points_world, background_points_world = merge_multiview_pointclouds(
        base_path, object_id, visualize_after_fusion=args.visualize_fusion_pc, return_background=True
    )


    grasp_object = GraspCandidateGenerator(
        object_points_world=object_points_world,
        background_points_world=background_points_world.points
    )

    candidate=grasp_object.generate_grasp_candidates(
        num_points=args.num_points, num_views=args.num_views, num_angles=args.num_angles, num_depths=args.num_depths,
        depth_range=args.depth_range,  visualize_each=args.visualize_grasp, visualize_num=args.visualize_num, num_threads=args.num_threads
    )
    scene_name = os.path.basename(os.path.normpath(scene_folder))      # 'scene0' 等

    root_dir   = "/media/labpc2x2080ti/data/dataset/Gen_Score/npy"     # 顶层目录
    scene_dir  = os.path.join(root_dir, scene_name)                    # …/npy/scene0
    os.makedirs(scene_dir, exist_ok=True)                              # 自动建文件夹

    save_path  = os.path.join(scene_dir, f"{object_id:03d}_candidates.npz")
    np.savez_compressed(save_path, candidates=candidate)               # 建议用 np.savez_compressed

    print(f"[✓] Saved to {save_path}")

def main():
    base_path = args.base_path
    scene_id = args.scene_id
    for scene_id in range(1, 10):  # 1 to 499   
        scene_path = os.path.join(base_path, f"scene{scene_id}")
        scene_folder = scene_path
        order_path = os.path.join(scene_path, "0_order.json")
        with open(order_path, 'r') as f:
            order_data = json.load(f)

        ordered_ids = [int(k) for k in sorted(order_data.keys(), key=lambda x: order_data[x]["layer"]) if int(k) > 0]


        for object_id in ordered_ids:
            process_object(object_id, base_path=scene_path, scene_folder=scene_folder)

if __name__ == "__main__":
    main()