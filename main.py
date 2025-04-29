from tools.generate_object_grasp import GraspCandidateGenerator
import os
import json
from utils.extract import *
def main():
    # === 文件路径 ===
    base_path = "/media/labpc2x2080ti/data/dataset/MetaGraspNet/scene2/"

    order_path = os.path.join(base_path, "2_order.json")
    out_path   = "/media/labpc2x2080ti/data/dataset/meta_ann_gen/merged_2.npz"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
   


    with open(order_path, 'r') as f:
        order_data = json.load(f)

    ordered_ids = [int(k) for k in sorted(order_data.keys(), key=lambda x: order_data[x]["layer"]) if int(k) > 0]

    scene_folder = "/media/labpc2x2080ti/data/dataset/MetaGraspNet/scene2"
    for object_id in ordered_ids:
        print(f"\n[Object {object_id}] Processing...")
        object_points_world, background_points_world=merge_multiview_pointclouds(base_path,object_id, visualize_after_fusion=False, return_background=True)
        # object_mesh_path = "/media/labpc2x2080ti/data/dataset/Gen_Score/Meshes/scene2_object1.obj"
        object_mesh,object_mesh_path= create_mesh_from_points(object_points_world.points, method='bpa', depth=9, scene_folder=scene_folder, object_id=object_id)
        grasp_object=GraspCandidateGenerator(mesh_path=object_mesh_path, background_points_world=background_points_world.points)
        grasp_object.generate_grasp_candidates(num_points=1000, num_views=300, num_angles=12, num_depths=4, depth_range=(0.02, 0.06), mu=0.8, visualize_each=False, visualize_num=24)
        break





if __name__ == "__main__":
    main()