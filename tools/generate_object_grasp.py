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

class GraspCandidateGenerator:
    def __init__(self, mesh_path: str, background_points_world):
        self.mesh_path = mesh_path
        self.mesh_trimesh = trimesh.load(mesh_path, process=True)
        self.mesh_o3d = self.trimesh_to_open3d(self.mesh_trimesh)
        self.background_points_world = self.extract_convex_hull_points(background_points_world)
        self.background_points_world_show = self.background_points_world
        print(f"[DEBUG] mesh_o3d type = {type(self.mesh_o3d)}")
        
    def extract_convex_hull_points(self,background_points, voxel_size=0.01):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(background_points)

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        hull, _ = pcd.compute_convex_hull()
        hull_pts = np.asarray(hull.vertices)
        all_pts = np.asarray(hull.vertices)
        return np.asarray(hull.vertices)

    def trimesh_to_open3d(self, mesh_trimesh):
        
        import open3d as o3d
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        mesh_o3d.compute_vertex_normals()
        return mesh_o3d

    def generate_grasp_candidates(
        self,
        num_points=1000,
        num_views=300,
        num_angles=12,
        num_depths=4,
        depth_range=(0.02, 0.06),
        mu=0.8,
        visualize_each=False,
        visualize_num=240,
    ) -> List[dict]:
        grasp_centers = sample_points_on_mesh(self.mesh_o3d, num_points=num_points, visualize=False)
        print(f"Generated {len(grasp_centers)} combined grasp centers.")
        all_candidates = []

        for center in grasp_centers:
            print(f"\n=== Grasp Center @ {np.round(center, 3)} ===")

            sub_candidates = sample_contact_candidates_from_grasp_center(
                mesh_trimesh=self.mesh_trimesh,
                background_points_world=self.background_points_world, 
                center=center,
                num_views=num_views,
                num_angles=num_angles,
                num_depths=num_depths,
                depth_range=depth_range,
                mu=mu
            )

            valid_grasps = [
                            g for g in sub_candidates
                            if g.get("fc", False) and not g.get("collides_with_background", True)
                            and "p1" in g and "p2" in g
                           ]
            semi_valid_grasps = [
                            g for g in sub_candidates
                            if g.get("fc", False)
                            and "p1" in g and "p2" in g
                           ]

            invalid_grasps = [
                              g for g in sub_candidates
                              if (not g.get("fc", False) or g.get("collides_with_background", False))
                              and "p1" in g and "p2" in g
                             ]

            grasps=[g for g in sub_candidates if "p1" in g and "p2" in g]
            print(f"  → Valid FC grasps:   {len(valid_grasps)} / {len(sub_candidates)}")
            print(f"  → Invalid FC grasps: {len(invalid_grasps)}")
            print(f"  → Semi-valid grasps: {len(semi_valid_grasps)}")
            if visualize_each:  
                visualize_pi_grippers(self.mesh_o3d, grasps[1:1+visualize_num],finger_length=2, background_points_world=self.background_points_world_show)
                visualize_pi_grippers(self.mesh_o3d, valid_grasps[1:24], finger_length=2)


            all_candidates += sub_candidates

        print(f"Total candidates found: {len(all_candidates)}")
        return all_candidates


