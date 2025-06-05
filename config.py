import argparse

parser = argparse.ArgumentParser()




#defining the dataset settings
parser.add_argument('--base_path', type=str, default="/media/labpc2x2080ti/data/dataset/MetaGraspNet/", help="Base dataset path")
parser.add_argument('--scene_id', type=int, default=None, help="Scene ID to process (1-499); leave empty to process all")




#defining the force closure hyperparameters
parser.add_argument('--num_directions', type=int, default=8, help='Number of directions to discretize the friction cone.')



#defining the grasp generation hyperparameters
parser.add_argument('--num_points', type=int, default=100, help='Number of points to sample from the mesh.')
parser.add_argument('--num_views', type=int, default=300, help='Number of views to sample from the mesh.')
parser.add_argument('--num_angles', type=int, default=12, help='Number of angles to sample from the mesh.')
parser.add_argument('--num_depths', type=int, default=4, help='Number of depths to sample from the mesh.')
parser.add_argument('--depth_range', type=float, nargs=2, default=(0.01, 0.04), help='Range of depths to sample from the mesh.')

#defining the visualization hyperparameters
parser.add_argument('--visualize_grasp', action='store_false', help='Visualize each grasp candidate.')
parser.add_argument('--visualize_num', type=int, default=100, help='Number of grasp candidates to visualize.')
parser.add_argument('--visualize_fusion_pc', action='store_false', help='Visualize each fused point cloud.')



#defining the sampling hyperparameters
parser.add_argument('--voxel_size', type=float, default=0.0002, help='Voxel size for downsampling the point cloud.')
parser.add_argument('--normal_z_thresh', type=float, default=0.6, help='Z threshold for downsampling the point cloud.')
parser.add_argument('--scale', type=float, default=0.01, help='Scale factor for downsampling the point cloud.')
# parser.add_argument('--target_ratio', type=float, default=0.01, help='Target ratio for downsampling the point cloud.')
#defining the multiprocessing hyperparameters   
parser.add_argument('--num_threads', type=int, default=2, help='Number of threads to use for multiprocessing.')

#defining the gripper hyperparameters
parser.add_argument('--gripper_width', type=float, default=0.02, help='Width of the gripper.')
parser.add_argument('--gripper_height', type=float, default=0.10, help='Height of the gripper.')
parser.add_argument('--gripper_thickness', type=float, default=0.01, help='Thickness of the gripper.')
parser.add_argument('--grasp_max_width', type=float, default=0.30, help='Maximum width of the grasp.')