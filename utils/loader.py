import open3d as o3d
import numpy as np
import glob
import json
import h5py
import os

def load_intrinsics_and_poses(scene_dir):
    npz_paths = sorted([
    p for p in glob.glob(os.path.join(scene_dir, "*.npz"))
    if not os.path.basename(p).startswith("_")])

    json_paths = [p.replace(".npz", "_camera_params.json") for p in npz_paths]
    hdf5_paths = [p.replace(".npz", "_scene.hdf5") for p in npz_paths]

    intrinsics = []
    extrinsics = []

    for json_path, hdf5_path in zip(json_paths, hdf5_paths):
        with open(json_path, "r") as f:
            cam = json.load(f)
        fx = cam["fx"]
        fy = cam["fy"]
        cx = cam["resolution"]["width"] // 2
        cy = cam["resolution"]["height"] // 2
        intrinsics.append((fx, fy, cx, cy))

        with h5py.File(hdf5_path, "r") as f:
            pose = f["camera/pose_relative_to_world"][0]
        extrinsics.append(pose)

    return npz_paths, intrinsics, extrinsics



def extract_object_and_background_from_npz(npz_path, object_id, fx, fy, cx, cy, extrinsic):
    data = np.load(npz_path)
    depth = data["depth"]
    label = data["instances_objects"] 
    mask_object = (label == object_id)
    mask_background = (label != object_id)

    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth

    x3 = (x - cx) * z / fx
    y3 = (y - cy) * z / fy
    z3 = z
    points_cam = np.stack([x3, y3, z3], axis=-1).reshape(-1, 3)

  
    mask_object_flat = mask_object.reshape(-1)
    mask_background_flat = mask_background.reshape(-1)

    object_points_cam = points_cam[mask_object_flat]
    background_points_cam = points_cam[mask_background_flat]

    object_points_world = (extrinsic[:3, :3] @ object_points_cam.T + extrinsic[:3, 3:4]).T
    background_points_world = (extrinsic[:3, :3] @ background_points_cam.T + extrinsic[:3, 3:4]).T

    return object_points_world, background_points_world
