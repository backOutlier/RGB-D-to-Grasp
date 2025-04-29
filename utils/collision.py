import open3d as o3d
import numpy as np

def get_grasp_transform(p1, p2, view, epsilon=1e-6):
    """
    给定两个抓取点和抓取方向view，计算抓取中心和局部坐标轴（x, y, z）作为变换矩阵。
    保证z轴严格沿着view。
    """
    y_axis = p2 - p1
    norm = np.linalg.norm(y_axis)
    if norm < epsilon:
        return None, None  # 非法抓取：夹爪宽度太小
    y_axis /= norm

    z_axis = view / (np.linalg.norm(view) + 1e-9)

    # 保证 z_axis 和 y_axis 正交
    if abs(np.dot(y_axis, z_axis)) > 0.99:
        # 如果方向几乎平行，抓取方向非法
        return None, None

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    center = (p1 + p2) / 2.0
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return center, rotation


def build_grasp_box(center, rotation, width, depth, height=0.01):
    """
    构造一个抓取区域的 OrientedBoundingBox。
    宽度：夹爪张开方向；深度：推进方向；高度：夹爪厚度（较小）。
    """
    extent = np.array([width, height, depth])
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=rotation, extent=extent)
    return obb

def check_grasp_box_collision_with_background(obb, background_points):
    """
    判断抓取框 obb 是否与背景点云相交（即是否有点落入该框内）。
    """
    background_points = np.asarray(background_points, dtype=np.float64)
    if background_points.ndim != 2 or background_points.shape[1] != 3:
        raise ValueError(f"Invalid background_points shape: {background_points.shape}, expected (N, 3)")
    indices = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(background_points))
    return len(indices) > 0