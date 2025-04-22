import numpy as np
from scipy.spatial import ConvexHull

def generate_friction_cone(contact_point, normal, mu=0.5, num_directions=8):
    """
    生成某接触点的摩擦锥内向量（unit wrench）。
    """
    # 基于正交向量生成摩擦锥基底
    z = normal / np.linalg.norm(normal)
    if abs(z[2]) > 0.9:
        x = np.array([1, 0, 0])
    else:
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
    y = np.cross(z, x)

    cone_directions = []
    for i in range(num_directions):
        theta = 2 * np.pi * i / num_directions
        dir_local = x * np.cos(theta) + y * np.sin(theta)
        dir_world = (dir_local + mu * z)
        dir_world /= np.linalg.norm(dir_world)
        cone_directions.append(dir_world)

    # wrench = [force, torque], 这里 torque = r x f
    wrenches = []
    for f in cone_directions:
        torque = np.cross(contact_point, f)
        wrench = np.concatenate([f, torque])
        wrenches.append(wrench)

    return np.array(wrenches)


from scipy.optimize import linprog

def check_force_closure(contact1, normal1, contact2, normal2, mu=0.8, num_directions=8):
    def friction_cone_wrenches(p, n):
        z = n / np.linalg.norm(n)
        if abs(z[2]) > 0.9:
            x = np.array([1, 0, 0])
        else:
            x = np.cross(z, [0, 0, 1])
            x /= np.linalg.norm(x)
        y = np.cross(z, x)

        directions = []
        for i in range(num_directions):
            theta = 2 * np.pi * i / num_directions
            dir_local = x * np.cos(theta) + y * np.sin(theta)
            f = dir_local + mu * z
            f /= np.linalg.norm(f)
            torque = np.cross(p, f)
            wrench = np.concatenate([f, torque])
            directions.append(wrench)
        return directions

    try:
        wrenches = friction_cone_wrenches(contact1, normal1) + friction_cone_wrenches(contact2, normal2)
        wrenches = np.array(wrenches).T
        c = np.zeros(wrenches.shape[1])
        A_eq = np.vstack([wrenches, np.ones((1, wrenches.shape[1]))])
        b_eq = np.zeros(7)
        b_eq[-1] = 1
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
        return res.success
    except Exception as e:
        # print(f"[ERROR] FC check: {e}")
        return False
