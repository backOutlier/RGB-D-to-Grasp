import numpy as np
from scipy.optimize import linprog
import sys
sys.path.append('/media/labpc2x2080ti/data/dataset/Gen_Score/')
from config import parser
args = parser.parse_args()
from scipy.spatial import ConvexHull, QhullError
from functools import lru_cache

@lru_cache(maxsize=None)
def _get_dirs(num_dirs):
    thetas = np.linspace(0, 2*np.pi, num_dirs, endpoint=False)
    return np.cos(thetas), np.sin(thetas)

def check_force_closure_with_score(p1, n1, p2, n2,
                                   num_dirs=8, mu_max=1.1,
                                   mu_min=0.1, tol=1e-4):
    """
    Force Closure 判定 + 打分。
    - 若抓取成功（可闭合），返回 True 及 score = mu_max - μ*（μ*为最小可行摩擦系数）
    - 若不可闭合，返回 False, -1.0
    """
    cos_t, sin_t = _get_dirs(num_dirs)

    def cone_wrenches(p, n, mu):
        z = n / np.linalg.norm(n)
        x = np.array([1.0, 0.0, 0.0]) if abs(z[2]) > 0.90 else np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        α = 0.95
        dirs = cos_t[:, None]*x[None, :] + sin_t[:, None]*y[None, :]
        f = z + (α * mu) * dirs
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        taus = np.cross(p[None, :], f)
        W_mat = np.concatenate([f, taus], axis=1)
        return [W_mat[i] for i in range(W_mat.shape[0])]

    def feasible(mu):
        W = cone_wrenches(p1, n1, mu) + cone_wrenches(p2, n2, mu)
        W = np.asarray(W)
        try:
            hull = ConvexHull(W, qhull_options='QJ')
            return np.all(hull.equations[:, -1] <= 1e-8)
        except QhullError:
            Wt = W.T
            A_eq = np.vstack([Wt, np.ones((1, Wt.shape[1]))])
            b_eq = np.zeros(7); b_eq[-1] = 1
            res = linprog(c=np.zeros(Wt.shape[1]),
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=(0, None), method='highs')
            return res.success

    if not feasible(mu_max):
        return False, -1.0

    lo, hi = mu_min, mu_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid

    score = mu_max - hi
    return True, score
