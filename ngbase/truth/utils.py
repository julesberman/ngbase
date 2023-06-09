
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ngbase.sample.sampler_helper import sample_equidistant_helper


def get_relative_error(truth_fs, solutions, space_time_grid):

    T = solutions[0].shape[0]
    rel_errs = []
    rel_err_times = []
    for (gt_f, sol) in zip(truth_fs, solutions):
        gt_val = gt_f(space_time_grid)
        rel_err = np.linalg.norm(sol.ravel() - gt_val.ravel()) / np.linalg.norm(gt_val.ravel())
        rel_err_time = np.linalg.norm(sol.reshape(T, -1) - gt_val.reshape(T, -1),
                                      axis=1) / np.linalg.norm(gt_val.reshape(T, -1), axis=1)
        rel_errs.append(rel_err)
        rel_err_times.append(rel_err_time)

    return rel_errs, rel_err_times


def get_solutions(ansatz, thetas, grid):
    sols = []
    for theta in thetas:
        t_steps = theta.shape[0]
        sol = np.zeros((t_steps, grid.shape[0]))
        for t in range(t_steps):
            sol[t, ...] = ansatz.U(theta[t], grid)
        sols.append(sol)
    return sols


def get_grid(omega, N_pts, time=None):

    dims = omega.shape[-1]
    # take nth root
    N = int(N_pts**(1/dims))
    if time is None:
        spacing = []
    else:
        spacing = [time]
    for d in range(dims):
        pts = np.linspace(omega[0, d], omega[1, d], N, dtype=np.float32)
        spacing.append(pts)
    m_grids = np.meshgrid(*spacing, indexing='ij')
    m_grids = [m.flatten() for m in m_grids]
    X = np.array(m_grids, dtype=np.float32).T
    return X


def get_interpolated_gt(gt, spacing):
    gt_f = RegularGridInterpolator(spacing, gt, method='linear', bounds_error=True)
    return gt_f
