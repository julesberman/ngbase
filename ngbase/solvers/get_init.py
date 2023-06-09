from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import jit
from jax.experimental.host_callback import id_print, id_tap

from ngbase.dnn.ansatz import Ansatz
from ngbase.misc.jaxtools import ravelwrap
from ngbase.misc.misc import timer
from ngbase.problems.problem import Problem
from ngbase.sample.data_sampler import DataSampler
from ngbase.sample.sampler_helper import sample_equidistant_helper
from ngbase.solvers.adam import adam_opt
from ngbase.truth.get import get_truth
from ngbase.truth.utils import get_grid


@timer
def get_init(problem: Problem, sampler: DataSampler, ansatz: Ansatz, theta_init, iters: int, lr: float, optimizer: str,  lbfgs_i: int, tol: float, fit_ds: int, key: jax.random.PRNGKey):

    theta_0s = []

    # paramaterized funs for derivatives
    if problem.dim == 1:
        U_funs = [ansatz.U, ansatz.U_dx, ansatz.U_ddx, ansatz.U_dddx]
    if problem.dim > 1:
        U_funs = [ansatz.U, ravelwrap(
            ansatz.U_grad), ravelwrap(ansatz.U_2grad)]
    theta_inits = jnp.concatenate([theta_init]*problem.quantities)

    @jit
    def init_loss(thetas, u0_mats, X):
        Q = len(u0_mats)
        thetas = jnp.split(thetas, Q)
        total_loss = 0.0
        for q in range(Q):
            u0_mat_d = u0_mats[q]
            theta = thetas[q]
            for d in range(len(u0_mat_d)):
                U = U_funs[d]
                u_hat = U(theta, X)
                u_0 = u0_mat_d[d]
                total_loss += jnp.linalg.norm(u_0 - u_hat) / \
                    jnp.linalg.norm(u_0)
        return total_loss / Q

    Q, D = problem.quantities, problem.derivatives
    if fit_ds is not None:
        D = fit_ds

    all_ic_fs = problem.ics

    def get_args(key):
        u0_mats = []
        for q in range(Q):
            u0_mats_d = []
            for d in range(D):
                ic_f = all_ic_fs[q][d]
                X_init, key = sampler.sample_data_init(key)
                u0_mats_d.append(ic_f(X_init))
            u0_mats.append(u0_mats_d)

        return (u0_mats, X_init), key

    # solve optimization
    args, key = get_args(key)

    if lbfgs_i > 0.0:
        solver = jaxopt.ScipyMinimize(
            fun=init_loss, method='L-BFGS-B', maxiter=lbfgs_i, tol=tol)
        theta_inits, state = solver.run(theta_inits, *args)

    theta_0s, opt_loss, solver_state, loss_history = adam_opt(
        theta_inits, init_loss, args, steps=iters, learning_rate=lr, optimizer=optimizer, verbose=True, scheduler=True, loss_tol=tol, key=key)

    theta_0s = jnp.split(theta_0s, problem.quantities)

    # record losses
    # with jax.default_device(jax.devices("cpu")[0]):
    final_losses = np.zeros((Q, D))
    loss_grid = sample_equidistant_helper(
        sampler.omega, sampler.batch_size_init)
    for q in range(Q):
        theta = theta_0s[q]
        for d in range(D):
            U = U_funs[d]
            u_hat = U(theta, loss_grid)
            u_0 = all_ic_fs[q][d](loss_grid)
            relative_loss = jnp.linalg.norm(u_0 - u_hat)/jnp.linalg.norm(u_0)
            final_losses[q, d] = relative_loss
            print(f'q{q} d{d} relative_loss: {relative_loss:.2E}')

    return theta_0s, loss_history, final_losses
