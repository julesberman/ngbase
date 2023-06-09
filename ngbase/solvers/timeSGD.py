from typing import Callable, List

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
from jax import jit
from optax import OptState
from tqdm import tqdm

from ngbase.dnn.ansatz import Ansatz
from ngbase.io.store import RESULT
from ngbase.misc.misc import relative_error, timer
from ngbase.problems.problem import Problem
from ngbase.sample.data_sampler import DataSampler
from ngbase.solvers.adam import adam_opt


@timer
def timeSGD(loss: Callable, ansatz: Ansatz, sampler: DataSampler, problem: Problem, t_eval: jnp.ndarray, t_store: jnp.ndarray, theta_0s: List[jnp.ndarray], key: jax.random.PRNGKey,
            iters: int, lr: float,  scheduler: str, optimizer: str, lbfgs_i: int, corrector: int, tol: float):

    dt = t_eval[1]

    loss = jit(loss)
    k_steps = len(t_eval)
    store_i = 0

    quantities = len(theta_0s)
    theta_0s_flat = jnp.concatenate(theta_0s)
    thetas = np.zeros((len(t_store), *theta_0s_flat.shape))  # store
    X, key = sampler.sample_data(key)
    residuals = np.zeros((quantities, k_steps, len(X)))
    # start from init
    theta_k = theta_0s_flat
    pbar = tqdm(range(k_steps))

    aux = []
    for k in pbar:

        theta_k_split = jnp.split(theta_k, quantities)
        t = t_eval[k]
        X, key = sampler.sample_data(key)

        if t in t_store:
            thetas[store_i] = theta_k
            store_i += 1

        fs, Uks = [], []
        for i, (theta, rhs) in enumerate(zip(theta_k_split, loss.rhsides)):
            f = rhs(t, theta_k_split, X, ansatz)
            Uk = ansatz.U(theta, X)
            fs.append(f)
            Uks.append(Uk)

        args = (X, Uks, fs, dt)

        if lbfgs_i > 0.0:
            solver = jaxopt.ScipyMinimize(fun=loss, method='L-BFGS-B',  maxiter=lbfgs_i, tol=1e-7)
            theta_k, state = solver.run(theta_k, *args)

        theta_k, opt_loss, prev_state, loss_history = adam_opt(
            theta_k, loss, args,  steps=iters, learning_rate=lr, scheduler=scheduler, optimizer=optimizer, verbose=False, loss_tol=tol)

        for c in range(corrector):
            theta_k_split = jnp.split(theta_k, quantities)
            corr_fs = []
            for f, rhs in zip(fs, loss.rhsides):
                fs_new = (f+rhs(t+dt, theta_k_split, X, ansatz))/2
                corr_fs.append(fs_new)

            args = (X, Uks, corr_fs, dt)

            if lbfgs_i > 0.0:
                solver = jaxopt.ScipyMinimize(
                    fun=loss, method='L-BFGS-B',  maxiter=lbfgs_i, tol=1e-7)
                theta_k, state = solver.run(theta_k, *args)

            theta_k, opt_loss, prev_state, loss_history = adam_opt(
                theta_k, loss, args,  steps=iters, learning_rate=lr, scheduler=scheduler, optimizer=optimizer, verbose=False, loss_tol=tol)

        if corrector > 0:
            fs = corr_fs

        # compute stats
        relative_loss = []
        theta_k_split = jnp.split(theta_k, quantities)

        for q, (Uk, f) in enumerate(zip(Uks, fs)):
            Uk1 = ansatz.U(theta_k_split[q], X)
            residual = Uk1 - Uk-dt*f
            rl = relative_error(true=Uk+dt*f, test=Uk1)
            residuals[q][k] = residual
            relative_loss.append(f'{rl:.2E}')

        pbar.set_postfix({f'relative_loss': ' | '.join(relative_loss)})

    return thetas
