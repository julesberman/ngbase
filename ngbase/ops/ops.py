from functools import partial
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import jit
from sklearn.decomposition import TruncatedSVD

from ngbase.dnn.ansatz import Ansatz
from ngbase.io.store import jit_save
from ngbase.problems.problem import Problem


def get_ops(methodName: str, problem: Problem, ansatz: Ansatz, rcond: float, sub_params: int, sub_sampler: str, sub_sampler_n: int, ls_rank: int) -> Callable:

    if methodName == 'dis_opt':
        ops = Forward_Euler_DtO(problem.rhsides, problem.quantities, ansatz)

    elif methodName == 'opt_dis':
        ops = OtD(problem.rhsides, problem.quantities,
                  ansatz, rcond=rcond)

    elif methodName == 'opt_dis_sub':
        sub_sampler = get_sub_sampler(sub_sampler, ls_rank, rcond)

        ops = OtD_sub(
            problem.rhsides, problem.quantities, ansatz, sub_params, sub_sampler, sub_sampler_n, rcond=rcond)

    else:
        raise Exception(f"Unknown method: {methodName}")
    return ops


class Forward_Euler_DtO:

    def __init__(self, rhsides: List[Callable], quantities: int, ansatz: Ansatz):
        """
        Parameters
        ----------
        rhsides : List[Callable]
            list of RHS for each quantity of the given system
        quantities : int
            number of quantities, should equal to len(rhsides) and len(rhsides_bc)
        ansatz : Ansatz
            ansatz, nonlinear paramaterize of class Ansatz
        """
        self.rhsides = rhsides
        self.ansatz = ansatz
        self.quantities = quantities

    @partial(jit, static_argnums=(0,))
    def __call__(self, thetas, X, Uks, fs, dt):

        total_loss = 0.0
        q = len(Uks)

        thetas = jnp.split(thetas, q)
        tufs = zip(thetas, Uks, fs)

        for q, (theta, Uk, f) in enumerate(tufs):
            Uk1 = self.ansatz.U(theta, X)
            target = Uk + dt*f
            loss = jnp.linalg.norm(Uk1 - target)  # / jnp.linalg.norm(target)
            total_loss += loss

        return total_loss/self.quantities


class OtD:
    def __init__(self, rhsides: List[Callable], quantities: int, ansatz: Ansatz, rcond: float = None):

        self.rhsides = rhsides
        self.ansatz = ansatz
        self.quantities = quantities
        self.rcond = rcond

    def __call__(self, t, thetas, X, state, key):

        thetas = jnp.split(thetas, self.quantities)

        sols = []
        for i, (rhs, theta) in enumerate(zip(self.rhsides, thetas)):
            grad_theta = self.ansatz.U_dtheta(theta, X)
            f = rhs(t, thetas, X, self.ansatz)
            sol, res, rank, s_vals = jnp.linalg.lstsq(
                grad_theta, f, rcond=self.rcond)

            jit_save(res, 'residuals')
            jit_save(rank, 'rank')
            jit_save(s_vals, 's_vals')

            sols.append(sol)

        sols = jnp.concatenate(sols)
        return sols, state


class OtD_sub:
    def __init__(self, rhsides: List[Callable], quantities: int, ansatz: Ansatz, sub_params: int, sub_sampler: Callable, sub_sampler_n: int, rcond: float = None):

        self.rhsides = rhsides
        self.ansatz = ansatz
        self.quantities = quantities
        self.rcond = rcond
        self.sub_params = sub_params
        self.sub_sampler = sub_sampler
        self.sub_sampler_n = sub_sampler_n

    def __call__(self, t, thetas, X, state, key):

        count, pi = state
        thetas = jnp.split(thetas, self.quantities)

        sols = []
        for i, (rhs, theta) in enumerate(zip(self.rhsides, thetas)):
            P = len(theta)
            randomize = self.sub_params < P

            grad_theta = self.ansatz.U_dtheta(theta, X)
            f = rhs(t, thetas, X, self.ansatz)

            # take randomizeed subset of columsn
            if randomize:
                cols_take, pi = self.sub_sampler(
                    grad_theta, self.sub_params, pi, sub_sampler_n=self.sub_sampler_n, key=key, count=count)
                grad_theta = jnp.take(grad_theta, cols_take, axis=1)

            sol, res, rank, s_vals = jnp.linalg.lstsq(
                grad_theta, f, rcond=self.rcond)

            jit_save(res, 'residuals')
            jit_save(rank, 'rank')
            jit_save(s_vals, 's_vals')

            # re-expand dim
            if randomize:
                jit_save(pi, 'pi')
                jit_save(cols_take, 'cols_take')
                sol_z = jnp.zeros(P)
                sol_z = sol_z.at[cols_take].set(sol)
                sol = sol_z

            sols.append(sol)

        sols = jnp.concatenate(sols)

        state = (count+1, pi)
        return sols, state


def get_sub_sampler(sub_sampler: str, rank: int, rcond: float) -> Callable:
    if sub_sampler == 'rand':
        return get_rand
    elif sub_sampler == 'rank':
        return get_ranks
    elif sub_sampler == 'norm':
        return get_norms
    elif sub_sampler == 'score':
        return partial(get_scores, rank)
    elif sub_sampler == 'scoreauto':
        return partial(get_scores_auto, rcond)
    else:
        raise Exception(f"Unknown sub_sampler: {sub_sampler}")


def get_rand(grad_theta, c, pi, key=None, **kwargs):
    return jax.random.choice(key, grad_theta.shape[1], shape=(c,), replace=False), pi


def host_n_rank(args):
    M, k = args
    _, _, P = scipy.linalg.qr(M, mode='economic', pivoting=True)
    return P[:k]


def get_ranks(M, c, **kwargs):
    return jax.experimental.host_callback.call(host_n_rank, (M, c), result_shape=jax.ShapeDtypeStruct((c,), jnp.int_))


def get_norms(grad_theta, c, pi, sub_sampler_n=4, key=None, count=0):
    n_params = grad_theta.shape[1]

    def get_pi(gt): return jnp.linalg.norm(gt, axis=0)
    def old_pi(gt): return pi
    pi = jax.lax.cond(count % sub_sampler_n == 0,
                      get_pi, old_pi, grad_theta)

    cols_take = jax.random.choice(
        key, n_params, shape=(c,), p=pi, replace=False)

    return cols_take, pi


def host_scores(args):

    grad_theta, k = args
    k = min(int(k), grad_theta.shape[1])
    tsvd = TruncatedSVD(n_components=k, n_iter=4)
    tsvd.fit(grad_theta)
    V = tsvd.components_.T[:, :k]
    pi = np.linalg.norm(V, axis=1)**2 / k
    return pi


def get_pi(grad_theta, c):

    n_params = grad_theta.shape[1]
    pi = jax.experimental.host_callback.call(
        host_scores, (grad_theta, c), result_shape=jax.ShapeDtypeStruct((n_params,), jnp.float32))
    return pi


def get_scores(rank, grad_theta, c, pi, sub_sampler_n=4, key=None, count=0):

    n_params = grad_theta.shape[1]

    def old_pi(gt, c): return pi
    pi = jax.lax.cond(count % sub_sampler_n == 0,
                      get_pi, old_pi, grad_theta, rank)

    cols_take = jax.random.choice(
        key, n_params, shape=(c,), p=pi, replace=False)

    return cols_take, pi


def get_scores_auto(rcond, grad_theta, c, pi, sub_sampler_n=4, key=None, count=0):

    n_params = grad_theta.shape[1]

    def old_pi(gt, rcond): return pi
    pi = jax.lax.cond(count % sub_sampler_n == 0,
                      get_pi_auto, old_pi, grad_theta, rcond)

    cols_take = jax.random.choice(
        key, n_params, shape=(c,), p=pi, replace=False)

    return cols_take, pi


def get_pi_auto(gt, rcond):
    u, s, vt = jnp.linalg.svd(gt)
    s /= s[0]
    k = jnp.argmin(s > rcond)
    v = vt.T
    v = jnp.where(s > 1e-4, x=v, y=0.0)
    pi = jnp.linalg.norm(v, axis=1)**2 / k
    return pi
