
from typing import List

import jax.numpy as jnp
from jax import vmap

from ngbase.dnn.ansatz import Ansatz
from ngbase.misc.jaxtools import grad1d
from ngbase.problems.problem import Problem


def bz_rhs_u(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta_u, theta_v, theta_w = thetas
    u = a.U(theta_u, X)
    v = a.U(theta_v, X)
    u_xx = a.U_ddx(theta_u, X)
    return 1e-5*u_xx+u+v-u*v-u**2


def bz_rhs_v(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta_u, theta_v, theta_w = thetas
    u = a.U(theta_u, X)
    v = a.U(theta_v, X)
    w = a.U(theta_w, X)
    v_xx = a.U_ddx(theta_v, X)
    return 2e-5*v_xx+w-v-u*v


def bz_rhs_w(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta_u, theta_v, theta_w = thetas
    u = a.U(theta_u, X)
    w = a.U(theta_w, X)
    w_xx = a.U_ddx(theta_w, X)
    return 1e-5*w_xx + u - w


def get_bz_eq() -> Problem:
    omega = jnp.array([[-1], [1]])

    def ic_0_s(x): return jnp.squeeze(jnp.exp(-100*(x+0.5)**2))
    def ic_1_s(x): return jnp.squeeze(jnp.exp(-100*x**2))
    def ic_2_s(x): return jnp.squeeze(jnp.exp(-100*(x-0.5)**2))

    ic_0 = vmap(ic_0_s)
    ic_1 = vmap(ic_1_s)
    ic_2 = vmap(ic_2_s)

    d_ic_0 = vmap(grad1d(ic_0_s))
    d_ic_1 = vmap(grad1d(ic_1_s))
    d_ic_2 = vmap(grad1d(ic_2_s))

    problem = Problem(
        dim=1,
        quantities=3,
        derivatives=2,
        ics=[[ic_0, d_ic_0], [ic_1, d_ic_1], [ic_2, d_ic_2]],
        omega=omega,
        omega_init=omega,
        rhsides=[bz_rhs_u, bz_rhs_v, bz_rhs_w],
        period=2,
    )
    return problem
