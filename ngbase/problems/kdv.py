from typing import List

import jax.numpy as jnp
from jax import vmap

from ngbase.dnn.ansatz import Ansatz
from ngbase.misc.jaxtools import grad1d
from ngbase.problems.problem import Problem
from ngbase.truth.kdv import kdv_scalar


def Kdv_RHS(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta = thetas[0]
    return -1*(a.U_dddx(theta, X)+6*a.U(theta, X)*a.U_dx(theta, X))


def get_kdv() -> Problem:

    omega = jnp.array([[-10.0], [20.0]])
    rhs = Kdv_RHS
    def ic_scalar(x): return kdv_scalar(x, 0)

    kd_u = vmap(ic_scalar)
    kd_ud = vmap(grad1d(ic_scalar))

    problem = Problem(
        dim=1,
        quantities=1,
        derivatives=2,
        ics=[[kd_u, kd_ud]],
        omega=omega,
        omega_init=omega,
        rhsides=[rhs],
        period=30.0
    )
    return problem
