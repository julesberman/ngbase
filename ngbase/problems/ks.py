
from typing import List

import jax.numpy as jnp
from jax import vmap

from ngbase.dnn.ansatz import Ansatz
from ngbase.misc.jaxtools import grad1d
from ngbase.problems.problem import Problem


def ks_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta = thetas[0]
    u = a.U(theta, X)
    u_x = a.U_dx(theta, X)
    u_xx = a.U_ddx(theta, X)
    u_xxxx = a.U_ddddx(theta, X)
    return -u*u_x - u_xx - u_xxxx


def get_ks_eq() -> Problem:
    omega = jnp.array([[0], [32*jnp.pi]])

    def ic_0(x): return jnp.squeeze(jnp.cos(x/16)*(1 + jnp.sin(x/16)))
    ic_1 = vmap(grad1d(ic_0))

    problem = Problem(
        dim=1,
        quantities=1,
        derivatives=2,
        ics=[[ic_0, ic_1]],
        omega=omega,
        omega_init=omega,
        rhsides=[ks_rhs],
        period=32*jnp.pi,
    )
    return problem
