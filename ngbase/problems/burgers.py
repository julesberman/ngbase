from typing import List

import jax.numpy as jnp
from jax import vmap

from ngbase.dnn.ansatz import Ansatz
from ngbase.misc.jaxtools import grad1d
from ngbase.problems.problem import Problem


def burgers_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta = thetas[0]
    u = a.U(theta, X)
    u_x = a.U_dx(theta, X)
    u_xx = a.U_ddx(theta, X)
    return 1e-3*u_xx-u_x*u


def get_burgers_eq() -> Problem:
    omega = jnp.array([[-1], [1]])

    def ic_0(x): return jnp.squeeze((1-x**2)*jnp.exp(-30*(x+0.5)**2))
    ic_1 = vmap(grad1d(ic_0))
    ic_2 = vmap(grad1d(grad1d(ic_0)))

    problem = Problem(
        dim=1,
        quantities=1,
        derivatives=2,
        ics=[[ic_0, ic_1, ic_2]],
        omega=omega,
        omega_init=omega,
        rhsides=[burgers_rhs],
        period=2,
    )
    return problem
