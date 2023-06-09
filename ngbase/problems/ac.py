from typing import List

import jax.numpy as jnp
from jax import vmap

from ngbase.dnn.ansatz import Ansatz
from ngbase.misc.jaxtools import grad1d
from ngbase.problems.problem import Problem


def ac_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta = thetas[0]
    u = a.U(theta, X)
    u_xx = a.U_ddx(theta, X)
    return 5e-3*u_xx+u-u**3


def get_ac_eq() -> Problem:
    omega = jnp.array([[0], [2*jnp.pi]])

    def ic_0(x): return jnp.squeeze((1/3)*jnp.tanh(2*jnp.sin(x)) - jnp.exp(-23.5*(x-jnp.pi/2)**2) + jnp.exp(-27*(x-4.2)**2)
                                    + jnp.exp(-38*(x-5.4)**2))
    ic_1 = vmap(grad1d(ic_0))
    ic_2 = vmap(grad1d(grad1d(ic_0)))

    problem = Problem(
        dim=1,
        quantities=1,
        derivatives=2,
        ics=[[ic_0, ic_1, ic_2]],
        omega=omega,
        omega_init=omega,
        rhsides=[ac_rhs],
        period=2*jnp.pi,
    )
    return problem
