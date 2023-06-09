from typing import List

import jax.numpy as jnp
from jax import vmap

from ngbase.dnn.ansatz import Ansatz
from ngbase.misc.jaxtools import grad1d
from ngbase.problems.problem import Problem


def vlasov_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):

    theta = thetas[0]
    x, v = X[:, 0], X[:, 1]
    grad_U = a.U_grad(theta, X)
    du_dx, du_dv = grad_U[:, 0], grad_U[:, 1]
    E = -jnp.sin(x)

    return - v*du_dx + E*du_dv


def get_vlasovfix_eq() -> Problem:

    omega = jnp.array([[0, -6], [jnp.pi*2, 6]])

    def ic_scalar(x):
        v = x[1]
        return 1/jnp.sqrt(2*jnp.pi)*jnp.exp(-v**2/2)

    ic_u = vmap(ic_scalar)

    problem = Problem(
        dim=2,
        derivatives=1,
        quantities=1,
        ics=[[ic_u]],
        omega=omega,
        omega_init=omega,
        rhsides=[vlasov_rhs],
        period=jnp.array([2*jnp.pi, 12])
    )
    return problem
