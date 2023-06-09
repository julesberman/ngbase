from typing import List

import jax.numpy as jnp
from jax import vmap

from ngbase.dnn.ansatz import Ansatz
from ngbase.problems.problem import Problem


def wave_pos_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta_vel = thetas[1]
    vels = a.U(theta_vel, X)
    return vels


def wave_vel_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    c = 1.0
    theta_pos = thetas[0]
    laplacian = a.U_lap(theta_pos, X)
    return c**2*laplacian


def get_2D_wave_eq_bc() -> Problem:

    dim = 2
    R = jnp.pi
    omega = jnp.array([dim*[-R], dim*[R]])
    sigma = jnp.array([0.25]*dim)
    mean = jnp.array([-0.5]*dim)

    def init_cond_pos_scalar(x):
        cons = 1.0/jnp.sqrt((2*jnp.pi)**mean.size*jnp.product(sigma))
        inner = jnp.sum(jnp.multiply(x.T - mean, jnp.multiply((1./sigma), x.T - mean)))
        return cons*jnp.exp(-0.5*inner)

    init_cond_pos = vmap(init_cond_pos_scalar)
    init_cond_vel = init_cond_pos

    problem = Problem(
        dim=dim,
        quantities=2,
        derivatives=1,
        ics=[[init_cond_pos], [init_cond_vel]],
        omega=omega,
        omega_init=omega,
        rhsides=[wave_pos_rhs, wave_vel_rhs],
        zero_bc=R*2,
    )
    return problem
