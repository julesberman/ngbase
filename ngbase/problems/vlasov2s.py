from functools import partial
from typing import List

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental.host_callback import id_print, id_tap
from jax.numpy import trapz
from scipy.fftpack import diff

from ngbase.dnn.ansatz import Ansatz
from ngbase.io.store import RESULT
from ngbase.misc.jaxtools import gradient, ravelwrap
from ngbase.problems.problem import Problem

eps = 1.0
x_period = 21.0


@partial(jit, static_argnums=(3,))
def electron_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta_e, theta_i = thetas
    x, v = X[:, 0], X[:, 1]
    fe_grad = a.U_grad(theta_e, X)
    dx_fe, dv_fe = fe_grad[:, 0], fe_grad[:, 1]
    E, energy = get_E(thetas, X, a)

    return -(1/eps)*v*dx_fe+(1/eps)*E*dv_fe


def ion_rhs(t: float, thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    theta_e, theta_i = thetas
    x, v = X[:, 0], X[:, 1]
    fi_grad = a.U_grad(theta_i, X)
    dx_fi, dv_fi = fi_grad[:, 0], fi_grad[:, 1]
    E, energy = get_E(thetas, X, a)

    return -v*dx_fi-E*dv_fi


def compute_E(integral_f, modes):
    rhok = jnp.fft.fft(integral_f)/modes
    return jnp.real(jnp.fft.ifft(-1j*rhok))


def get_E(thetas: List[jnp.ndarray], X: jnp.ndarray, a: Ansatz):
    # NOTE here we assume X is equi spaced, will not work for nonequi sampling

    theta_e, theta_i = thetas
    n = int(len(X)**(1/2))

    f_e = a.U(theta_e, X).reshape(n, n)
    f_i = a.U(theta_i, X).reshape(n, n)
    v_space = X.reshape(n, n, 2)[0, :, 1]

    integral_f = trapz(f_i - f_e, v_space)

    # Modes for Poisson equation
    modes = jnp.zeros(n)
    k = 2 * jnp.pi / x_period
    modes = modes.at[:n//2].set(k * jnp.arange(n//2))
    modes = modes.at[n//2:].set(- k * jnp.arange(n//2, 0, -1))
    modes += modes == 0

    E = compute_E(integral_f, modes)

    x_space = X.reshape(n, n, 2)[:, 0, 0]
    e_energy = jnp.sqrt(0.5*trapz(jnp.abs(E)**2, x_space))

    # this is essentially interpolation but
    # NOTE once again assuming equi spaced grid
    E = jnp.repeat(E[:, jnp.newaxis], n, axis=1)
    E = E.reshape(len(X))

    return E, e_energy


def get_vlasov2s_eq() -> Problem:

    L = 21.0
    sigma = 0.5
    k = 2*jnp.pi/L
    v_max = 2*jnp.pi
    A = 0.01
    omega = jnp.array([[0.0, -v_max], [L, v_max]])
    period = jnp.array([L, 2*v_max])

    def ic_electron_scalar(X):
        x, v = X
        return (1/jnp.sqrt(2*jnp.pi))*jnp.exp(-v**2/2)

    def ic_ion_scalar(X):
        x, v = X
        return (v**2/(jnp.sqrt(2*jnp.pi)*sigma**3))*jnp.exp(-v**2/(2*sigma**2))*(1+A*jnp.cos(k*x))

    ic_electron = vmap(ic_electron_scalar)
    ic_ion = vmap(ic_ion_scalar)

    ic_electron_d = ravelwrap(vmap(gradient(ic_electron_scalar)))
    ic_ion_d = ravelwrap(vmap(gradient(ic_ion_scalar)))

    problem = Problem(
        dim=2,
        derivatives=2,
        quantities=2,
        ics=[[ic_electron, ic_electron_d], [ic_ion, ic_ion_d]],
        omega=omega,
        omega_init=omega,
        rhsides=[electron_rhs, ion_rhs],
        period=period
    )
    return problem
