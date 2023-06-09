from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(1))
def sample_uniform_helper(omega, Nx, key):
    key, subkey = jax.random.split(key)
    return jax.random.uniform(subkey, (Nx, omega.shape[1]), minval=0., maxval=1.)*(omega[1, :] - omega[0, :]) + omega[0, :], key


@partial(jit, static_argnums=(2))
def sample_uniform_helper_focus(omega, omega_init, Nx, key):
    key, subkey = jax.random.split(key)
    key, ssubkey = jax.random.split(key)
    def scale(jnparr): return jnparr * \
        (omega[1, :] - omega[0, :]) + omega[0, :]
    return jax.numpy.vstack([
        scale(jax.random.uniform(
            subkey, (int(Nx/4), omega.shape[1]), minval=0., maxval=1.)),
        scale(jax.random.uniform(ssubkey, (int(3*Nx/4),
              omega_init.shape[1]), minval=0., maxval=1.))
    ]), key


def sample_equidistant_helper(omega, N_pts):

    dims = omega.shape[-1]

    # take nth root
    N = int(N_pts**(1/dims))
    pts = []
    for d in range(dims):
        pt = jnp.linspace(omega[0, d], omega[1, d], N)
        pts.append(pt)
    m_grids = jnp.meshgrid(*pts, indexing='ij')
    m_grids = [m.flatten() for m in m_grids]
    X = jnp.array(m_grids).T

    return X


def sample_equidistant_helper_focus(omega, omega_init, N_pts, ratio=0.7):

    N_pts_f = int(N_pts*ratio)
    N_pts_o = N_pts - N_pts_f

    Xf, key = sample_equidistant_helper(omega_init, N_pts_f, key)
    Xo, key = sample_equidistant_helper(omega, N_pts_o, key)

    X = jnp.concatenate([Xo, Xf], axis=0)
    return X
