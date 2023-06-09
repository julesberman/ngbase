import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

# """
# ADAPTED: https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/ode.html
# """


def get_odeint(scheme):
    if scheme == 'rk4':
        return odeint_rk4_state


def odeint_rk4(fn, y0, t):
    @jit
    def rk4(carry, t):
        y, t_prev = carry
        h = t - t_prev
        k1 = fn(t_prev, y)
        k2 = fn(t_prev + h / 2, y + h * k1 / 2)
        k3 = fn(t_prev + h / 2, y + h * k2 / 2)
        k4 = fn(t, y + h * k3)
        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y

    (yf, _), y = jax.lax.scan(rk4, (y0, jnp.array(t[0])), t)
    return y


def odeint_rk4_state(fn, y0, t, state, key):
    @jit
    def rk4(carry, t):
        y, t_prev, state, key = carry
        h = t - t_prev
        key, subkey = jax.random.split(key)

        k1, state = fn(t_prev, y, state, subkey)
        k2, state = fn(t_prev + h / 2, y + h * k1 / 2,  state, subkey)
        k3, state = fn(t_prev + h / 2, y + h * k2 / 2,  state, subkey)
        k4, state = fn(t, y + h * k3,  state, subkey)

        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t, state, key), y

    (yf, _, _, _), y = jax.lax.scan(rk4, (y0, jnp.array(t[0]), state, key), t)
    return y
