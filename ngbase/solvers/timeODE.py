
from typing import Callable, List

import jax
import jax.numpy as jnp

from ngbase.dnn.ansatz import Ansatz
from ngbase.io.store import jit_save
from ngbase.misc.misc import jqdm, timer
from ngbase.sample.data_sampler import DataSampler
from ngbase.solvers.adam import adam_opt
from ngbase.solvers.odeint import get_odeint


@timer
def timeODE(rhs_reparam: Callable, sampler: DataSampler, t_eval: jnp.ndarray, t_store: jnp.ndarray, theta_0s: List[jnp.ndarray], method: str, scheme: str,  ansatz: Ansatz, seed: int):

    dt = t_eval[1]
    Tend = t_eval[-1] + dt

    key = jax.random.PRNGKey(int(seed*1e9))
    X, key = sampler.sample_data(key)

    @jqdm(Tend)
    def rhs_wrap(t, thetas, state, key):
        return rhs_reparam(t, thetas, X, state, key)

    theta_0 = jnp.concatenate(theta_0s)

    odeint = get_odeint(scheme)
    if method == 'opt_dis_sub':
        state = (0, jnp.zeros_like(theta_0))
    else:
        state = None

    thetas = odeint(rhs_wrap, theta_0, t_eval, state, key)

    return thetas
