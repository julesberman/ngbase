import haiku as hk
import jax.numpy as jnp


class GaussianPeriodic(hk.Module):

    def __init__(self, nodes, period, name=None):
        super().__init__(name=name)
        self.nodes = nodes
        self.period = period

    def __call__(self, x):
        # init
        d, n = x.shape[-1], self.nodes
        w_init = hk.initializers.TruncatedNormal(stddev=1. / jnp.sqrt(n))
        w = hk.get_parameter("w", shape=[n], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[n, d], dtype=x.dtype, init=w_init)

        # function
        o = jnp.sin(jnp.pi*(x-b)/self.period)
        o = jnp.linalg.norm(o, axis=1)
        o = jnp.exp(-w**2 * o**2)

        return o


class Gaussian(hk.Module):

    def __init__(self, nodes, name=None):
        super().__init__(name=name)
        self.nodes = nodes

    def __call__(self, x):
        # init
        d, n = x.shape[-1], self.nodes
        w_init = hk.initializers.TruncatedNormal(stddev=1. / jnp.sqrt(n))
        w = hk.get_parameter("w", shape=[n], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[n, d], dtype=x.dtype, init=w_init)

        # function
        o = x-b
        o = jnp.linalg.norm(o, axis=1)
        o = jnp.exp(-w**2 * o**2)

        return o
