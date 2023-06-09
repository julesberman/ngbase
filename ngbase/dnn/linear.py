import haiku as hk
import jax.numpy as jnp


class Periodic_Linear(hk.Module):

    def __init__(self, nodes, period, name=None):
        super().__init__(name=name)
        self.nodes = nodes
        self.period = period

    def __call__(self, x):
        d, m = x.shape[-1], self.nodes
        w_init = hk.initializers.TruncatedNormal(1.0)
        a = hk.get_parameter("a", shape=[m, d], dtype=x.dtype, init=w_init)
        phi = hk.get_parameter("phi", shape=[m, d], dtype=x.dtype, init=w_init)
        c = hk.get_parameter("c", shape=[m, d], dtype=x.dtype, init=w_init)

        omeg = jnp.pi*2/self.period
        o = a*jnp.cos(omeg*x+phi)+c
        o = jnp.mean(o, axis=1)

        return o
