import haiku as hk
import jax.numpy as jnp


class Fourier(hk.Module):

    def __init__(self, nodes, period, name=None):
        super().__init__(name=name)
        self.nodes = nodes
        self.period = period
        if period is None:
            self.period = 20

    def __call__(self, x):
        # init
        d, n = x.shape[-1], self.nodes

        assert d == 1, 'Fourier only supports one spatial dim'
        w_init = hk.initializers.TruncatedNormal(1.0)
        theta = hk.get_parameter("w", shape=[n], dtype=x.dtype, init=w_init)

        o = theta[0]
        # x += bias
        for i in range(1, n):
            o += theta[i]*jnp.cos(jnp.pi/self.period*i*x)

        return o
