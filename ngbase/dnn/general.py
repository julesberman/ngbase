import haiku as hk
import jax.numpy as jnp


class QuadraticBoundryCondition(hk.Module):
    """
    TODO: Only supports semetric bc conds
    """

    def __init__(self, boundary, name=None):
        super().__init__(name=name)
        self.boundary = boundary

    def __call__(self, x):
        d = x.shape[-1]
        b = self.boundary

        a = -1/(b**2)  # calculate width of parabola based on boundary
        o = a*x**2 + 1

        return jnp.squeeze(o)


class SrectBoundryCondition(hk.Module):
    """
    TODO: Only supports semetric bc conds
    """

    def __init__(self, boundary, smoothness=10, name=None):
        super().__init__(name=name)
        self.boundary = boundary
        self.smoothness = smoothness

    def __call__(self, x):
        b = self.boundary//2
        s = self.smoothness
        o = srect(x, a=-b, b=b, s=s)
        return jnp.squeeze(o)


def srect(x, a=-4, b=4, s=10):
    y = jnp.prod(jnp.tanh(s*(x - a))) * jnp.prod(jnp.tanh(s*(b - x)))
    return y
