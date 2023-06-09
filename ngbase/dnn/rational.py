import haiku as hk
import jax.numpy as jnp


class Rational(hk.Module):
    """
    Rational activation function
    ref: Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend,
        Rational neural networks,
        arXiv preprint arXiv:2004.01902 (2020).

    Source: https://github.com/yonesuke/RationalNets/blob/main/src/rationalnets/rational.py

    """

    def __init__(self, p=3, name=None):
        super().__init__(name=name)
        self.p = 3
        self.alpha_init = lambda *args: jnp.array([1.1915, 1.5957, 0.5, 0.0218][:p+1])
        self.beta_init = lambda *args: jnp.array([2.383, 0.0, 1.0][:p])

    def __call__(self, x):
        alpha = hk.get_parameter("alpha", shape=[self.p+1], dtype=x.dtype, init=self.alpha_init)
        beta = hk.get_parameter("beta", shape=[self.p], dtype=x.dtype, init=self.beta_init)

        return jnp.polyval(alpha, x)/jnp.polyval(beta, x)
