import haiku as hk
import jax.numpy as jnp


class Wavelet(hk.Module):

    def __init__(self, nodes, name=None):
        super().__init__(name=name)
        self.nodes = nodes

    def __call__(self, x):
        # init
        d, n = x.shape[-1], self.nodes

        # assert d == 1, 'Wavelet only supports one spatial dim'
        f_init = hk.initializers.TruncatedNormal(1.0)
        b_init = hk.initializers.RandomUniform(minval=-jnp.pi, maxval=jnp.pi)
        w_init = hk.initializers.TruncatedNormal(1.0)
        f = hk.get_parameter("f", shape=[n, d], dtype=x.dtype, init=f_init)
        w = hk.get_parameter("w", shape=[n], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[n, d], dtype=x.dtype, init=b_init)

        return wavelet_f2(x, f, w, b)


# def wavelet_f(x, f, w, b, return_parts=False):

#     f = jnp.abs(f)
#     w = jnp.abs(w)
#     x = x - b
#     f_norm = jnp.linalg.norm(f)
#     s_f = f_norm*(1/w)
#     s_t = 1/(2*jnp.pi*s_f)

#     norm = (s_t*(2*jnp.pi)**(-1/2))**(-1/2)

#     xf_sum = jnp.sum(f*x, axis=1)

#     sin = jnp.exp(1j*2*jnp.pi*(xf_sum))
#     x_sq = jnp.sum(jnp.square(x), axis=1)
#     gauss = jnp.exp(-(x_sq) / (2*s_t**2))

#     res = (norm*sin*gauss).real

#     res = jnp.nan_to_num(res)

#     if return_parts:
#         return res, (norm, sin, gauss)

#     return res


def wavelet_f2(x, f, w, b, return_parts=False):

    f = jnp.abs(f)
    w = jnp.abs(w)
    x = x - b

    norm = 1/jnp.sqrt(jnp.pi*w)

    xf_sum = jnp.sum(f*x, axis=1)

    sin = jnp.exp(1j*2*jnp.pi*(xf_sum))

    # sin = jnp.sin(2*jnp.pi*(xf_sum))
    x_sq = jnp.sum(jnp.square(x), axis=1)
    gauss = jnp.exp(-(x_sq) / w)

    res = (norm*sin*gauss).real

    if return_parts:
        return res, (norm, sin, gauss)

    return res
