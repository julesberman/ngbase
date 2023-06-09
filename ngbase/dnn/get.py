from typing import Callable, Iterable

import haiku as hk
import jax
import jax.numpy as jnp

from ngbase.dnn.ansatz import Ansatz
from ngbase.dnn.fourier import Fourier
from ngbase.dnn.gaussian import Gaussian, GaussianPeriodic
from ngbase.dnn.general import SrectBoundryCondition
from ngbase.dnn.linear import Periodic_Linear
from ngbase.dnn.rational import Rational
from ngbase.dnn.wavelet import Wavelet


def get_ansatz(model, activation, width, depth, period, zero_bc, key: jax.random.PRNGKey) -> Ansatz:
    net = build_nn(model, activation, width, depth, period, zero_bc)
    ansatz = Ansatz(net, key)
    return ansatz


def build_nn(model, activation, width, depth, period, zero_bc):

    def net(x):
        if model == 'gaussian':
            layers = get_gaussian(width, period)
        elif model == 'dnn':
            layers = get_dnn(activation, width, depth, period)
        elif model == 'fourier':
            layers = [Fourier(width, period)]
        elif model == 'wavelet':
            layers = get_wavelet(activation, width, depth)

        f = hk.Sequential([*layers, hk.Linear(1)])
        y = f(x)

        if zero_bc is not None:
            bc = get_bc(zero_bc)
            y = y * bc(x)

        y = jnp.squeeze(y)

        return y

    return net


def get_wavelet(activation, width, depth) -> Iterable[Callable]:
    layers = []
    for d in range(depth-1):
        u = hk.Linear(width)
        a = get_activation(activation)
        layers.append(u)
        layers.append(a)
    layers.append(Wavelet(width))
    return layers


def get_bc(zero_bc):
    bc = SrectBoundryCondition(zero_bc, smoothness=1.0)
    return bc


def get_gaussian(width, period) -> Iterable[Callable]:
    if period is not None:
        u = GaussianPeriodic(width, period=period)
    else:
        u = Gaussian(width)
    return [u]


def get_dnn(activation, width, depth, period) -> Iterable[Callable]:
    layers = []
    if period is not None:
        layers = [Periodic_Linear(width, period=period), Rational()]
        depth -= 1
    for d in range(depth-1):
        u = hk.Linear(width)
        a = get_activation(activation)
        layers.append(u)
        layers.append(a)

    return layers


def get_activation(activation) -> Callable:
    if activation == 'relu':
        a = jax.nn.relu
    elif activation == 'tanh':
        a = jax.nn.tanh
    elif activation == 'sigmoid':
        a = jax.nn.sigmoid
    elif activation == 'selu':
        a = jax.nn.selu
    elif activation == 'rational':
        a = Rational()
    elif activation == 'swish':
        a = jax.nn.swish

    return a
