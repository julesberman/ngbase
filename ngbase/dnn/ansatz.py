from functools import wraps
from typing import Callable

import haiku as hk
import jax
import jax.flatten_util
from jax import grad, jit, vmap

from ngbase.misc.jaxtools import divergence, grad1d, gradient, hessian


def unraveler(f, unravel, axis=0):
    @wraps(f)
    def wrapper(*args, **kwargs):
        val = args[axis]
        if (type(val) != dict):
            args = list(args)
            args[axis] = unravel(val)
            args = tuple(args)
        return f(*args, **kwargs)
    return wrapper


class Ansatz:
    """
    Ansatz is a Class which wraps a Callable function which describes the non-linear paramaterization
    here the Callable function is assumpted to be a Haiku Module which maps some x -> y
    Ansatz peforms the hk.transform necessary to make generate a pure function of the parameters, so we have u(theta, x) -> y
    In addition, ansatz applies all the necesssary jax wrappers such as vmaps, grads, etc... that are necessary to 
    to describe any sort of time dynamics as a function of spatial derivatives. 
    These are all made accessaible through the following class Properties:

    Let D be the dimension of the spatial domain.
    Let M be the number of parameters.
    Let N be the number of samples.

    Attributes
    ----------
    self.u : ((M), (D)) -> (N)
        scalar version of u
    self.U : ((M), (N, D)) -> (N)
        vectorized version of U
    self.U_dtheta : ((M)), X) -> (N, M)
        derivative with repsect to theta

    # One-Dimensional Derivatives
    self.U_dx : ((M), (N, 1)) -> (N)
        first derivative with respect to x, when space is of dim 1
    self.U_ddx : ((M), (N, 1)) -> (N)
        second derivative, when space is of dim 1
    self.U_dddx = ((M), (N, 1)) -> (N)
        third derivative, when space is of dim 1

    # Multi-Dimensional Derivative Operators
    self.U_div : ((M), (N, D)) -> (N)
        divergance operator, or sum of diagonal of jacobian
    self.U_lap : ((M), (N, D)) -> (N)
        laplacian operator, or sum of diagonal of hessian
    self.U_grad : ((M), (N, D)) -> (N, D)
        grad operator, or multi dimensional first deriviative
    """

    def __init__(self, net: Callable, key: jax.random.PRNGKey):
        self.key = key
        self.net = net

        # deal with haiku net
        trans = hk.without_apply_rng(hk.transform(net))
        self._net_apply = trans.apply
        self._net_init = trans.init

    def init_ansatz(self, X):

        theta_init = self._net_init(self.key, X)
        theta_init_flat, unravel = jax.flatten_util.ravel_pytree(theta_init)

        U_scalar = unraveler(self._net_apply, unravel)
        self.__init_grads(U_scalar)
        self.unravel = unravel

        return theta_init_flat

    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     pass

    def __init_grads(self, U_scalar):
        self.u = jit(U_scalar)  # scalar version of u
        self.U = jit(vmap(U_scalar, (None, 0)))  # vectorized U

        # derivative with repsect to theta
        self.U_dtheta = jit(vmap(grad(U_scalar), (None, 0)))

        # 1D grad, returns no spacial dims
        self.U_dx = jit(vmap(grad1d(U_scalar, 1), (None, 0)))
        self.U_ddx = jit(vmap(grad1d(grad1d(U_scalar, 1), 1), (None, 0)))
        self.U_dddx = jit(vmap(
            grad1d(grad1d(grad1d(U_scalar, 1), 1), 1), (None, 0)))
        self.U_ddddx = jit(vmap(
            grad1d(grad1d(grad1d(grad1d(U_scalar, 1), 1), 1), 1), (None, 0)))
        self.U_dddddx = jit(vmap(
            grad1d(grad1d(grad1d(grad1d(grad1d(U_scalar, 1), 1), 1), 1), 1), (None, 0)))

        # ND grads, returns no spacial dims
        self.U_div = jit(vmap(divergence(U_scalar, 1, degree=1),
                              (None, 0)))  # divergence
        self.U_lap = jit(vmap(divergence(U_scalar, 1, degree=2),
                              (None, 0)))  # laplacian
        self.U_grad = jit(vmap(gradient(U_scalar, 1, degree=1),
                               (None, 0)))  # gradient
        self.U_2grad = jit(vmap(gradient(U_scalar, 1, degree=2),
                                (None, 0)))  # gradient

        self.U_hess_theta = jit(vmap(hessian(U_scalar), (None, 0)))


def gradmap(f): return jit(vmap(grad(f), (None, 0)))
