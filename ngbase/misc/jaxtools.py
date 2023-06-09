import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from jax import numpy as jnp
from jax.experimental.host_callback import id_print, id_tap


def hessian(f):
    return jacfwd(jacrev(f))


def divergence(f, argnum=0, degree=1):

    def f_split_dims(dims, *args):
        X = jnp.stack(args[:dims])
        args = list(args[dims:])
        args[argnum] = X
        return f(*args)

    def wrap(*fargs, **fkwarg):
        X = fargs[argnum]
        dims = X.shape[0]

        result = 0.0
        for d in range(dims):
            df = f_split_dims
            for _ in range(degree):
                df = grad(df, argnums=d+1)
            result += df(dims, *X, *fargs, **fkwarg)

        return result

    return wrap


def gradient(f, argnum=0, degree=1):

    def f_split_dims(dims, *args):
        X = jnp.stack(args[:dims])
        args = list(args[dims:])
        args[argnum] = X
        return f(*args)

    def wrap(*fargs, **fkwarg):
        X = fargs[argnum]
        dims = X.shape[0]

        result = []

        for d in range(dims):
            df = f_split_dims
            for _ in range(degree):
                df = grad(df, argnums=d+1)
            result.append(df(dims, *X, *fargs, **fkwarg))

        return jnp.array(result)

    return wrap


def ravelwrap(f, *args, **kwargs):
    return lambda *fargs, **fkwargs: jnp.ravel(f(*fargs, **fkwargs), *args, **kwargs)


def grad1d(f, *args, **kwargs):
    return lambda *fargs, **fkwargs: jnp.squeeze(jacfwd(f, *args, **kwargs)(*fargs, **fkwargs))


def batchmap(f, n_batches, argnum=0):

    def wrap(*fargs, **fkwarg):
        fargs = list(fargs)
        X = fargs[argnum]
        batches = jnp.split(X, n_batches, axis=0)

        result = []
        for B in batches:
            fargs[argnum] = B
            a = f(*fargs, **fkwarg)
            result.append(a)

        return jnp.concatenate(result)

    return wrap
