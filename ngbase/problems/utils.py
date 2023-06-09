import jax.numpy as jnp


def compute2DMeshGrid(Omega, Nx, Ny=None):
    assert Omega.shape == (2, 2)
    Ny = Nx if Ny is None else Ny
    x = jnp.linspace(Omega[0, 0], Omega[1, 0], num=Nx).reshape((-1,))
    y = jnp.linspace(Omega[0, 1], Omega[1, 1], num=Ny).reshape((-1,))
    X, Y = jnp.meshgrid(x, y)
    xvec = X.reshape((Nx * Ny, 1))
    yvec = Y.reshape((Nx * Ny, 1))
    XYflat = jnp.hstack([xvec, yvec])

    return X, Y, XYflat


def compute2DBoundary(Omega, Nx, Ny=None):
    assert Omega.shape == (2, 2)
    X, Y, _ = compute2DMeshGrid(Omega, Nx, Ny)
    left = jnp.vstack([X[:, 0], Y[:, 0]]).transpose()[::-1, :]
    bottom = jnp.vstack([X[0, :], Y[0, :]]).transpose()[1:(Nx - 1), :]
    right = jnp.vstack([X[:, -1], Y[:, -1]]).transpose()
    top = jnp.vstack([X[-1, :], Y[-1, :]]).transpose()[1:(Nx - 1), :][::-1, :]
    return jnp.vstack([left, bottom, right, top])


def computeNeumann2DBoundaryHelper(Omega, Nx, Ny=None):
    assert Omega.shape == (2, 2)
    X, Y, _ = compute2DMeshGrid(Omega, Nx, Ny)
    Ny = X.shape[0]
    left = jnp.vstack([X[:, 0], Y[:, 0]]).transpose()[::-1, :]
    bottom = jnp.vstack([X[0, :], Y[0, :]]).transpose()
    right = jnp.vstack([X[:, -1], Y[:, -1]]).transpose()
    top = jnp.vstack([X[-1, :], Y[-1, :]]).transpose()[::-1, :]
    def ones(dim): return jnp.ones((dim, 1))
    def zeros(dim): return jnp.zeros((dim, 1))
    left = jnp.hstack([left, -ones(Ny), zeros(Ny)])
    bottom = jnp.hstack([bottom, zeros(Nx), -ones(Nx)])
    right = jnp.hstack([right, ones(Ny), zeros(Ny)])
    top = jnp.hstack([top, zeros(Nx), ones(Nx)])
    return left, bottom, right, top


def computeNeumann2DBoundary(Omega, Nx, Ny=None):
    left, bottom, right, top = computeNeumann2DBoundaryHelper(Omega, Nx, Ny)
    return jnp.vstack([left, bottom, right, top])


def compute2DBoundaryBottomLeftCorner(Omega, Nx, Ny=None):
    assert Omega.shape == (2, 2)
    X, Y, _ = compute2DMeshGrid(Omega, Nx, Ny)
    Ny = X.shape[0]
    left = jnp.vstack([X[:, 0], Y[:, 0]]).transpose()[::-1, :]
    bottom = jnp.vstack([X[0, :], Y[0, :]]).transpose()[1:, :]
    return jnp.vstack([left, bottom])


def compute2DBoundaryRightFace(Omega, Nx, Ny=None):
    assert Omega.shape == (2, 2)
    X, Y, _ = compute2DMeshGrid(Omega, Nx, Ny)
    right = jnp.vstack([X[:, -1], Y[:, -1]]).transpose()
    return right


def mvnpdf(X, mean=None, covdiag=None):
    cons = 1./jnp.sqrt((2*jnp.pi)**mean.size*jnp.product(covdiag))
    inner = jnp.sum(jnp.multiply(X.T - mean.reshape((-1, 1)), jnp.multiply(
        (1./covdiag).reshape((-1, 1)), X.T - mean.reshape((-1, 1)))), axis=0).reshape((-1,))
    return cons*jnp.exp(-0.5*inner)
