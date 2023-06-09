
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from ngbase.io.utils import save_pickle
from ngbase.misc.misc import jqdm
from ngbase.problems.get import get_problem
from ngbase.sample.data_sampler import get_data_sampler
from ngbase.solvers.odeint import odeint_fe, odeint_rk4


def true_wavebc(problem, t_eval, T, err_pts=100_000, fwd_euler=False, periodic=False, order_2=False):

    sampler = get_data_sampler('equi', problem.omega, problem.omega_init,
                               err_pts, 0, None, None)

    X, _ = sampler.sample_data(None)
    (a, b), (c, d) = problem.omega
    bc_idx = (X[:, 0] == a) | (X[:, 0] == c) | (X[:, 1] == b) | (X[:, 1] == d)

    N = int(len(X)**(1/2))
    v_space = X.reshape(N, N, 2)[0, :, 1]
    x_space = X.reshape(N, N, 2)[:, 0, 0]
    dx = x_space[1] - x_space[0]
    dy = v_space[1] - v_space[0]

    def get_laplacian(f_p, dx):
        u = f_p.reshape(N, N)

        if periodic:
            u_ext = jnp.zeros((N+4, N+4))
            u_ext = u_ext.at[2:-2, 2:-2].set(u)

            u_ext = u_ext.at[2:-2, 0].set(u[:, -3])
            u_ext = u_ext.at[2:-2, 1].set(u[:, -2])

            u_ext = u_ext.at[0, 2:-2].set(u[-3, :])
            u_ext = u_ext.at[1, 2:-2].set(u[-2, :])

            u_ext = u_ext.at[2:-2, -1].set(u[:, 2])
            u_ext = u_ext.at[2:-2, -2].set(u[:, 1])

            u_ext = u_ext.at[-1, 2:-2].set(u[2, :])
            u_ext = u_ext.at[-2, 2:-2].set(u[1, :])

            # 4th order central difference
            u_lap = (1/(dx*dy))\
                * (- 1 * u_ext[2:-2, 0:-4]
                   + 16 * u_ext[2:-2, 1:-3]
                   - 1 * u_ext[0:-4, 2:-2]
                   + 16 * u_ext[1:-3, 2:-2]
                   - 60 * u_ext[2:-2, 2:-2]
                    + 16 * u_ext[3:-1, 2:-2]
                   - 1 * u_ext[4:, 2:-2]
                    + 16 * u_ext[2:-2, 3:-1]
                   - 1 * u_ext[2:-2, 4:]
                   ) / 12

        # elif order_2:

        #     # # 2nd order central difference
        #     # l = (1/(dx*dx))\
        #     #     * (0.25 * u[0:-2, 0:-2]
        #     #         + 0.5 * u[1:-1, 0:-2]
        #     #         + 0.25 * u[2:, 0:-2]
        #     #         + 0.5 * u[0:-2, 1:-1]
        #     #         - 3 * u[1:-1, 1:-1]
        #     #         + 0.5 * u[2:, 1:-1]
        #     #         + 0.25 * u[0:-2, 2:]
        #     #         + 0.5 * u[1:-1, 2:]
        #     #         + 0.25 * u[2:, 2:]
        #     #        )
        #     # u_lap = u_lap.at[1:-1, 1:-1].set(l)

        #     # l = (1/(dx*dx))\
        #     #     * (1 * u[0:-2, 0:-2]
        #     #         + 4 * u[1:-1, 0:-2]
        #     #         + 1 * u[2:, 0:-2]
        #     #         + 4 * u[0:-2, 1:-1]
        #     #         - 20 * u[1:-1, 1:-1]
        #     #         + 4 * u[2:, 1:-1]
        #     #         + 1 * u[0:-2, 2:]
        #     #         + 4 * u[1:-1, 2:]
        #     #         + 1 * u[2:, 2:]
        #     #        )/6
        #     # u_lap = u_lap.at[1:-1, 1:-1].set(l)

        #     u_ext = jnp.zeros((N+2, N+2))
        #     u_ext = u_ext.at[1:-1, 1:-1].set(u)

        #     u_ext = u_ext.at[0, 1:-1].set(-1*u[1, :])
        #     u_ext = u_ext.at[1:-1, 0].set(-1*u[:, 1])
        #     u_ext = u_ext.at[-1, 1:-1].set(-1*u[-2, :])
        #     u_ext = u_ext.at[1:-1, -1].set(-1*u[:, -2])

        #     # 4th order central difference
        #     l = (1/(dx*dy))\
        #         * (- 1 * u_ext[2:-2, 0:-4]
        #            + 16 * u_ext[2:-2, 1:-3]
        #            - 1 * u_ext[0:-4, 2:-2]
        #            + 16 * u_ext[1:-3, 2:-2]
        #            - 60 * u_ext[2:-2, 2:-2]
        #             + 16 * u_ext[3:-1, 2:-2]
        #            - 1 * u_ext[4:, 2:-2]
        #             + 16 * u_ext[2:-2, 3:-1]
        #            - 1 * u_ext[2:-2, 4:]
        #            ) / 12

        #     u_lap = u_lap.at[1:-1, 1:-1].set(l)

        # # 4th order 9pt Laplacian
        # l = (1/(dx*dy))\
        #     * (1 * u[0:-2, 0:-2]
        #        + 4 * u[1:-1, 0:-2]
        #        + 1 * u[2:, 0:-2]
        #        + 4 * u[0:-2, 1:-1]
        #        - 20 * u[1:-1, 1:-1]
        #        + 4 * u[2:, 1:-1]
        #        + 1 * u[0:-2, 2:]
        #        + 4 * u[1:-1, 2:]
        #        + 1 * u[2:, 2:]
        #        ) / 6
        # u_lap = u_lap.at[1:-1, 1:-1].set(l)

        else:

            u_lap = jnp.zeros_like(u)
            if order_2:
                # 2nd order 9pt Laplacian
                l = (1/(dx*dx))\
                    * (1 * u[0:-2, 0:-2]
                       + 4 * u[1:-1, 0:-2]
                       + 1 * u[2:, 0:-2]
                       + 4 * u[0:-2, 1:-1]
                       - 20 * u[1:-1, 1:-1]
                       + 4 * u[2:, 1:-1]
                       + 1 * u[0:-2, 2:]
                       + 4 * u[1:-1, 2:]
                       + 1 * u[2:, 2:]
                       )/6
                u_lap = u_lap.at[1:-1, 1:-1].set(l)

            else:

                # 4th order 2SHOC scheme
                # https://carretero.sdsu.edu/publications/postscript/2SHOC.pdf
                d = jnp.zeros_like(u)
                dl = (1/(dx*dx))\
                    * (
                    + 1 * u[1:-1, 0:-2]
                    + 1 * u[0:-2, 1:-1]
                    - 4 * u[1:-1, 1:-1]
                    + 1 * u[2:, 1:-1]
                    + 1 * u[1:-1, 2:]
                )
                d = d.at[1:-1, 1:-1].set(dl)

                p1 = (-1/(12))\
                    * (
                    + 1 * d[1:-1, 0:-2]
                    + 1 * d[0:-2, 1:-1]
                    - 12 * d[1:-1, 1:-1]
                    + 1 * d[2:, 1:-1]
                    + 1 * d[1:-1, 2:]
                )
                p2 = (1/(dx*dx))\
                    * (1 * u[0:-2, 0:-2]
                       + 1 * u[2:, 0:-2]
                       - 4 * u[1:-1, 1:-1]
                       + 1 * u[0:-2, 2:]
                       + 1 * u[2:, 2:]
                       ) / 6
                l = p1 + p2
                u_lap = u_lap.at[1:-1, 1:-1].set(l)

        return u_lap

    @jit
    @jqdm(T)
    def rhs(t, y, c=1.0):
        f_p, f_v = jnp.split(y, 2)

        f_p = jax.lax.cond(periodic, lambda f_p: f_p, lambda f_p: f_p.at[bc_idx].set(0.0), f_p)

        laplacian = get_laplacian(f_p, dx)
        laplacian = laplacian.ravel()
        dt_p = f_v
        dt_v = c**2*laplacian
        return jnp.concatenate([dt_p, dt_v])

    ic_p_f, ic_b_f = problem.ics[0][0], problem.ics[1][0]
    ic_p = ic_p_f(X)
    ic_v = ic_b_f(X)
    ic = jnp.concatenate([ic_p, ic_v])

    #     result = scipy.integrate.solve_ivp(
    #         rhs, (t_eval[0], t_eval[-1]), ic, method='RK45', t_eval=t_store, max_step=dt)
    #     y = result.y

    if fwd_euler:
        y = odeint_fe(rhs, ic, t_eval)
    else:
        y = odeint_rk4(rhs, ic, t_eval)
    y = jnp.swapaxes(y, 0, 1)

    y_p, y_v = np.split(y, 2, axis=0)
    sol_p = np.transpose(y_p.reshape(N, N, -1), axes=[2, 0, 1])
    sol_v = np.transpose(y_v.reshape(N, N, -1), axes=[2, 0, 1])

    return [sol_p, sol_v], [t_eval, x_space, v_space]


if __name__ == "__main__":

    problem = get_problem('wavebc')

    T = 8.0
    dt = 5e-3
    t_store = np.linspace(0.0, T, int(T/dt)+1)
    print(t_store)
    sols, spacing = true_wavebc(problem, t_store, T, err_pts=500_000, periodic=True)
    print('solved!')

    sols = [np.asarray(s, dtype=np.float32) for s in sols]
    spacing = [np.asarray(s, dtype=np.float32) for s in spacing]

    data = {'true': sols, 'spacing': spacing}

    # f_p = get_interpolated_gt(sols[0], spacing)
    # f_v = get_interpolated_gt(sols[1], spacing)

    outpath = Path(f'/Users/julesberman/sc/ngbase/outputs/truth/gt_{problem.name}_8.pkl')

    save_pickle(outpath, data)
    print('done!')
