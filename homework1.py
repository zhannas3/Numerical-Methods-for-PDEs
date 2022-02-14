import numpy as np
import numpy.linalg as la
import scipy.integrate as si
import matplotlib.pyplot as plt

def set_up_grid(dx, dt, T=1, L=1):
    Nx = int(np.ceil(1+L/dx))
    Nt = int(np.ceil(1+T/dt))
    dx = L/(Nx-1)
    dt = T/(Nt-1)
    x = np.linspace(0, L, Nx)
    return dx, dt, x, Nt

def run_simulation(dx, dt, init_data, true_data, T=1, scheme_name="second_order_centered"):
    assert dt <= dx

    dx, dt, x, Nt = set_up_grid(dx, dt, T=T)
    Nx = len(x)

    u0 = init_data.copy()
    C = (1.0 * dt/dx)**2.0
    u1 = np.zeros_like(u0)
    u2 = np.zeros_like(u0)
    ut = np.zeros_like(u0)
    ux = np.zeros_like(u0)
    if scheme_name == "first_order_forward":
        u1 = u0.copy()
    elif scheme_name == "second_order_centered":
        for i in range(1, Nx-1):
            u1[i] = u0[i] + 0.5 * C * (u0[i+1] - 2.0 * u0[i] + u0[i-1])
    else:
        raise ValueError("Unexpected scheme name passed to run_simulation.")
    t = dt
    energy=[]
    for n in range(1, Nt-1):
        t += dt
        for i in range(1, Nx-1):
            u2[i] = - u0[i] + 2.0 * u1[i] + C*(u1[i+1] - 2.0 * u1[i] + u1[i-1])
            ut[i] = 1.0 / (2.0 * dt) * (u2[i] - u0[i])
            ux[i] = 1.0 / (2.0 * dx) * (u1[i+1] - u1[i-1])
        # include boundary terms for ut,  ux to compute the energy
        ut[0] = 1.0 / (2.0 * dt) * (u2[0] - u0[0])
        ut[-1] = 1.0 / (2.0 * dt) * (u2[-1] - u0[-1])
        # left boundary -- use second order forward differencing
        assert ux.shape[0] > 2
        ux[0] = (-1.5 * u1[0] + 2.0 * u1[1] - 0.5 * u1[2]) / (dx)
        ux[-1] = (1.5 * u1[-1] - 2.0 * u1[-1 - 1] + 0.5 * u1[-1 - 2]) / (dx)
        energy.append(si.simps(ut*ut+ux*ux, x=x))
        u0 = u1.copy()
        u1 = u2.copy()

    assert np.isclose(T, t)
    linf_error = la.norm(u2-true_data, np.inf)
    return x, u2, linf_error, energy

# {{{

# Verify by self-convergence that the scheme is second-order accurate for a more
# general solution.

def selfconv():
    def f(x):
        return np.exp(x) * x *(1-x)

    results = []
    for dx in [0.05, 0.025, 0.0125]:
        dx, dt, x, Nt = set_up_grid(dx, dt=0.5*dx)
        initial = f(x)

        _, solution, _, _ = run_simulation(dx, dt, init_data=initial, true_data=initial, T=1)
        results.append((dx, x, solution))

    (dx0, x0, sol0), (dx1, x1, sol1), (dx2, x2, sol2) = results

    assert np.allclose(x1[::2], x0)
    assert np.allclose(x2[::4], x0)

    err0 = la.norm(sol0 - sol2[::4], np.inf)
    err1 = la.norm(sol1 - sol2[::2], np.inf)
    print(err0, err1, err1/err0)

# selfconv()

# }}}

dx = 0.1
dt = 0.05
x, solution, linf_errors, energy = run_simulation(dx, dt, initial_data, true_solution)
_, other_solution, _, _ = run_simulation(dx, dt, initial_data, true_solution,T=1,scheme_name = "first_order_forward")
plt.semilogy(x, np.abs(true_solution-solution), label="Error")
plt.legend()
plt.title("Numerical and Analytical Solution of Wave Equation,  T=1")
plt.ylabel("$u(x, T=1)$")
plt.xlabel("$x$")
plt.show()

if 0:
    plt.plot(x, initial_data, label="initial")
    plt.plot(x, solution, label="sol")
    plt.legend()
    plt.show()

# spatial convergence test
dt = 0.001
linf_errors = []
err_energy = []
first_ord = []
second_ord = []
third_ord = []

for dx,  init_data,  true_sol in spatial_convergence_data:
    _, _, linf_error, energy = run_simulation(dx, dt, init_data, true_sol)
    linf_errors.append(linf_error)
    err_energy.append(np.max(energy)-np.min(energy))
    first_ord.append(dx)
    second_ord.append(dx ** 2.0)
    third_ord.append(dx ** 3.0)
    print(dx, linf_error)

plt.loglog(first_ord, linf_errors,  "o-", label="Error")
plt.loglog(first_ord, first_ord, "o-", label=r"$\Delta x^1$")
plt.loglog(first_ord, second_ord, "o-", label=r"$\Delta x^2$")
plt.loglog(first_ord, third_ord, "o-", label=r"$\Delta x^3$")
plt.xlabel(r"$\Delta x$")
plt.ylabel(r"$||\tilde{u} - u||_\infty$")
plt.title(r"Errors as a function of $\Delta x$")
plt.legend()
plt.show()

plt.loglog(first_ord, err_energy, "o-", label="Error")
plt.loglog(first_ord, first_ord, "o-", label=r"$\Delta x^1$")
plt.loglog(first_ord, second_ord, "o-", label=r"$\Delta x^2$")
plt.loglog(first_ord, third_ord, "o-", label=r"$\Delta x^3$")
plt.xlabel(r"$\Delta x$")
plt.ylabel("$|\\max{E}-\\min{E}|$")
plt.title(r"Energy Difference as a Function of $\Delta x$")
plt.legend()
plt.show()
