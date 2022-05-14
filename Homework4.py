import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt

test_nx = 100

def burgers_true_sol(a, b, initial_condition, x, t):
    ic_num = initial_condition(x)
    # for safety
    min_u = np.min(ic_num) - 1
    max_u = np.max(ic_num) + 1

    u = np.zeros_like(x)
    for i, x_i in enumerate(x):

        def rootfunc(u):
            var = x_i - u * t
            var =  a + (var - a) % (b-a)
            return u - initial_condition(var)

        u[i] = sopt.bisect(rootfunc, min_u, max_u, xtol=1e-10)

    return u

def test_u(x):
    u = np.zeros_like(x)

    where = np.abs(x-1/6) < 1/7
    f = (x-1/6)**2
    u[where] = f[where]

    where = np.abs(x-3/6) < 1/7
    f = -(x-3/6)**2
    u[where] = f[where]

    where = np.abs(x-5/6) < 1/7
    f = -(x-5/6)**2 + (1/7)**2
    u[where] = f[where]

    return u

conv_nxs = [50, 50*3, 50*3**2, 50*3**3, 50*3**4]

def initial_condition_burgers(x):
    return 0.2+np.sin(2*np.pi*x)

t_burgers_before_shock = 0.1
t_burgers_after_shock = 0.5

def initial_condition_lwr(x):
    return 0.5+0.5*np.sin(2*np.pi*(x-0.25))

nx_lwr = 50*3**4
t_final_lwr = 1.5

class RefSolutionEarlyEscape(Exception):
    pass
  
x3 = np.linspace(0, 1, test_nx, endpoint=False)

def minmod(a, b):
    return (np.sign(a) + np.sign(b)) / 2 * np.minimum(np.abs(a), np.abs(b))

u3 = test_u(x3)
test_hx = 1.0/test_nx

def f(u):
    return u**2/2

def fprime(u):
    return u

#testing minmod construction
def minm(u):
    J = np.arange(0, test_nx)
    Jp1 = np.roll(J, 1) #use for j-1
    Jm1 = np.roll(J, -1) #use for j+1
    Jp2 = np.roll(J, -2) #use for j+1
    test_uplus = u[Jm1] + minmod(0.5 * (u[J] - u[Jm1]), 0.5 * (u[Jm1] - u[Jp2]))
    test_uminus = u[J] + minmod(0.5 *(u[Jm1] - u[J]), 0.5 * (u[J] - u[Jp1]))
    alpha = np.maximum(np.abs(fprime(test_uplus)), np.abs(fprime(test_uminus)))
    test_flux = (
        (f(test_uplus)+f(test_uminus))/2
        - alpha/2*(test_uplus - test_uminus))
    return test_uplus, test_uminus, test_flux

test_uplus = minm(u3)[0]
test_uminus = minm(u3)[1]
test_flux = minm(u3)[2]

#solver for burgers
def solver(nx, tt):
    tvv = []
    hx = 1/nx
    x = np.linspace(0, 1, nx, endpoint = False)
    u = initial_condition_burgers(x)
    def rhs(u, nx):
        J = np.arange(0, nx)
        Jp1 = np.roll(J, 1) #use for j-1
        Jm1 = np.roll(J, -1) #use for j+1
        Jp2 = np.roll(J, -2) #use for j+1
        uplus = u[Jm1] + minmod(0.5 * (u[J] - u[Jm1]), 0.5 * (u[Jm1] - u[Jp2]))
        uminus = u[J] + minmod(0.5 *(u[Jm1] - u[J]), 0.5 * (u[J] - u[Jp1]))
        alpha = np.maximum(np.abs(fprime(uplus)), np.abs(fprime(uminus)))
        flux = (
            (f(uplus)+f(uminus))/2
            - alpha/2*(uplus - uminus))
        return - 1/hx*(flux[J]-flux[Jp1])
    t = 0
    while t <= tt:
        ht = 2*hx/(3*np.max(np.abs(fprime(u))))
        nt = int(tt/ht)
        def SSPRK(ht, u, rhs, nx):
            u1 = u + ht*rhs(u, nx)
            return u*0.5 + 0.5*(u1 + ht*rhs(u1, nx))
        if t + ht > tt:
            ht = tt - t
            nt = int(tt/ht)
            u[:] = SSPRK(ht, u, rhs, nx)
            dd = np.diff(u)
            tv = np.sum(np.linalg.norm(abs(dd), 1))
            break
        u[:] = SSPRK(ht, u, rhs, nx)
        dd = np.diff(u)
        tv = np.sum(np.linalg.norm(abs(dd), 1))
        t = t + ht
        tvv.append(tv)
    return u, tvv
def q(k):
    return k * (1.0 - k)*(1.1 - k)

def qprime(k):
    return (1 - 2 *k) * (1.1 - k) - (k - np.square(k))

#traffic flow model
def lwr(tt):
    hx = 1/(nx_lwr)
    x5 = np.linspace(0, 1, nx_lwr, endpoint = False)
    u2 = initial_condition_lwr(x5)
    def rhs1(u2, nx_lwr):
        J = np.arange(0, nx_lwr)
        Jp1 = np.roll(J, 1) #use for j-1
        Jm1 = np.roll(J, -1) #use for j+1
        Jp2 = np.roll(J, -2) #use for j+1
        test_uplus = u2[Jm1] + minmod(0.5 * (u2[J] - u2[Jm1]), 0.5 * (u2[Jm1] - u2[Jp2]))
        test_uminus = u2[J] + minmod(0.5 *(u2[Jm1] - u2[J]), 0.5 * (u2[J] - u2[Jp1]))
        alpha = np.maximum(np.abs(qprime(test_uplus)), np.abs(qprime(test_uminus)))
        test_flux = (
            (q(test_uplus) + q(test_uminus))/2
            - alpha/2*(test_uplus - test_uminus))
        return - 1/hx*(test_flux[J]-test_flux[Jp1])
    t = 0
    while t <= tt:
        ht5 = 2*hx/(3*np.max(np.abs(qprime(u2))))
        def SSPRK1(ht5, u2, rhs1, nx_lwr):
            u11 = u2 + ht5*rhs1(u2, nx_lwr)
            return u2*0.5 + 0.5*(u11 + ht5*rhs1(u11, nx_lwr))
        if t + ht5 > tt:
            ht5 = tt - t
            u2[:] = SSPRK1(ht5, u2, rhs1, nx_lwr)
            break
        u2[:] = SSPRK1(ht5, u2, rhs1, nx_lwr)
        t = t + ht5
    return u2, x5

error_before = []
error_after = []
hxx = [1/50, 1/150, 1/450, 1/1350, 1/4050]
for i in conv_nxs:
    hx = 1/i
    x10 = np.linspace(0, 1, i, endpoint = False)
    u10 = initial_condition_burgers(x10)
    true_b = burgers_true_sol(0, 1, initial_condition_burgers , x10, t_burgers_before_shock)
    true_a = burgers_true_sol(0, 1, initial_condition_burgers, x10, t_burgers_after_shock)
    u1 = solver(i, t_burgers_before_shock)[0]
    u2 = solver(i, t_burgers_after_shock)[0]
    err_b = abs(u1 - true_b)
    err_a = abs(u2 - true_a)
    error_before.append(np.linalg.norm(err_b/i, ord = 1))
    error_after.append(np.linalg.norm(err_a/i, ord = 1))

#plotting convergence plots
plt.figure()
plt.title('Errors as a function of $\Delta x$ for Burgers Before Shock')
plt.loglog(hxx, error_before, '-o', label='Error for Burgers Before Shock')
plt.loglog(hxx, hxx, '-o', label='$O(\Delta x)$')
plt.loglog(hxx, np.square(hxx), '-o', label ='$O(\Delta^x)$')
plt.xlabel("$\Delta x$")
plt.ylabel("$||\overline{u}-u || _{\infty}$")
plt.legend()
plt.show()

plt.figure()
plt.title('Errors as a function of $\Delta x$ for Burgers After Shock')
plt.loglog(hxx, error_after, '-o' , label='Error for Burgers Before Shock')
plt.loglog(hxx, hxx, '-o', label='$O(\Delta x)$')
plt.loglog(hxx, np.square(hxx), '-o', label ='$O(\Delta^x)$' )
plt.xlabel("$\Delta x$")
plt.ylabel("$||\overline{u}-u || _{\infty}$")
plt.legend()
plt.show()

#plotting total variation of the solution
time = np.linspace(0, t_burgers_after_shock, 3457)
plt.figure()
plt.title('Total variation of the solution for Burgers')
plt.plot(time, solver(4050, t_burgers_after_shock)[1])
plt.xlabel('Time')
plt.ylabel('Total variation')
plt.show()

#plotting the solutions for the traffic flow model
#x1 = np.linspace(0, 1, nx_lwr, endpoint = False)
plt.figure()
plt.title('Solution plot for traffic flow model')
#plt.plot(lwr(0.25)[1], lwr(0.25)[0],'o-', label = 'Numerical solution until time = 0.25')
#plt.plot(lwr(0.5)[1], lwr(0.5)[0], 'o-',label = 'Numerical solution until time = 0.5')
#plt.plot(lwr(0.75)[1], lwr(0.75)[0],'o-', label = 'Numerical solution until time = 0.75')
#plt.plot(lwr(1)[1], lwr(1)[0],'o-', label = 'Numerical solution until time = 1')
plt.plot(lwr(1.25)[1], lwr(1.25)[0],'o-', label = 'Numerical solution until time = 1.25')
plt.plot(lwr(1.5)[1], lwr(1.5)[0], 'o-', label = 'Numerical solution until time = 1.5')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.show()
