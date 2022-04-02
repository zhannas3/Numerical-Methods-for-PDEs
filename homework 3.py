import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize as so

def mesh(a, b, Nx):
    """
    Generates points 1,...,Nx starting from a and ending at b
    """
    x = np.linspace(a, b, Nx)
    dx = x[1] - x[0]
    return x, dx


def gen_dt(t0, tf, Nt):
    return (tf - t0) / Nt


class SimulationConstants:
    def __init__(self):
        self.a = 0
        self.b = 5
        self.t0 = 0
        self.tf = 0.4
        self.test_Nx = 100
        self.test_Nt = self.test_Nx
        self.Nxs = [100, 200, 400, 800]
        self.Nts = [i for i in self.Nxs]


constants = SimulationConstants()
_, test_hx = mesh(constants.a, constants.b, constants.test_Nx)
test_ht = gen_dt(constants.t0, constants.tf, constants.test_Nt)
t_final = constants.tf
convergence_data = []
for Nx, Nt in zip(constants.Nxs, constants.Nts):
    _, dx = mesh(constants.a, constants.b, Nx)
    dt = gen_dt(constants.t0, constants.tf, Nt)
    convergence_data.append((dt, dx))

def initial_condition(x):
    return np.sin(2 * np.pi * x/constants.b * 3)
  
nx=int(5/test_hx)
x=np.linspace(0,5-test_hx,nx)
nt=int(t_final/test_ht)
t=np.linspace(0,t_final, nt)
u = np.zeros((nx, nt+1))
u[:,0] = initial_condition(x)
#true solution through characteristics
def true(h_x, t):
    nx = int(5/h_x)
    x = np.linspace(0,5-h_x,nx)
    k= []
    for x0 in x:
        def f(u):
            return u - initial_condition(x0 - u*t)
        k0 = so.brentq(f, 2*min(initial_condition(x)), 2*max(initial_condition(x)))
        k.append(k0)
    k=np.array(k)
    return k
true_solution = true(test_hx, t_final)
#plt.plot(x, initial_condition(x),'o-')
#plt.plot(x, true_solution, 'o-')

#First order-upwind difference
def u_d(hx, ht, s):
    nx=int(5/hx)
    x=np.linspace(0,5-hx,nx)
    nt=int(t_final/ht)
    u = np.zeros((nx, nt+1))
    u[:,0] = initial_condition(x)
    if (s): #s stands for single step
        nt=1
    for i in range(0, nt):
        for j in range(nx):
            if u[j,i] > 0:
                if j == 0: #boundary condition
                    u[j-1, i+1] = (u[j, i] - ht * u[j, i] * (u[j, i] - u[-1, i+1]) / hx)
                else:
                    u[j, i+1] = (u[j, i] - ht * u[j, i] * (u[j, i] - u[j-1, i]) / hx)

            elif u[j,i] < 0:
                if j == nx-1: #boundary condition
                    u[j, i+1] = (u[j, i] - ht * u[j, i] * (u[0, i+1] - u[j, i]) / hx)
                else:
                    u[j, i+1] = (u[j, i] - ht * u[j, i] * (u[j+1, i] - u[j, i]) / hx)
    solution = u[:,nt]
    return solution

#Central differencing
def c_d(hx, ht, s):
    nx=int(5/hx)
    x=np.linspace(0,5-hx,nx)
    nt=int(t_final/ht)
    u = np.zeros((nx, nt+1))
    u[:,0] = initial_condition(x)
    if (s): #s stands for single step
        nt=1
    for i in range(nt):
        u[1:nx-1, i+1] = (u[1:nx-1, i] - 0.25* ht * (u[2:nx, i]**2 - u[0:nx-2, i]**2) / hx)
        #boundary condition
        u[0, i+1] = (u[0, i] - 0.25 * ht * (u[1, i]**2 - u[-1, i]**2) / hx)
        u[-1, i+1] = (u[-1, i] - 0.25 * ht * (u[0, i]**2 - u[-2, i]**2) / hx)
    solution = u[:,nt]
    return solution

single_step_upwind = u_d(test_hx, test_ht, 1)
#print(single_step_upwind)
single_step_central = c_d(test_hx, test_ht, 1)
#print(single_step_central)

#infinity-norm error plots with multiple steps
htt = []
hxx = []
ud_error=[]
cd_error=[]
for i, j in enumerate(convergence_data):
    htt.append(j[0])
    hxx.append(j[1])
    nx= int(5/j[1])
    u1 = u_d(j[1], j[0], 0)
    v = true(j[1], t_final) #true solution
    ud_err = abs(u1 - v )
    ud_error.append(np.linalg.norm(ud_err, np.inf))
    u2 = c_d(j[1], j[0], 0)
    cd_err = abs(u2 - v)
    cd_error.append(np.linalg.norm(cd_err, np.inf))

#true solution vs numerical solution plots
plt.figure(figsize=(8, 6))
plt.title('Solution plots')
plt.plot(x, true(test_hx, t_final), label = 'True Solution')
plt.plot(x, u_d(test_hx, test_ht, 0), label= 'Upwind')
plt.plot(x, c_d(test_hx, test_ht, 0), label = 'Central')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.show()

#infinity norm plots
plt.figure(figsize=(8, 6))
plt.title('Errors as a function of $\Delta x$ for Upwind Scheme and Central Difference')
plt.loglog(hxx, ud_error, "o-", label='Error for Upwind Scheme')
plt.loglog(hxx, hxx, "o-",label='$O(\Delta x)$')
plt.loglog(hxx, np.square(hxx),"o-" , label='$O(\Delta x^2)$')
plt.loglog(hxx, cd_error, "o-", label='Error for Central Difference')
plt.xlabel("$\Delta x$")
plt.ylabel("$||\overline{u}-u || _{\infty}$")
plt.legend()
plt.show()
