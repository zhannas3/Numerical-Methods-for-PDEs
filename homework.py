import scipy.sparse as sp
import scipy.sparse.linalg as sppla
import numpy as np
import matplotlib.pyplot as plt
def initial_condition(x):
    gaussian = lambda x, height, position, hwhm: height * np.exp(-np.log(2) * ((x - position)/hwhm)**2)
    u_init = gaussian(x, 1, 0.5, 0.1)
    return u_init
test_dx = 0.02
test_dt = 0.01
Nxs = [2 ** i for i in range(5,11)]
Nts = [2*Nx for Nx in Nxs]
convergence_data = [(1.0/Nt,1.0/(Nx-1)) for Nt, Nx in zip(Nts,Nxs)]
CFLs = np.array([2 ** i for i in range(0,4)])
nxx=int(1/test_dx)+1
x=np.linspace(0,1,nxx)
#crank-nicolson, centered space
def c_n(ht, hx, s):
    nx=int(1/hx)
    alpha=ht/hx
    nt=int(1/ht)
    u=[]
    u=np.array([initial_condition(hx*i) for i in range(nx)]) # initial solution, or also true solution
    offset = [-1,0,1]
    P1 = sp.diags([-0.25*alpha*np.ones(nx-1), np.ones(nx), 0.25*alpha*np.ones(nx-1)],offset, shape=(nx, nx)).toarray()
    P1[0][-1]=-0.25*alpha #applying boundary conditions
    P1[-1][0]=0.25*alpha
    A1 = sp.csr_matrix(P1)
    Q1 = sp.diags([0.25*alpha*np.ones(nx-1), np.ones(nx), -0.25*alpha*np.ones(nx-1)], offset, shape=(nx, nx)).toarray()
    Q1[0][-1]=0.25*alpha #applying boundary conditions
    Q1[-1][0]=-0.25*alpha
    B1 = sp.csr_matrix(Q1)
    if (s): #s stands for single step
        nt=1
    for i in range(nt):
        u=sppla.spsolve(A1, B1@u)
    u=np.append(u, u[0])
    return u
#euler-backward, centered space
def e_b(ht, hx, k):
    nx=int(1/hx)
    alpha=ht/hx
    nt=int(1/ht)
    u=[]
    offset = [-1,0,1]
    u=np.array([initial_condition(hx*i) for i in range(nx)]) # initial solution, or also true solution
    P2 = sp.diags([-0.5*alpha*np.ones(nx-1), np.ones(nx), 0.5*alpha*np.ones(nx)], offset, shape=(nx, nx)).toarray()
    P2[0][-1]=-0.5*alpha #applying boundary conditions
    P2[-1][0]=0.5*alpha
    A2 = sp.csr_matrix(P2)
    if (k): #s stands for single step
        nt=1
    for i in range(nt):
        u=sppla.spsolve(A2, u)
    u=np.append(u, u[0])
    return u
# calculate single step solutions
single_step_crank_nicholson = c_n(test_dt, test_dx, 1)
single_step_euler_backward = e_b(test_dt, test_dx, 1)
# part 2: infinity-norm error plots with multiple steps
cn_error=[]
eb_error=[]
ht=[]
for i, j in enumerate(convergence_data):
    ht.append(j[0])
    nx= int(1/j[1])
    xy = np.linspace(0,1, nx+1)
    v = initial_condition(xy) #true solution in our case
    u1 = c_n(j[0], j[1], 0)
    cn_err = abs(u1-v)
    cn_error.append(np.linalg.norm(cn_err, np.inf))
    u2 = e_b(j[0], j[1], 0)
    eb_err = abs(u2-v)
    eb_error.append(np.linalg.norm(eb_err, np.inf))
#print(ht)
#print(cn_error)
#print(eb_error)
#print(CFLs)
plt.figure()
plt.title('Errors as a function of $\Delta t$ for Crank-Nicolson')
plt.loglog(ht, cn_error, "o-", label='Error for Crank-Nicolson')
plt.loglog(ht, ht, "o-",label='$O(\Delta t)$')
plt.loglog(ht, np.square(ht),"o-" , label='$O(\Delta x^2)$')
plt.loglog(ht, np.square(ht)*ht,"o-", label='$O(\Delta x^3)$')
plt.xlabel("$\Delta t$")
plt.ylabel( "$||\overline{u}-u || _{\infty}$")
plt.legend()
plt.show()
plt.figure()
plt.title('Errors as a function of $\Delta t$ for Euler-Backward')
plt.loglog(ht, eb_error, "o-", label='Error for Euler-Backward')
plt.loglog(ht, ht, "o-", label='$O(\Delta t)$')
plt.loglog(ht, np.square(ht),"o-", label= '$O(\Delta x^2)$' )
plt.loglog(ht, np.square(ht)*ht,"o-", label='$O(\Delta x^3)$')
plt.xlabel("$\Delta t$")
plt.ylabel("$||\overline{u}-u || _{\infty}$")
plt.legend()
plt.show()
# part 3: plot x vs solutions
ts=initial_condition(x) #true solution 
hts=[CFLs[0]*test_dx,CFLs[1]*test_dx, CFLs[2]*test_dx , CFLs[3]*test_dx] #getting h_t values 
plt.rcParams["figure.figsize"]=(8, 8)
plt.figure()
plt.title('Solution after one cycle, Crank-Nicolson')
plt.plot(x, c_n(hts[0], 0.02, 0), label= 'CFL = 1')
plt.plot(x, c_n(hts[1], 0.02,0), label= 'CFL = 2')
plt.plot(x, c_n(hts[2], 0.02,0), label= 'CFL = 4')
plt.plot(x, c_n(hts[3], 0.02, 0), label= 'CFL = 8')
plt.plot(x, ts, label = 'True Solution')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.show()
plt.figure()
plt.title('Solution after one cycle, Euler-Backward')
plt.plot(x, e_b(hts[0], 0.02, 0), label= 'CFL = 1')
plt.plot(x, e_b(hts[1], 0.02,0), label= 'CFL = 2')
plt.plot(x, e_b(hts[2], 0.02,0), label= 'CFL = 4')
plt.plot(x, e_b(hts[3], 0.02, 0), label= 'CFL = 8')
plt.plot(x, ts, label = 'True Solution')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.show()
