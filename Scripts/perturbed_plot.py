###
# Phase-plane trajectories of the two-component travelling-wave ODE system.
###
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def two_component_ode(z, y, c, gamma):
    '''
    Function that defines the travelling-wave two-component ODE.
    '''
    #z is the independent variable, y is the state vector, c is the parameter (wave speed)
    U = y[0]
    V = y[1]
    W = y[2]
    #Define the system
    dU_dz = W
    dV_dz = gamma/c * U * V
    dW_dz = 1/(1-V) * (gamma/c*U*V*W -c * W - U * (1 - U - V))
    return [dU_dz, dV_dz, dW_dz]

def unstable_eigenvalue(c):
    '''
    Define function for the positive eigenvalue, which defines the unstable manifold
    of the saddle equilibrium (1,0,0).
    '''
    return (-c + np.sqrt(c**2 + 4)) / 2

def v_reaches_one(z, y, c, gamma):
    return y[1] - 0.99
v_reaches_one.terminal = True
v_reaches_one.direction = 1

c = 2
gamma = 10

#Start near the unstable eigenvalue
lambda_unstable = unstable_eigenvalue(c)
#Define small perturbation
eps = 1e-3

best = []

for a in [0, 0.1, 0.5, 1, 2, 5, 10]:
    for b in [0.1, 0.5, 1, 2, 5, 10]:

        U0 = 1.0 - eps*b
        V0 = eps*a
        W0 = -eps*b*lambda_unstable

        # Integrate forward in z
        sol = solve_ivp(two_component_ode, [0, 60], [U0, V0, W0],
                        args=(c,gamma,),
                        #we want to use the BDF solver
                        method='BDF',
                        #want a continuous approximation of the solution
                        dense_output=True,
                        #max delta z step
                        max_step=0.05,
                        #relative error tolerance
                        rtol=1e-8,
                        #absolute error tolerance
                        atol=1e-10,
                        events=v_reaches_one)
        
        z_plot = np.linspace(0, sol.t[-1], 5000)
        U_traj, V_traj, W_traj = sol.sol(z_plot)

        U_end = U_traj[-1]
        V_end = V_traj[-1]
        W_end = W_traj[-1]

        error = abs(U_traj[-1]) + abs(W_traj[-1])

        if 0 < V_end < 0.99:
            best.append((error, a, b, U_end, V_end, W_end))
            
best = sorted(best, key=lambda x: x[0])

for item in best[:10]:
    print(item)




fig = plt.figure(figsize=(14, 5))

# 3D plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(U_traj, V_traj, W_traj, 'k-', lw=2)
ax1.scatter(1, 0, 0, facecolors='white', edgecolors='black', s=50, zorder=5)
ax1.scatter(U_traj[-1], V_traj[-1], W_traj[-1], color='black', s=50, zorder=5)
ax1.set_xlabel(r'$U$')
ax1.set_ylabel(r'$V$')
ax1.set_zlabel(r'$W$')
ax1.set_title('3D trajectory')

# (U,W) projection
ax2 = fig.add_subplot(132)
ax2.plot(U_traj, W_traj, 'k-', lw=2)
ax2.scatter(1, 0, facecolors='white', edgecolors='black', s=50, zorder=5)
ax2.scatter(U_traj[-1], W_traj[-1], color='black', s=50, zorder=5)
ax2.set_xlabel(r'$U$')
ax2.set_ylabel(r'$W$')
ax2.set_title(r'Projection onto $(U,W)$')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# (U,V) projection
ax3 = fig.add_subplot(133)
ax3.plot(U_traj, V_traj, 'k-', lw=2)
ax3.scatter(1, 0, facecolors='white', edgecolors='black', s=50, zorder=5)
ax3.scatter(U_traj[-1], V_traj[-1], color='black', s=50, zorder=5)
ax3.set_xlabel(r'$U$')
ax3.set_ylabel(r'$V$')
ax3.set_title(r'Projection onto $(U,V)$')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.suptitle(rf'Trajectory near $(1,0,0)$, $\gamma={gamma}$, $c={c}$', fontsize=12)
plt.tight_layout()
plt.savefig('two_component_heteroclinic.pdf', bbox_inches='tight', dpi=300)
plt.show()