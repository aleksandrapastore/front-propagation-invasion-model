###
# Phase-plane trajectories of the Fisher-KPP travelling-wave system.
###
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#close the currently open plots
plt.close('all')

def fisher_kpp_ode(z, y, c):
    '''
    Function that defines the travelling-wave Fisher-KPP ODE.
    '''
    #z is the independent variable, y is the state vector, c is the parameter (wave speed)
    U = y[0]
    W = y[1]
    #Define the system
    dU_dz = W
    dW_dz = -c * W - U * (1 - U)
    return [dU_dz, dW_dz]

def unstable_eigenvalue(c):
    '''
    Define function for the positive eigenvalue, which defines the unstable manifold
    of the saddle equilibrium (1,0).
    '''
    return (-c + np.sqrt(c**2 + 4)) / 2

#We want to output two figures: one with c<cmin and second one with c=cmin
fig, axes = plt.subplots(1, 2)
wave_speeds = [0.5, 2.0]
titles = [r'$c = 0.5 < c_{\min}$', r'$c = c_{\min} = 2$']

#Loop over each wave speed and subplot
for ax, c, title in zip(axes, wave_speeds, titles):

    #Start near the unstable eigenvalue
    lambda_unstable = unstable_eigenvalue(c)
    #Define small perturbation
    eps = 1e-3
    #We want to perturb the system in a way that the initial point lies on the unstable eigendirection.
    #The unstable eigenvector is (1, lambda)^T
    #(1,0)-eps(1, lambda)^T=(1-eps, -eps*lambda)
    #Perturb the initial value of U slightly below 1
    U0 = 1.0 - eps
    #Perturb W0 accordingly as well so that the perturbation lies on the unstable eigenvector.
    W0 = -eps * lambda_unstable

    # Integrate forward in z
    sol = solve_ivp(fisher_kpp_ode, [0, 60], [U0, W0],
                   args=(c,),
                   #want a continuous approximation of the solution
                   dense_output=True,
                   #max delta z step
                   max_step=0.05,
                   #relative error tolerance
                   rtol=1e-10,
                   #absolute error tolerance
                   atol=1e-12)

    #create plotting points
    z_plot = np.linspace(0, 60, 5000)
    #evaluate the solution of the plotting points to obtain y
    y_plot = sol.sol(z_plot)
    #extract the trajectories for U and W
    U_traj = y_plot[0]
    W_traj = y_plot[1]

    #Plot the phase plane trajectory
    ax.plot(U_traj, W_traj, 'k-', lw=2, label='Trajectory', zorder=5)
    
    #Mark the two equilibria
    #Saddle equilibrium - empty black circle
    ax.plot(1, 0,
            marker='o',
            markersize=7,
            markerfacecolor='white',
            markeredgecolor='black',
            zorder=7,
            label='Saddle $(1,0)$')

    #Stable equilibrium - filled black circle
    ax.plot(0, 0,
            marker='o',
            markersize=7,
            markerfacecolor='black',
            markeredgecolor='black',
            zorder=7,
            label='Stable node $(0,0)$' if c >= 2 else 'Stable spiral $(0,0)$')

    #Define axis limits
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-0.45, 0.25)

    #Label the axes and the plot
    ax.set_xlabel(r'$U$', fontsize=12)
    ax.set_ylabel(r'$W$', fontsize=12)
    ax.set_title(title, fontsize=12)

    #Reference lines for U=0, W=0
    ax.axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, alpha=0.5)

    #Annotate the equilibria in the plot
    ax.annotate(r'$(1,0)$', (1,0), xytext=(10,8), textcoords='offset points')
    ax.annotate(r'$(0,0)$', (0,0), xytext=(10,8),textcoords='offset points')

    #Remove top and right box lines for style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

#Add plot to the whole figure
plt.suptitle('Phase-plane trajectories illustrating the Fisher-KPP minimum wave speed', fontsize=12)
plt.tight_layout()
plt.savefig('fisher_kpp_ode_heteroclinic.pdf', bbox_inches='tight', dpi=300)
plt.show()