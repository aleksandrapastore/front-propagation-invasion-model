###
# Plot of the heteroclinic trajectory Gamma_1 in the K_1 blow-up chart.
###

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.serif": "serif",
})

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

# We only need the critical Fisher--KPP wave speed for Gamma_1
c = 2
# Compute the unstable eigenvalue at the saddle equilibrium (1,0)
lambda_unstable = unstable_eigenvalue(c)
# Define a small perturbation away from the saddle equilibrium
eps = 1e-3

# Initialise the trajectory close to (1,0), along the unstable eigendirection
# The unstable eigenvector is (1, lambda_unstable)^T
# We subtract eps*(1, lambda_unstable)^T to move into the biological region
U0 = 1.0 - eps
W0 = -eps * lambda_unstable

# Integrate the Fisher--KPP travelling-wave system forward in z
sol = solve_ivp(
    fisher_kpp_ode,
    [0, 80],
    [U0, W0],
    args=(c,),
    dense_output=True,
    max_step=0.05,
    rtol=1e-10,
    atol=1e-12
)

# Create plotting points in the travelling-wave coordinate z
z_plot = np.linspace(0, 80, 5000)

# Evaluate the numerical solution
y_plot = sol.sol(z_plot)

# Extract the Fisher--KPP variables U and W
U_traj = y_plot[0]
W_traj = y_plot[1]

# We exclude very small values of U to avoid division by zero numerically.
mask = U_traj > 1e-25

# Transform the Fisher--KPP orbit into the K_1 blow-up coordinates
# In K_1: r_1 = U and w_1 = W/U.
r1_traj = U_traj[mask]
w1_traj = W_traj[mask] / U_traj[mask]

# Create the figure
fig, ax = plt.subplots()

# Plot the singular orbit Gamma_1 in the (r_1,w_1) plane
ax.plot(
    r1_traj,
    w1_traj,
    'k-',
    lw=2,
    label=r'$\Gamma_1$',
    zorder=5
)

# Mark the invaded \bar P_1 equilibrium in the K_1 chart
ax.plot(
    1, 0,
    marker='o',
    markersize=7,
    markerfacecolor='white',
    markeredgecolor='black',
    zorder=7,
    label=r'$\bar P_1=(1,0)$'
)

# Mark the blown-up leading-edge equilibrium \bar Q_1 in the K_1 chart
ax.plot(
    0, -1,
    marker='o',
    markersize=7,
    markerfacecolor='black',
    markeredgecolor='black',
    zorder=7,
    label=r'$\bar Q_1=(0,-1)$'
)

# Add an arrow to indicate the direction of the flow along Gamma_1
# The trajectory flows from P_1 towards Q_1.
arrow_idx = int(0.22 * len(r1_traj))
arrow_step = 40

ax.annotate(
    '',
    xy=(r1_traj[arrow_idx + arrow_step], w1_traj[arrow_idx + arrow_step]),
    xytext=(r1_traj[arrow_idx], w1_traj[arrow_idx]),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.8),
    zorder=6
)

# Add reference lines
# w_1 = 0 corresponds to W = 0
#ax.axhline(0, color='grey', lw=0.5, alpha=0.5)

# w_1 = -1 corresponds to the leading-edge direction W = -U
#ax.axhline(-1, color='grey', lw=0.8, ls='--', alpha=0.7)

# r_1 = 0 corresponds to U = 0
#ax.axvline(0, color='grey', lw=0.5, alpha=0.5)

# Label the axes
ax.set_xlabel(r'$r_1$', fontsize=12)
ax.set_ylabel(r'$w_1$', fontsize=12)

ax.xaxis.set_label_coords(0.5, 1)


# Annotate the two equilibria
ax.annotate(
    r'$\bar P_1=(1,0)$',
    (1, 0),
    xytext=(-10, 15),
    textcoords='offset points'
)

ax.annotate(
    r'$\bar Q_1=(0,-1)$',
    (0, -1),
    xytext=(10, 4),
    textcoords='offset points'
)

# Label the heteroclinic trajectory
ax.annotate(
    r'$\Gamma_1$',
    (0.45, -0.30),
    xytext=(8, -8),
    textcoords='offset points'
)

# Choose axis limits
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-1.05, 0.08)

ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

# Place the horizontal axis at w_1 = 0
ax.spines['bottom'].set_position(('data', 0))

# Place the vertical axis at r_1 = 0
ax.spines['left'].set_position(('data', 0))

# Remove top and right box lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Optional legend
#ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig('gamma1_K1_orbit.pdf', bbox_inches='tight', dpi=300)
plt.show()