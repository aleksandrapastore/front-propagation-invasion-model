###
# Plot of the heteroclinic trajectory Gamma_2 in the K_2 rescaling chart.
###

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.serif": "serif",
})

# Gamma_2 is given by w_2 = -u_2 for r_2 = 0, v_2 = 0.
u2_traj = np.linspace(1.2, 0.0, 400)
w2_traj = -u2_traj
fig, ax = plt.subplots()

# Plot the heteroclinic trajectory Gamma_2 in the (u_2,w_2) plane
ax.plot(
    u2_traj,
    w2_traj,
    'k-',
    lw=2,
    label=r'$\Gamma_2$',
    zorder=5
)

# Mark the degenerate node Q_2
ax.plot(
    0, 0,
    marker='o',
    markersize=7,
    markerfacecolor='black',
    markeredgecolor='black',
    zorder=7,
    label=r'$Q_2=(0,0)$'
)

# Add an arrow to indicate the direction of the flow along Gamma_2
# The trajectory approaches Q_2 as z increases
arrow_idx = 170
arrow_step = 40

ax.annotate(
    '',
    xy=(u2_traj[arrow_idx + arrow_step], w2_traj[arrow_idx + arrow_step]),
    xytext=(u2_traj[arrow_idx], w2_traj[arrow_idx]),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.8),
    zorder=6
)

# Place the horizontal axis at w_2 = 0
ax.spines['bottom'].set_position(('data', 0))

# Place the vertical axis at u_2 = 0
ax.spines['left'].set_position(('data', 0))

# Remove top and right box lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Choose axis limits
ax.set_xlim(-0.05, 1.25)
ax.set_ylim(-1.25, 0.15)

# Remove the duplicate zero on the u_2 axis by setting x-ticks manually
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

# Label the axes
ax.set_xlabel(r'$u_2$', fontsize=12)
ax.set_ylabel(r'$w_2$', fontsize=12)

# Place the u_2 label near the horizontal axis
ax.xaxis.set_label_coords(0.5, 0.97)

# Annotate Q_2
ax.annotate(
    r'$Q_2=(0,0)$',
    (0, 0.02),
    xytext=(10, 8),
    textcoords='offset points'
)

# Anotate the \Gamma_2 trajectory
ax.annotate(
    r'$\Gamma_2$',
    (0.65, -0.57),
    xytext=(8, -10),
    textcoords='offset points'
)

plt.tight_layout()
plt.savefig('gamma2_K2_orbit.pdf', bbox_inches='tight', dpi=300)
plt.show()