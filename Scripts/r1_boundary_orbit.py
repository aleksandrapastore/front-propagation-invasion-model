###
# Plot of the boundary orbit on r_1 = 0 in the K_1 chart.
###

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.serif": "serif",
})

# Create the figure
fig, ax = plt.subplots()

# Invariant boundary orbit w_1 = -1
eps_plot = np.linspace(0, 1.2, 400)
w_plot = -np.ones_like(eps_plot)

ax.plot(
    w_plot,
    eps_plot,
    'k-',
    lw=2,
    zorder=5
)

# Mark the leading-edge equilibrium
ax.plot(
    -1, 0,
    marker='o',
    markersize=7,
    markerfacecolor='black',
    markeredgecolor='black',
    zorder=7
)

# Matching section epsilon_1 = 1
ax.axhline(
    1,
    color='grey',
    lw=0.8,
    ls='--'
)

# Flow direction
arrow_idx = 120
arrow_step = 50

ax.annotate(
    '',
    xy=(w_plot[arrow_idx + arrow_step],
        eps_plot[arrow_idx + arrow_step]),
    xytext=(w_plot[arrow_idx],
            eps_plot[arrow_idx]),
    arrowprops=dict(
        arrowstyle='->',
        color='black',
        lw=1.8
    ),
    zorder=6
)

# Labels
ax.set_xlabel(r'$w_1$', fontsize=12)
ax.set_ylabel(r'$\varepsilon_1$', fontsize=12)

# Match your style
ax.yaxis.set_label_coords(-0.08, 0.5)

# Annotations
ax.annotate(
    r'$\bar Q_1=(-1,0)$',
    (-1, 0),
    xytext=(10, 10),
    textcoords='offset points'
)

ax.annotate(
    r'Boundary orbit',
    (-1, 0.55),
    xytext=(10, 0),
    textcoords='offset points'
)

ax.annotate(
    r'$\varepsilon_1=1$',
    (-0.97, 1.02)
)

# Limits
ax.set_xlim(-1.08, -0.92)
ax.set_ylim(-0.05, 1.25)

# Place axes
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', -1))

# Remove box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(
    'r1_boundary_orbit.pdf',
    bbox_inches='tight',
    dpi=300
)

plt.show()