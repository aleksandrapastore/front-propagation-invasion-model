###
# Blow-up geometry for the two-component travelling-wave problem in the $\gamma \to \infty$ limit.
###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Define axis ranges
U_range = np.linspace(0, 1, 60)
V_range = np.linspace(0, 0.9, 60)
W_range = np.linspace(-0.35, 0.45, 60)

# Define colours
blue = "#4C72B0"
orange = "#DD8452"
green = "#55A868"
red = "#C44E52"
purple = "#8172B3"

# Plot the critical manifold branches
# M_{0V}
U_grid, W_grid = np.meshgrid(U_range, W_range)
V_grid = np.zeros_like(U_grid)
ax.plot_surface(
    U_grid, V_grid, W_grid,
    alpha=0.25, color=blue, edgecolor=blue, linewidth=0.15
)
# M_{0U}
V_grid2, W_grid2 = np.meshgrid(V_range, W_range)
U_grid2 = np.zeros_like(V_grid2)
ax.plot_surface(
    U_grid2, V_grid2, W_grid2,
    alpha=0.25, color=orange, edgecolor=orange, linewidth=0.15
)

# Plot the intersection line L
W_L = np.linspace(-0.35, 0.45, 100)
ax.plot(
    np.zeros_like(W_L),
    np.zeros_like(W_L),
    W_L,
    color="black",
    linewidth=2,
    linestyle="--",
    zorder=20
)
ax.text(0.025, 0.03, 0.39, r"$L$", fontsize=15)

# Plot the horizontal blow-up cylinder S^2_+ x R_V
# This is shown schematically with V as the axial coordinate
radius = 0.2
# Show biologically relevant half-cylinder U >= 0
theta = np.linspace(-np.pi / 2, np.pi / 2, 80)
V_cyl = np.linspace(0, 0.85, 90)
Theta, Vc = np.meshgrid(theta, V_cyl)
Uc = radius * np.cos(Theta)
Wc = radius * np.sin(Theta)

ax.plot_surface(
    Uc, Vc, Wc,
    alpha=0.28,
    color=green,
    edgecolor=green,
    linewidth=0.15,
    zorder=10
)

# Central axis of blown-up cylinder V_cyl
ax.plot(
    np.zeros_like(V_cyl),
    V_cyl,
    np.zeros_like(V_cyl),
    color="black",
    linewidth=2,
    zorder=25
)

# Mark the K2 chart
ax.text(0.14, 0.55, 0.11, r"$K_2:\ U= O(\varepsilon)$",
        fontsize=14, color=green, fontweight="bold")

# Plot Fisher-KPP orbit in K1 chart from (1,0,0) toward (0,0,0)
s = np.linspace(0, 1, 220)
U_outer = 1 - s
V_outer = np.zeros_like(s)
W_outer = -0.30 * np.sin(np.pi * s) * (1 - 0.15 * s)

# Stop the K1 orbit when it collides with cylinder
mask = U_outer >= 0.75 * radius
ax.plot(
    U_outer[mask],
    V_outer[mask],
    W_outer[mask],
    color=red,
    linewidth=3.0,
    zorder=30
)

# Plot an arrow on K1 orbit
i = 95
ax.scatter(
    U_outer[i],
    V_outer[i],
    W_outer[i],
    marker=">",
    s=90,
    color=red,
    depthshade=False,
    zorder=40
)

ax.text(0.67, 0.03, -0.07, r"$K_1:\ U\gg\varepsilon$",
        fontsize=14, color=red, fontweight="bold")

ax.text(0.78, 0.003, -0.15, r"$\Gamma_1$",
        fontsize=15, color=red, fontweight="bold")

# Plot the starting point P_1
ax.scatter(1, 0, 0, s=90, facecolors="white", edgecolors="black", linewidths=1.7, zorder=40)
ax.text(0.99, 0.045, 0.04, r"$\bar P_1$", fontsize=14, fontweight="bold")

# Endpoint of the plotted K1 orbit
U_entry = U_outer[mask][-1]
V_entry = V_outer[mask][-1]
W_entry = W_outer[mask][-1]

# Plot inner orbit Gamma_2 in K2
u = np.linspace(U_entry, 0.0, 100)
v = np.zeros_like(u)
w = (W_entry / U_entry) * u
ax.plot(
    u,
    v,
    w,
    color=purple,
    linewidth=3.0,
    zorder=35
)

ax.text(0.19, 0.035, 0.05, r"$\Gamma_2$",
        fontsize=15, color=purple, fontweight="bold")

# Mark endpoint Q2 of the trajectory
ax.scatter(0, 0, 0, s=90, color="black", zorder=45)
ax.text(0.035, 0.04, 0.05, r"$Q_2$", fontsize=15, fontweight="bold")

# Labels for the critical-manifold sheets
ax.text(0.70, 0.02, 0.35, r"$M_{0V}$", fontsize=15, color=blue, fontweight="bold")
ax.text(0.03, 0.58, 0.35, r"$M_{0U}$", fontsize=15, color=orange, fontweight="bold")

# Axes labels
ax.set_xlabel(r"$U$", fontsize=13, labelpad=10)
ax.set_ylabel(r"$V$", fontsize=13, labelpad=10)
ax.set_zlabel(r"$W$", fontsize=13, labelpad=10)

# Axes limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.9)
ax.set_zlim(-0.35, 0.45)

# View angle
ax.view_init(elev=20, azim=45)

# legend_elements = [
#     Patch(facecolor=green, edgecolor="gray", alpha=0.35,
#           label=r"Blown-up cylinder $S^2_+\times\mathbb R_V$"),
#     Patch(facecolor=blue, edgecolor="gray", alpha=0.20,
#           label=r"$\mathcal M_{0V}=\{V=0\}$"),
#     Patch(facecolor=orange, edgecolor="gray", alpha=0.20,
#           label=r"$\mathcal M_{0U}=\{U=0\}$"),
#     Line2D([0], [0], color="black", lw=2.5, ls="--",
#            label=r"Nonhyperbolic line $\mathcal L$"),
#     Line2D([0], [0], color=red, lw=3,
#            label=r"Outer orbit in $K_1$"),
#     Line2D([0], [0], color=purple, lw=3,
#            label=r"Inner orbit $\Gamma_2$ in $K_2$")
# ]

# ax.legend(
#     handles=legend_elements,
#     loc="upper right",
#     bbox_to_anchor=(0.98, 0.98),
#     fontsize=9,
#     frameon=True,
#     facecolor="white",
#     edgecolor="#EAEAEA"
# )

plt.tight_layout()
plt.savefig("blowup_cylinder_schematic.pdf", bbox_inches="tight", dpi=300)
plt.show()