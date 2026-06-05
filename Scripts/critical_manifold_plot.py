###
# Critical manifold and equilibria of the two-component travelling-wave ODE.
###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

#Create the 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#Define the coordinate ranges for each axis
U_range = np.linspace(0, 1, 50)
V_range = np.linspace(0, 0.9, 50)
W_range = np.linspace(-0.3, 0.5, 50)

#Plot branch M_0V = {V=0} which is the (U,W)-plane
U_grid, W_grid = np.meshgrid(U_range, W_range)
V_grid = np.zeros_like(U_grid)
ax.plot_surface(U_grid, V_grid, W_grid, alpha=0.15, color='#D3D3D3', edgecolor="#5C5C5C", lw=0.3, zorder=1)

#Plot branch M_0U = {U=0} which is the (V,W)-plane
V_grid2, W_grid2 = np.meshgrid(V_range, W_range)
U_grid2 = np.zeros_like(V_grid2)
ax.plot_surface(U_grid2, V_grid2, W_grid2, alpha=0.35, color='#808080', edgecolor='none', zorder=2)

#Plot the saddle equilibrium point (1,0,0)
ax.scatter(1, 0, 0, s=90, facecolors='white', edgecolors='black', linewidths=1.5, zorder=10)
ax.text(1.05, 0, 0.04, r'$(1,0,0)$', fontsize=11, zorder=6)

#Plot the equilibrium family (0, V_inf, 0)
V_inf_range = np.linspace(0, 0.9, 100)
ax.plot(np.zeros_like(V_inf_range), V_inf_range, np.zeros_like(V_inf_range), color='black', lw=3.5, zorder=4)
ax.text(0.02, 0.5, 0.05, r'$(0,V_\infty,0)$', fontsize=11, zorder=6)

#Plot the intersection line between critical manifold branches (0,0,W)
W_int = np.linspace(-0.3, 0.5, 100)
ax.plot(np.zeros_like(W_int), np.zeros_like(W_int), W_int, color='black', lw=2.5, ls='--', zorder=3)

#Label the axes and the plot
ax.set_xlabel(r'$U$', fontsize=13, labelpad=10)
ax.set_ylabel(r'$V$', fontsize=13, labelpad=10)
ax.set_zlabel(r'$W$', fontsize=13, labelpad=10)
ax.set_title(r'Critical manifold $\mathcal{M}_0=\{UV=0\}$', fontsize=14, pad=15)

#Define axes limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.9)
ax.set_zlim(-0.3, 0.5)

#Choose the viewing angle
ax.view_init(elev=20, azim=45)

#Annotate the two branches of the critical manifold
ax.text(0.75, 0.02, 0.35, r'$\mathcal{M}_{0V} = \{V=0\}$', fontsize=11, zorder=6)
ax.text(0.02, 0.55, 0.35, r'$\mathcal{M}_{0U} = \{U=0\}$', fontsize=11, zorder=6)

#Create a legend
legend_elements = [
    Patch(facecolor='#D3D3D3', edgecolor='#5C5C5C', alpha=0.4, label=r'$\mathcal{M}_{0V} = \{V=0\}$'),
    Patch(facecolor='#808080', edgecolor='none', alpha=0.5, label=r'$\mathcal{M}_{0U} = \{U=0\}$'),
    Line2D([0], [0], color='black', lw=3.5, label=r'Equilibrium family $(0,V_\infty,0)$'),
    Line2D([0], [0], color='black', lw=2.5, ls='--', label=r'Intersection line $(0,0,W)$')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10, frameon=True, facecolor='white', edgecolor='#EAEAEA')

plt.tight_layout()
plt.savefig('critical_manifold.pdf', bbox_inches='tight', dpi=300)
plt.show()