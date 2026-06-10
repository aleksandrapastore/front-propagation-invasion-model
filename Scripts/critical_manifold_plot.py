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

blue = '#4C72B0'
orange = '#DD8452'

#Plot branch M_0V = {V=0} which is the (U,W)-plane
U_grid, W_grid = np.meshgrid(U_range, W_range)
V_grid = np.zeros_like(U_grid)
ax.plot_surface(U_grid, V_grid, W_grid, alpha=0.25, color=blue, edgecolor=blue, lw=0.25, zorder=1)

#Plot branch M_0U = {U=0} which is the (V,W)-plane
V_grid2, W_grid2 = np.meshgrid(V_range, W_range)
U_grid2 = np.zeros_like(V_grid2)
ax.plot_surface(U_grid2, V_grid2, W_grid2, alpha=0.25, color=orange, edgecolor=orange, lw=0.25, zorder=2)

#Plot the saddle equilibrium point (1,0,0)
ax.scatter(1, 0, 0, s=90, facecolors='white', edgecolors='black', linewidths=1.5, zorder=10)
ax.text(0.999, 0.05, 0.03, r'$(1,0,0)$', fontsize=14, fontweight='bold')

#Plot the equilibrium family (0, V_inf, 0)
V_inf_range = np.linspace(0, 0.9, 100)
ax.plot(np.zeros_like(V_inf_range), V_inf_range, np.zeros_like(V_inf_range), color='black', lw=3.5, zorder=4)
ax.text(0.03, 0.5, 0.06, r'$(0,V_\infty,0)$', fontsize=14, fontweight='bold')

#Plot the intersection line between critical manifold branches (0,0,W)
W_int = np.linspace(-0.3, 0.5, 100)
ax.plot(np.zeros_like(W_int), np.zeros_like(W_int), W_int, color='black', lw=4, ls='--', zorder=8)
ax.text(0.02, 0.04, 0.42, r'$L$', fontsize=14)

#Label the axes and the plot
ax.set_xlabel(r'$U$', fontsize=13, labelpad=10)
ax.set_ylabel(r'$V$', fontsize=13, labelpad=10)
ax.set_zlabel(r'$W$', fontsize=13, labelpad=10)
#ax.set_title(r'Critical manifold $\mathcal{M}_0=\{UV=0\}$', fontsize=14, pad=15)

#Define axes limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.9)
ax.set_zlim(-0.3, 0.5)

#Choose the viewing angle
ax.view_init(elev=20, azim=45)

#Annotate the two branches of the critical manifold
ax.text(0.72, 0.03, 0.35, r'$M_{0V}$', fontsize=16, fontweight='bold', color='black')
ax.text(0.03, 0.55, 0.35, r'$M_{0U}$', fontsize=16, fontweight='bold', color='black')


#Create a legend
legend_elements = [
    Patch(facecolor=blue, edgecolor='gray', alpha=0.35, label=r'$M_{0V}=\{V=0\}$'),
    Patch(facecolor=orange, edgecolor='gray', alpha=0.35, label=r'$M_{0U}=\{U=0\}$'),
    Line2D([0], [0], color='black', lw=3, label=r'Equilibrium family $(0,V_\infty,0)$'),
    Line2D([0], [0], color='black', lw=2, ls='--', label=r'Intersection line $L=\{(0,0,W)\}$')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10, frameon=True, facecolor='white', edgecolor='#EAEAEA')

plt.tight_layout()
plt.savefig('critical_manifold.pdf', bbox_inches='tight', dpi=300)
plt.show()