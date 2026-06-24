###
# Hamiltonian curves W(1-V) = A for the layer problem fast subsystem.
###
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.serif": "serif",
})

fig, ax = plt.subplots(figsize=(7, 5))
V_range = np.linspace(0, 0.95, 500)

# Plot curves for different values of A
A_values = [-0.05, -0.1, -0.2, -0.3, -0.4, -0.5]
#Define colours
colors = plt.cm.Blues(np.linspace(0.25, 0.95, len(A_values)))

#Loop through values of A and colours
for A, col in zip(A_values, colors):
    W_curve = A / (1 - V_range)
    # Only plot in biologically relevant region W < 0
    mask = W_curve < 0
    ax.plot(V_range[mask], W_curve[mask],
           color=col, lw=2,
           label=f'$A = {A}$')

# Mark the critical manifold branches
ax.axvline(0, color='steelblue', lw=2, ls='--',label=r'$M_{0V}=\{V=0\}$')
ax.axhline(0, color='grey', lw=0.5, alpha=0.5)

#Mark the intersection line L projection on (V,W)
ax.plot(0,0,'ko',markersize=6)
ax.annotate(r'$(0,0)$', (0,0), xytext=(0.02,0.02))

ax.set_xlim(-0.05, 1.0)
ax.set_ylim(-1.5, 0.1)
ax.set_xlabel(r'$V$', fontsize=13)
ax.set_ylabel(r'$W$', fontsize=13)
#ax.set_title(r'Hamiltonian curves $W(1-V) = A$ of the layer problem', fontsize=12)
ax.legend(
    fontsize=9,
    loc='lower left',
    bbox_to_anchor=(0.10, 0.05)
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('conserved_curves.pdf', bbox_inches='tight', dpi=300)
plt.show()