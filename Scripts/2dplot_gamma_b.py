###
# Cross-section of the matching region in the plane V=0.
###
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.serif": "serif",
})

# Green cylinder radius
radius = 0.2

# Endpoints on the blown-up boundary circle
theta_Q1 = np.deg2rad(-25)
theta_Pm = np.deg2rad(-45)

Q1 = np.array([
    radius * np.cos(theta_Q1),
    radius * np.sin(theta_Q1)
])

Pm = np.array([
    radius * np.cos(theta_Pm),
    radius * np.sin(theta_Pm)
])

# Boundary orbit Gamma_b
theta_b = np.linspace(theta_Q1, theta_Pm, 100)
U_b = radius * np.cos(theta_b)
W_b = radius * np.sin(theta_b)

# Inner orbit Gamma_2
u = np.linspace(Pm[0], 0.0, 120)
w = -u

# Outer orbit Gamma_1
t_red = np.linspace(0, 1, 80)
U_red = np.linspace(0.30, Q1[0], 80)
W_red = np.linspace(-0.28, Q1[1], 80)
W_red += 0.05 * np.sin(np.pi * t_red)

fig, ax = plt.subplots(figsize=(6.5, 6))

green = "#55A868"
red = "#C44E52"
purple = "#8172B3"

# Blown-up boundary circle
circle = plt.Circle(
    (0, 0),
    radius,
    fill=False,
    color=green,
    lw=2.2
)
ax.add_patch(circle)

# Gamma_1
ax.plot(U_red, W_red, color=red, lw=2.8)

nr = 45
ax.annotate(
    "",
    xy=(U_red[nr+5], W_red[nr+5]),
    xytext=(U_red[nr-5], W_red[nr-5]),
    arrowprops=dict(
        arrowstyle="-|>",
        color=red,
        lw=1.8,
        mutation_scale=20
    )
)

# Gamma_b
ax.plot(U_b, W_b, color="black", lw=2.8)

nb = len(U_b)//2
ax.annotate(
    "",
    xy=(U_b[nb+5], W_b[nb+5]),
    xytext=(U_b[nb-5], W_b[nb-5]),
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        lw=1.8,
        mutation_scale=20
    )
)

# Gamma_2
ax.plot(u, w, color=purple, lw=2.8)

ng = 45
ax.annotate(
    "",
    xy=(u[ng+5], w[ng+5]),
    xytext=(u[ng-5], w[ng-5]),
    arrowprops=dict(
        arrowstyle="-|>",
        color=purple,
        lw=1.8,
        mutation_scale=20
    )
)

# Points
ax.scatter(Q1[0], Q1[1], s=70, color="black", zorder=10)

ax.scatter(
    Pm[0],
    Pm[1],
    s=70,
    facecolors="white",
    edgecolors="black",
    linewidths=1.5,
    zorder=10
)

ax.scatter(
    0,
    0,
    s=70,
    color="black",
    zorder=10
)

# Point labels
ax.text(
    Q1[0]-0.02,
    Q1[1]+0.01,
    r"$\bar Q_1$",
    fontsize=12
)

ax.text(
    Pm[0]+0.01,
    Pm[1]-0.01,
    r"$P_1^{\mathrm{out}}\sim P_2^{\mathrm{in}}$",
    fontsize=12
)

ax.text(
    0.007,
    0.01,
    r"$Q_2$",
    fontsize=12
)

# Orbit labels
ax.text(
    0.26,
    -0.25,
    r"$\Gamma_1$",
    fontsize=12,
    color=red
)

ax.text(
    (Q1[0]+Pm[0])/2 - 0.019,
    (Q1[1]+Pm[1])/2 + 0.01,
    r"$\Gamma_b$",
    fontsize=12
)

ax.text(
    0.06,
    -0.10,
    r"$\Gamma_2$",
    fontsize=12,
    color=purple
)

# Boundary label
ax.text(
    0.16,
    0.02,
    r"$r_1=0$",
    fontsize=12,
    color=green
)

# Formatting
ax.set_xlabel(r"$U$")
ax.set_ylabel(r"$W$")
ax.set_xlim(-0.04, 0.32)
ax.set_ylim(-0.32, 0.05)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(
    "boundary_matching_geometry.pdf",
    bbox_inches="tight",
    dpi=300
)

plt.show()