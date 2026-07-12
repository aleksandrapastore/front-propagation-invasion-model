###
# Schematic of the interstitial gap.
###


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
})


def diffusion_matrix(v, dx):
    '''
    Finite difference matrix for the diffusion term of PDE.
    '''
    # Define the number of spatial grid points
    n = len(v)
    # Diffusion coefficient
    D = 1 - v
    D_half = 0.5 * (D[:-1] + D[1:])

    # Three diagonals of the sparse matrix
    lower = D_half / dx**2
    upper = D_half / dx**2
    main = np.zeros(n)

    # Interior finite-difference stencil
    main[1:-1] = -(D_half[:-1] + D_half[1:]) / dx**2

    # Left Neumann boundary condition u_x = 0
    main[0] = -2 * D_half[0] / dx**2
    upper[0] = 2 * D_half[0] / dx**2
    # Right Neumann boundary condition u_x = 0
    main[-1] = -2 * D_half[-1] / dx**2
    lower[-1] = 2 * D_half[-1] / dx**2

    # Assemble the sparse tridiagonal matrix
    return diags([lower, main, upper], [-1, 0, 1], format="csr")


def simulate(gamma=50.0, Vinf=0.3, L=400.0, nx=2001, dt=0.02, T=120.0):
    '''
    Solve the two-component invasion PDE using a Crank--Nicolson
    discretisation for the diffusion term and an explicit treatment
    of the reaction terms.
    '''

    # Spatial mesh grid
    x = np.linspace(-L / 2, L / 2, nx)
    dx = x[1] - x[0]
    n = len(x)

    # Heaviside profiles
    x0 = -L / 4
    u = np.where(x < x0, 1.0, 0.0)
    v = np.where(x < x0, 0.0, Vinf)

    # Identity matrix for the implicit solve
    I = identity(n, format="csr")

    # Number of time steps
    nsteps = int(T / dt)

    for step in range(nsteps):
        # Construct the diffusion matrix
        A = diffusion_matrix(v, dx)
        # Reaction term
        reaction = u*(1-u-v)

        # Crank-Nicolson update for diffusion
        lhs = I - 0.5 * dt * A
        rhs = (I + 0.5 * dt * A).dot(u) + dt * reaction

        u_new = spsolve(lhs, rhs)
        u_new = np.clip(u_new, 0.0, 1.2)

        # Exact update for the resident population
        v_new = v * np.exp(-dt * gamma * u_new)
        v_new = np.clip(v_new, 0.0, Vinf)

        u = u_new
        v = v_new

    return x, u, v


def domain_and_time_for_gamma(gamma, dx=0.2):
    '''
    Choose the computational domain and final simulation time.
    '''
    scale = np.log(gamma) / np.log(50)
    L = 500 * scale
    T = 180 * scale
    # Keep approximately the same spatial mesh size
    nx = int(L / dx) + 1
    dt = 0.02
    return L, T, nx, dt


def crossing_position(x, y, level, direction="down"):
    '''
    Find threshold crossings by linear interpolation.
    '''
    if direction == "down":
        idx = np.where((y[:-1] >= level) & (y[1:] < level))[0]
    elif direction == "up":
        idx = np.where((y[:-1] <= level) & (y[1:] > level))[0]
    else:
        raise ValueError("direction must be 'up' or 'down'")

    if len(idx) == 0:
        raise ValueError(f"No crossing found.")

    # Use the first crossing
    i = idx[0]

    denominator = y[i+1]-y[i]

    if np.isclose(denominator, 0.0):
        raise ValueError("Cannot interpolate between equal neighbouring values.")

    return x[i]+(level-y[i])*(x[i+1]-x[i])/denominator


def gap_width(x, u, v, Vinf, theta_u=0.05, theta_v=0.05):
    '''
    Measure the numerical interstitial gap width.
    '''
    x_u = crossing_position(x, u, theta_u, direction="down")
    x_v = crossing_position(x, v, theta_v * Vinf, direction="up")
    G = x_v - x_u

    return G, x_u, x_v


if __name__ == "__main__":

    # Representative solution profile and gap edges
    profile_data = None

    # Far-field resident-tissue density
    Vinf = 0.3

    # Degradation rates used
    gammas = [5000]

    # Numerical threshold definitions of the gap
    theta_u = 0.05
    theta_v = 0.05

    gap_results = []

    for gamma in gammas:
        print(f"\nRunning gamma = {gamma}")

        L, T, nx, dt = domain_and_time_for_gamma(gamma)
        x, u, v = simulate(gamma=gamma, Vinf=Vinf, L=L, nx=nx, dt=dt, T=T,)
        G, x_u, x_v = gap_width(x, u, v, Vinf, theta_u=theta_u, theta_v=theta_v,)
        if G <= 0.0:
            print(f"No positive threshold-defined gap for gamma={gamma}: "f"x_u={x_u:.6f}, x_v={x_v:.6f}, G={G:.6f}")
            continue

        # Store one representative travelling-wave profile
        profile_data = {
            "x": x.copy(),
            "u": u.copy(),
            "v": v.copy(),
            "G": G,
            "x_u": x_u,
            "x_v": x_v,
        }

        gap_results.append((gamma, np.log(gamma), G, x_u, x_v))

        print(f"z_- = {x_u:.6f}")
        print(f"z_+ = {x_v:.6f}")
        print(f"G(gamma) = {G:.6f}")

    gap_results = np.array(gap_results)

    gamma_vals = gap_results[:, 0]
    log_gamma = gap_results[:, 1]
    gap_vals = gap_results[:, 2]


    # Representative profile showing how the gap is measured
    if profile_data is None:
        raise RuntimeError(f"No profile was stored for gamma={gamma}.")
    
    x_profile = profile_data["x"]
    u_profile = profile_data["u"]
    v_profile = profile_data["v"]
    G_profile = profile_data["G"]
    x_minus = profile_data["x_u"]
    x_plus = profile_data["x_v"]

    # Restrict the horizontal axis to the neighbourhood of the gap.
    left_padding = 8.0
    right_padding = 10.0
    x_left = x_minus - left_padding
    x_right = x_plus + right_padding

    mask = (x_profile >= x_left) & (x_profile <= x_right)

    # Produce a schematic figure of the interstitial gap.

    # Plot the representative profile 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    z_profile = x_profile
    ax.plot(z_profile[mask], u_profile[mask], lw=2.2, label=r"$U(z)$",)
    ax.plot(z_profile[mask], v_profile[mask], lw=2.2, label=r"$V(z)$",)
    # Shade the interstitial gap
    ax.axvspan(x_minus, x_plus, alpha=0.10, label=r"Interstitial gap",)
    # Mark the gap edges
    ax.axvline(x_minus, linestyle="--", linewidth=1.1, zorder=2)
    ax.axvline(x_plus, linestyle="--", linewidth=1.1, zorder=2)
    # Label the gap edges
    label_box = dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9,)
    ax.annotate(
        r"$z_{-}$",
        xy=(x_minus, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -5),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=13,
        bbox=label_box,
        clip_on=False,
        annotation_clip=False,
        zorder=20,
    )
    ax.annotate(
        r"$z_{+}$",
        xy=(x_plus, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -6),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=13,
        bbox=label_box,
        clip_on=False,
        annotation_clip=False,
        zorder=20,
    )

    # Draw a double arrow for the gap width
    arrow_height = 0.2
    ax.annotate(
        "",
        xy=(x_plus - 0.05, arrow_height),
        xytext=(x_minus + 0.05, arrow_height),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.4,
        ),
    )
    ax.text(
        0.5 * (x_minus + x_plus),
        arrow_height + 0.035,
        r"$G$",
        ha="center",
        va="bottom",
    )
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"Population density")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig("interstitial_gap_profile.pdf", bbox_inches="tight",)

    plt.show()
