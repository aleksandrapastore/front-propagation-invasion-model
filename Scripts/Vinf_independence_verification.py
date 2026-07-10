###
# Numerical verification of the two-component travelling-wave PDE with Heaviside initial conditions.
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

# Vinf sweep colours
COLOR_VINF_01 = "C3"   # red
COLOR_VINF_03 = "C4"   # purple
COLOR_VINF_06 = "C5"   # brown


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


def front_position(x, u, level=0.5):
    '''
    Position of the travelling front.
    '''
    # Find grid intervals where u crosses chosen level u=0.5 from above.
    idx = np.where((u[:-1] >= level) & (u[1:] < level))[0]

    if len(idx) == 0:
        return np.nan
    
    # Use the first crossing
    i = idx[0]

    # Linear interpolation between two neighbouring grid points
    return x[i] + (level - u[i]) * (x[i + 1] - x[i]) / (u[i + 1] - u[i])


def estimate_speed(times, positions, discard_fraction=0.7):
    '''
    Estimate the travelling-wave speed from the computed front position.
    x_f(t) = ct + k_0 log(t) + k_1
    '''

    # Exclude any invalid front positions or t=0.
    valid = np.isfinite(positions) & (times > 0)
    t = times[valid]
    x = positions[valid]

    # Discard early-time transient
    start = int(discard_fraction * len(t))
    t = t[start:]
    x = x[start:]

    # Least-squares fitting matrix
    A = np.column_stack([t, np.log(t), np.ones_like(t)])

    # Find the best-fit coefficients
    coeff, *_ = np.linalg.lstsq(A, x, rcond=None)

    # Return fitted wave speed and corrections
    # c, k0, k1
    return coeff[0], coeff[1], coeff[2]

def c_pi2(gamma):
    '''
    Leading-order prediction obtained in the dissertation.
    '''
    return 2 - np.pi**2 / np.log(gamma)**2


def c_pi2_over_4(gamma):
    '''
    Leading-order prediction with coefficient 1/4.
    '''
    return 2 - (np.pi**2 / 4.0) / np.log(gamma)**2


def simulate(gamma=50.0, Vinf=0.3, L=400.0, nx=2001, dt=0.02, T=120.0, save_times=None,):
    '''
    Solve the two-component invasion PDE using a Crank--Nicolson
    discretisation for the diffusion term and an explicit treatment
    of the reaction terms.
    '''

    # Spatial mesh grid
    x = np.linspace(-L/2, L/2, nx)
    dx = x[1] - x[0]
    n = len(x)

    # Heaviside profiles
    x0 = -L/4
    u = np.where(x < x0, 1.0, 0.0)
    v = np.where(x < x0, 0.0, Vinf)

    # Identity matrix for the implicit solve
    I = identity(n, format="csr")

    # Storage for the front position
    times = []
    positions = []

    # Store selected solution profiles for plotting.
    profiles = {}

    if save_times is not None:
        save_steps = {int(t / dt): t for t in save_times}

    nsteps = int(T/dt)

    for step in range(nsteps + 1):

        t = step*dt

        # Save solution profile at selected times.
        if save_times is not None and step in save_steps:
            profiles[save_steps[step]] = (u.copy(), v.copy())

        # Record the front position every few time steps
        if step % 20 == 0:
            times.append(t)
            positions.append(front_position(x, u))

        # Construct the diffusion matrix
        A = diffusion_matrix(v, dx)

        # Reaction term
        reaction = u*(1-u-v)

        # Crank-Nicolson update for diffusion
        lhs = I - 0.5*dt*A
        rhs = (I + 0.5*dt*A).dot(u) + dt*reaction

        u_new = spsolve(lhs, rhs)
        u_new = np.clip(u_new, 0.0, 1.2)

        # Exact update for the resident population
        v_new = v*np.exp(-dt*gamma*u_new)
        v_new = np.clip(v_new, 0.0, Vinf)

        u = u_new
        v = v_new

    return x, u, v, np.array(times), np.array(positions), profiles


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


if __name__ == "__main__":

    # Validating independence from far-field values
    Vinf_values = [0.1, 0.3, 0.6]
    
    gammas = [50, 100, 200, 500, 1000, 3000, 5000, 8000, 10000]
    results = []

    # Running for different Vinf values
    plt.figure(figsize=(7,4.5))
    colors = [COLOR_VINF_01, COLOR_VINF_03, COLOR_VINF_06]
    for colour, Vinf in zip(colors, Vinf_values):    
        print(f"\nRunning Vinf = {Vinf}")
        results = []

        for gamma in gammas:
            print(f"\nRunning gamma = {gamma}")
            # Choose a sufficiently large domain for this gamma
            L, T, nx, dt = domain_and_time_for_gamma(gamma)
            # Solve the PDE
            x, u, v, times, positions, profiles = simulate(gamma=gamma, Vinf=Vinf, L=L, nx=nx, dt=dt, T=T,)

            # Estimate the travelling-wave speed
            c_num, k0, k1 = estimate_speed(times, positions, discard_fraction=0.75)
            # Compute the fitted asymptotic coefficient
            A_num = (2 - c_num)*np.log(gamma)**2

            # Leading-order asymptotic predictions for the wave speed
            c_th_pi2 = c_pi2(gamma)
            c_th_pi2_4 = c_pi2_over_4(gamma)

            # Absolute errors relative to the two asymptotic predictions
            E_pi2 = abs(c_num - c_th_pi2)
            E_pi2_4 = abs(c_num - c_th_pi2_4)

            results.append((gamma, c_num, A_num, c_th_pi2, c_th_pi2_4, E_pi2, E_pi2_4))

            print(f"c_num = {c_num:.6f}")
            print(f"A_num = {A_num:.6f}")
            print(f"pi^2  = {np.pi**2:.4f}")
            print(f"pi^2/4 = {np.pi**2/4:.4f}")
            print(f"E_pi2 = {E_pi2:.6e}")
            print(f"E_pi2_4 = {E_pi2_4:.6e}")

        results = np.array(results)
        plt.plot(results[:,0], results[:,2], "o-", color=colour, lw=2.5, ms=6, label=rf"$V_\infty={Vinf}$",)

    # Plot the numerical coefficient A_{num} for a range of Vinf.
    plt.axhline(np.pi**2, color="k", ls="--", lw=1.5, label=r"$\pi^2$",)
    plt.xscale("log")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$A_{\mathrm{num}}$")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("pde_Vinf_sweep.pdf", dpi=300, bbox_inches="tight")
    plt.show()