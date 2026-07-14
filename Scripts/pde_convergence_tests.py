###
# Convergence analysis of the two-component travelling-wave PDE solver.
# The script tests convergence with respect to final simulation time, spatial mesh width and time step.
###

import numpy as np
import csv
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve


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


def simulate(gamma=50.0, Vinf=0.3, L=400, nx=2001, dt=0.02, T=120.0, save_times=None, record_every=20, x0=None,):
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
    if x0 is None:
        x0 = -L / 4
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

    # Number of time steps
    nsteps = int(T/dt)

    for step in range(nsteps + 1):

        t = step*dt

        # Save solution profile at selected times.
        if save_times is not None and step in save_steps:
            profiles[save_steps[step]] = (u.copy(), v.copy())

        # Record the front position every few time steps
        if step % record_every == 0:
            times.append(t)
            positions.append(front_position(x, u))

        if step == nsteps:
            break

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

def run_convergence_case(gamma, Vinf, L, T, dx, dt, discard_fraction=0.75,):
    '''
    Run a convergence test, return the numerical speed and rescaled asymptotic coefficient.
    '''
    nx = int(L / dx) + 1

    # Record front position every 0.4 time units
    record_every = max(1, int(round(0.4 / dt)))

    x, u, v, times, positions, _ = simulate(
        gamma=gamma,
        Vinf=Vinf,
        L=L,
        nx=nx,
        dt=dt,
        T=T,
        save_times=None,
        record_every=record_every,
    )

    c_num, k0, k1 = estimate_speed(times, positions, discard_fraction=discard_fraction,)

    # Calculate the numerically computed A_{num} coefficient
    A_num = (2 - c_num) * np.log(gamma)**2

    return {
        "L": L,
        "T": T,
        "nx": nx,
        "dx": dx,
        "dt": dt,
        "c_num": c_num,
        "A_num": A_num,
    }


if __name__ == "__main__":

    # Choose a large value of gamma and the specific value Vinf that we used.
    gamma = 3000
    Vinf = 0.3

    dx_base = 0.2
    dt_base = 0.02
    L0, T0, _, _ = domain_and_time_for_gamma(gamma,dx=dx_base,)

    # Store simulations so identical cases do not duplicate
    cache = {}
    table_rows = []

    def get_result(L, T, dx, dt):
        '''
        Run a case only if it has not already been computed.
        '''
        key = (
            round(L, 10),
            round(T, 10),
            round(dx, 10),
            round(dt, 10),
        )

        if key not in cache:
            print(
                f"\nRunning: T={T:.2f}, L={L:.2f}, "
                f"dx={dx:.3f}, dt={dt:.3f}"
            )
            cache[key] = run_convergence_case(gamma=gamma, Vinf=Vinf, L=L, T=T, dx=dx, dt=dt, discard_fraction=0.75,)
            print(
                f"c_num = {cache[key]['c_num']:.8f}, "
                f"A_num = {cache[key]['A_num']:.6f}"
            )
        else:
            print(
                f"\nReusing saved result: "
                f"T={T:.2f}, dx={dx:.3f}, dt={dt:.3f}"
            )

        return cache[key]

    # Test 1. Check final time convergence T.
    print("\nFINAL-TIME CONVERGENCE")

    time_factors = [1, 2, 4]
    for factor in time_factors:
        T = factor * T0
        # Increase the domain for the longer run while retaining same spatial resolution
        L = L0 + 2.5 * (T - T0)
        result = get_result(
            L=L,
            T=T,
            dx=dx_base,
            dt=dt_base,
        )
        table_rows.append({
            "test": "final time",
            "setting": f"{factor}T",
            "T_over_T0": factor,
            **result,
        })

    # Test 2. Check spatial convergence delta x.
    print("\nSPATIAL-GRID CONVERGENCE")
    dx_values = [0.4, 0.2, 0.1]
    for dx in dx_values:
        result = get_result(
            L=L0,
            T=T0,
            dx=dx,
            dt=dt_base,
        )
        table_rows.append({
            "test": "spatial grid",
            "setting": f"dx={dx}",
            "T_over_T0": 1,
            **result,
        })

    # Test 3. Check time convergence delta t.
    print("\nTIME-STEP CONVERGENCE")
    dt_values = [0.04, 0.02, 0.01, 0.005]
    for dt in dt_values:
        result = get_result(
            L=L0,
            T=T0,
            dx=dx_base,
            dt=dt,
        )
        table_rows.append({
            "test": "time step",
            "setting": f"dt={dt}",
            "T_over_T0": 1,
            **result,
        })

    #Save the results in a table
    header = ("test,setting,T_over_T0,L,T,nx,dx,dt,c_num,A_num")
    output = []
    for row in table_rows:
        output.append([
            row["test"],
            row["setting"],
            row["T_over_T0"],
            row["L"],
            row["T"],
            row["nx"],
            row["dx"],
            row["dt"],
            row["c_num"],
            row["A_num"],
        ])

    with open(
        "pde_convergence_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(header.split(","))
        writer.writerows(output)

    # Print a summary table of convergence.
    print("\nCONVERGENCE SUMMARY")
    print(
        f"{'Test':<16}"
        f"{'Setting':<12}"
        f"{'T/T0':>7}"
        f"{'dx':>9}"
        f"{'dt':>9}"
        f"{'c_num':>14}"
        f"{'A_num':>13}"
    )
    print("-" * 90)
    for row in table_rows:
        print(
            f"{row['test']:<16}"
            f"{row['setting']:<12}"
            f"{row['T_over_T0']:>7}"
            f"{row['dx']:>9.3f}"
            f"{row['dt']:>9.3f}"
            f"{row['c_num']:>14.8f}"
            f"{row['A_num']:>13.6f}"
        )
    print("\nSaved: pde_convergence_results.csv")