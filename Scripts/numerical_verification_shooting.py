###
# Numerical verification of the two-component travelling-wave system using the shooting algorithm.
###

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

def TW_rhs(z, y, gamma, c):
    '''
    Travelling-wave ODEs corresponding to the two-component system.
    '''
    # Define the state variables.
    U, V, W = y
    # Restrict 0<=V<1 to avoid singularity in the W-equation denominator. 
    V = min(max(V, 0.0), 0.999999)
    dU = W
    dV = (gamma / c)*U*V
    dW = ((gamma / c)*U*V*W - c*W -U*(1-U-V))/(1-V)
    return [dU, dV, dW]

def unstable_initial_condition(c, gamma, eps, theta):
    '''
    Define the initial condition near (1,0,0) on the 2D unstable manifold with theta
    being the parameter of the eigendirection.
    '''
    lam_u = (-c + np.sqrt(c**2 + 4.0)) / 2.0

    # eigendirection of gamma/c
    E1 = np.array([0.0, 1.0, 0.0])
    # eigendirection leaving U=1 toward lower U
    E2 = np.array([-1.0, 0.0, -lam_u])

    Qminus = np.array([1.0, 0.0, 0.0])

    direction = np.cos(theta) * E1 + np.sin(theta) * E2
    return Qminus + eps * direction

def shooting_residual(params, gamma, Vinf, eps, z_end):
    '''
    Computes the shooting residual for the travelling-wave problem.
    '''
    # Shooting parameters
    c, theta = params
    # Perturb the initial condition
    y0 = unstable_initial_condition(c, gamma, eps, theta)
    # Integrate the travelling-wave system forward
    sol = solve_ivp(
        TW_rhs,
        (0, z_end),
        y0,
        args=(gamma, c),
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )
    # Return a large residual if the integration fails.
    if not sol.success:
        return [1e3, 1e3]
    # Find the final point of the computed trajectory.
    U, V, W = sol.y[:, -1]
    # Local stable eigenspace condition near (0,Vinf,0)
    C = c / (2 * (1 - Vinf))
    if C < 1:
        return [1e3, 1e3]
    lam_slow = (-c + np.sqrt(c**2 - 4 * (1 - Vinf)**2)) / (2 * (1 - Vinf))
    # Residual measuring the difference with equilibrium (0, V_inf, 0).
    return [V - Vinf, W - lam_slow * U]

def solve_speed(gamma, Vinf, c_guess, theta_guess, eps=1e-6):
    '''
    Compute the travelling-wave speed by minimising the residual.
    '''
    # Estimate the integration interval
    z_end = 80
    # Define the minimal wave speed
    c_min = 2.0 * (1.0 - Vinf)
    # Solve the shooting problem for the unknown parameters c, vo
    out = least_squares(shooting_residual, [c_guess, theta_guess],
        args=(gamma, Vinf, eps, z_end),
        bounds=(
            [c_min * 1.001, -np.pi],
            [2.0 - 1e-6, np.pi],
        ), xtol=1e-12, ftol=1e-12,gtol=1e-12,)

    # Check if we converged to a small shooting residual.
    if np.linalg.norm(out.fun) > 1e-4:
        print(
            f"  gamma={gamma}: "
            f"did not converge, residual={out.fun}"
        )
        return None, None
    
    c, theta = out.x
    return c, theta

def c_asym(g):
    '''
    Leading-order asymptotic prediction for the travelling-wave speed.
    '''
    return 2.0 - np.pi**2 / np.log(g)**2


if __name__ == "__main__":
    # Far-field resident-tissue density V_infty
    Vinf = 0.3
    # Values of gamma to try
    gammas = [50, 100, 300, 1000, 3000, 1e4, 1e5]
    # Initial guesses for the first shooting solve
    c_guess = 1.9
    theta_guess = 0.9 * np.pi / 2

    results = []

    for g in gammas:

        # Solve the shooting problem for this value of gamma
        c, theta = solve_speed(g, Vinf, c_guess, theta_guess)
        # Skip failed solves
        if c is None:
            continue
        # Print numerical and asymptotic speeds for comparison
        print(
            f"gamma={g:8.3g}  "
            f"c={c:.6f}  "
            f"c_pred={c_asym(g):.6f}"
        )
        # Store successful result
        results.append((g, c))

        # Start the next solve using the current solution
        c_guess = c
        theta_guess = theta

    if len(results) == 0:
        raise RuntimeError("No successful shooting runs.")

    results = np.array(results)

    gamma_vals = results[:, 0]
    c_vals = results[:, 1]

    # Plot numerical speed against asymptotic prediction.
    plt.figure(figsize=(6, 4))
    plt.plot(gamma_vals, c_vals, "o-", label="numerical")
    plt.plot(
        gamma_vals,
        [c_asym(g) for g in gamma_vals],
        "s--",
        label="asymptotic",
    )
    # Large-gamma Fisher--KPP limiting speed
    plt.axhline(2, color="k", lw=0.7, label=r"$c=2$")

    # Local minimum-speed threshold at the far-field equilibrium.
    plt.axhline(
        2 * (1 - Vinf),
        color="gray",
        ls=":",
        label=r"$2(1-V_\infty)$",
    )

    plt.xscale("log")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$c$")
    plt.legend()
    plt.tight_layout()

    # plt.savefig("shooting_basic.pdf", bbox_inches="tight", dpi=300)
    plt.show()