###
# Numerical verification of the selected two-component travelling-wave
# matching problem using a shooting algorithm.
###

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.serif": "serif",
})


def TW_rhs(z, y, gamma, c):
    '''
    Travelling-wave ODEs using the flux variable W=(1-V)dU/dz.
    '''
    U, V, W = y

    # Avoid numerical singularity at V=1.
    V = min(max(V, 0.0), 0.999999)

    dU = W / (1.0 - V)
    dV = (gamma / c) * U * V
    dW = -c * W / (1.0 - V) - U * (1.0 - U - V)

    return [dU, dV, dW]


def unstable_initial_condition(c, gamma, eps0, theta):
    '''
    Initial condition near the invaded equilibrium (1,0,0)
    on the two-dimensional unstable manifold.
    '''
    lam_u = (-c + np.sqrt(c**2 + 4.0)) / 2.0

    # Transverse resident-tissue direction.
    E1 = np.array([0.0, 1.0, 0.0])

    # Fisher--KPP unstable direction leaving U=1.
    E2 = np.array([-1.0, 0.0, -lam_u])

    Qminus = np.array([1.0, 0.0, 0.0])

    direction = np.cos(theta) * E1 + np.sin(theta) * E2

    return Qminus + eps0 * direction


def make_event_U_equals_eps(gamma):
    eps = 1.0 / gamma

    def event(z, y, gamma_arg=None, c_arg=None):
        U, V, W = y
        return U - eps

    event.terminal = True
    event.direction = -1

    return event


def selected_matching_residual(params, gamma, eps0=1e-7, z_end=300.0):
    '''
    Shooting residual for the selected Q2-matching problem.

    We integrate from the invaded equilibrium until U=epsilon.
    At this section, the K2 matching condition is

        u2 = 1,   w2 = -1,

    where W = epsilon*w2. Hence W/epsilon + 1 = 0.

    For the selected Q2-type matching, we also require V to be small
    at this section.
    '''
    c, theta = params

    if c <= 0 or c >= 2:
        return [1e3, 1e3]

    eps = 1.0 / gamma

    y0 = unstable_initial_condition(c, gamma, eps0, theta)

    sol = solve_ivp(
        TW_rhs,
        (0.0, z_end),
        y0,
        args=(gamma, c),
        method="BDF",
        events=make_event_U_equals_eps(gamma),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
    )

    if (not sol.success) or len(sol.t_events[0]) == 0:
        return [1e3, 1e3]

    Ue, Ve, We = sol.y_events[0][0]

    # K2 variables at U=epsilon:
    # u2 = U/epsilon = 1,
    # w2 = W/epsilon.
    w2 = We / eps

    return [
        w2 + 1.0,
        Ve
    ]


def solve_selected_speed(gamma, c_guess, theta_guess, eps0=1e-7):
    '''
    Solve the selected matching problem for a given gamma.
    '''
    out = least_squares(
        selected_matching_residual,
        [c_guess, theta_guess],
        args=(gamma, eps0),
        bounds=(
            [0.2, -np.pi],
            [2.0 - 1e-8, np.pi],
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )

    c, theta = out.x
    resnorm = np.linalg.norm(out.fun)

    return c, theta, resnorm, out.fun


def compute_matching_point(gamma, c, theta, eps0=1e-7, z_end=300.0):
    '''
    Reintegrate the selected solution and return the matching point
    at U=epsilon.
    '''
    eps = 1.0 / gamma

    y0 = unstable_initial_condition(c, gamma, eps0, theta)

    sol = solve_ivp(
        TW_rhs,
        (0.0, z_end),
        y0,
        args=(gamma, c),
        method="BDF",
        events=make_event_U_equals_eps(gamma),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
    )

    if (not sol.success) or len(sol.t_events[0]) == 0:
        return None

    Ue, Ve, We = sol.y_events[0][0]
    eps = 1.0 / gamma

    u2 = Ue / eps
    v2 = Ve
    w2 = We / eps

    return Ue, Ve, We, u2, v2, w2


def c_pi2(gamma):
    '''
    Full-jump prediction.
    '''
    return 2.0 - np.pi**2 / np.log(gamma)**2


def c_pi2_over_4(gamma):
    '''
    Half-jump prediction.
    '''
    return 2.0 - (np.pi**2 / 4.0) / np.log(gamma)**2


if __name__ == "__main__":

    gammas = [50, 100, 300, 1000, 3000, 10000]

    c_guess = 1.8
    theta_guess = 1.0

    results = []

    for gamma in gammas:

        print(f"\nRunning gamma = {gamma}")

        c, theta, resnorm, residual = solve_selected_speed(
            gamma,
            c_guess,
            theta_guess,
            eps0=1e-7,
        )

        match = compute_matching_point(gamma, c, theta, eps0=1e-7)

        if match is None:
            print("  Matching point not found.")
            continue

        Ue, Ve, We, u2, v2, w2 = match

        A_fit = (2.0 - c) * np.log(gamma)**2

        E_pi2 = abs(A_fit / np.pi**2 - 1.0)
        E_pi2_4 = abs(4.0 * A_fit / np.pi**2 - 1.0)

        results.append((gamma, c, theta, A_fit, E_pi2, E_pi2_4, Ve, We, w2, resnorm))

        print(f"  c = {c:.10f}")
        print(f"  theta = {theta:.10f}")
        print(f"  residual = {residual}")
        print(f"  residual norm = {resnorm:.3e}")
        print(f"  A_fit = {A_fit:.10f}")
        print(f"  E_pi2 = {E_pi2:.3e}")
        print(f"  E_pi2_4 = {E_pi2_4:.3e}")
        print(f"  At U=epsilon:")
        print(f"    V = {Ve:.3e}")
        print(f"    W = {We:.3e}")
        print(f"    w2 = W/epsilon = {w2:.10f}")

        c_guess = c
        theta_guess = theta

    results = np.array(results)

    gamma_vals = results[:, 0]
    c_vals = results[:, 1]
    A_vals = results[:, 3]
    E_pi2_vals = results[:, 4]
    E_pi2_4_vals = results[:, 5]
    V_match_vals = results[:, 6]
    W_match_vals = results[:, 7]
    w2_match_vals = results[:, 8]

    # Plot numerical speed against both asymptotic predictions.
    plt.figure(figsize=(7, 4))

    plt.plot(gamma_vals, c_vals, "o-", label=r"$c_{\mathrm{num}}$")
    plt.plot(gamma_vals, [c_pi2(g) for g in gamma_vals], "s--",
             label=r"$2-\pi^2/(\log\gamma)^2$")
    plt.plot(gamma_vals, [c_pi2_over_4(g) for g in gamma_vals], "d--",
             label=r"$2-\pi^2/[4(\log\gamma)^2]$")
    plt.axhline(2.0, color="k", lw=0.8, label=r"$2$")

    plt.xscale("log")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$c$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("selected_speed_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot fitted asymptotic coefficient.
    plt.figure(figsize=(7, 4))

    plt.plot(gamma_vals, A_vals, "o-", label=r"$A=(2-c)(\log\gamma)^2$")
    plt.axhline(np.pi**2, linestyle="--", label=r"$\pi^2$")
    plt.axhline(np.pi**2 / 4.0, linestyle=":", label=r"$\pi^2/4$")

    plt.xscale("log")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$A$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("selected_Afit_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot relative errors for the two candidate laws.
    plt.figure(figsize=(7, 4))

    plt.loglog(gamma_vals, E_pi2_vals, "o-", label=r"$E_{\pi^2}$")
    plt.loglog(gamma_vals, E_pi2_4_vals, "s--", label=r"$E_{\pi^2/4}$")

    plt.xlabel(r"$\gamma$")
    plt.ylabel("relative error")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("selected_error_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot V at the matching section U=epsilon.
    plt.figure(figsize=(7, 4))

    plt.loglog(gamma_vals, V_match_vals, "o-", label=r"$V$ at $U=\varepsilon$")

    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$V_{\mathrm{match}}$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("selected_Vmatch.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot w2 at the matching section.
    plt.figure(figsize=(7, 4))

    plt.semilogx(gamma_vals, w2_match_vals, "o-", label=r"$w_2=W/\varepsilon$")
    plt.axhline(-1.0, color="k", linestyle="--", label=r"$w_2=-1$")

    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$w_2$ at $U=\varepsilon$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("selected_w2match.pdf", dpi=300, bbox_inches="tight")
    plt.show()