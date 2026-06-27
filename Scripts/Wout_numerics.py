###
# Numerical computation of the exit boundary Wtilde_out as eta -> 0.
###
import numpy as np
from scipy.integrate import solve_ivp


def K2_system(z, y, c, r2):
    '''
    Full desingularised vector field in the K2 chart.
    '''
    u, v, w = y
    du = w
    dv = (u * v) / c
    dw = ((u*v*w)/c - c*w - u*(1 - r2*u - v)) / (1 - v)
    return [du, dv, dw]


def stable_eigvec(c, Vinf):
    '''
    Compute the strong stable eigendirection at (0,V_inf,0)
    '''
    # From the analytical eigenvalue calculations, the strong stable eigenvector is (1, lam_strong)^T.
    C = c / (2 * (1 - Vinf))

    if C**2 - 1 <= 0:
        raise ValueError("No real stable eigendirection for these parameters.")

    lam_strong = -C - np.sqrt(C**2 - 1)

    return lam_strong, np.array([1.0, lam_strong])


def compute_Wout(eta, r2, Vinf, delta=1e-6, z_back=500):
    '''
    Find Wout for a given eta.
    '''
    c = 2 - eta**2

    lam_strong, vec = stable_eigvec(c, Vinf)
    if vec is None:
        return None

    # Normalise so that delta controls the perturbation size
    vec = vec / np.linalg.norm(vec)

    # Initial condition near the equilibrium (0,Vinf,0),
    # perturbed along the strong stable eigendirection.
    y0 = [delta * vec[0], Vinf, delta * vec[1]]

    # Reverse the vector field, so that integrating forward with solver_ivp
    # corresponds to integrating backwards in the K2 dynamics.
    def backward_vector_field(z, y):
        '''
        Reverse the vector field for backward integration.
        '''
        return [-f for f in K2_system(z, y, c, r2)]

    def hit_u1(z, y):
        '''
        Event function defining the matching section u2=1.
        '''
        return y[0] - 1.0

    # Stop the integration when matching section is reached.
    hit_u1.terminal = True

    # Find crossings with u_2 increasing through 1.
    hit_u1.direction = 1

    # Integrate the reversed K2 vector field until matching section is reached.
    sol = solve_ivp(
        backward_vector_field,
        (0, z_back),
        y0,
        events=hit_u1,
        max_step=0.05,
        rtol=1e-10,
        atol=1e-13,
    )

    # If the trajectory does not reach the matching section, nothing is returned.
    if len(sol.t_events[0]) == 0:
        return None

    # Find the intersection coordinates with the matching section.
    u_cross, v_cross, w_cross = sol.y_events[0][0]
    Wout = w_cross + 1.0
    return Wout, v_cross, w_cross

if __name__ == "__main__":

    # Take a small value of the radial coordinate.
    r2 = 1e-3
    Vinf = 0.3
    etas = [0.4, 0.3, 0.2, 0.1, 0.05, 0.025]

    # Print table header
    print(f"{'eta':>8} {'Wtilde_out':>14} {'Wtilde_out/eta':>18}")
    print("-" * 44)

    for eta in etas:
        result = compute_Wout(
            eta=eta,
            r2=r2,
            Vinf=Vinf,
            delta=1e-6,
            z_back=500,
        )

        if result is None:
            print(f"{eta:8.3f} {'failed':>14}")
            continue

        Wtilde_out, v_cross, w_cross = result

        # Print the results
        print(
            f"{eta:8.3f} "
            f"{Wtilde_out:14.6e} "
            f"{Wtilde_out / eta:18.6e}"
        )