"""Interactive CPG parameter explorer using ipywidgets.

Provides an ``interactive_plot()`` function that renders sliders for every
CPG parameter and updates phase / magnitude time-series plots in real time.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, FloatLogSlider, FloatSlider, HBox, Output, Tab, VBox
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Simulation time grid
_T = np.arange(0, 20, 1e-3)
_N_OSC = 3  # number of oscillators in the demo


def _subscript(x: int) -> str:
    """Return *x* rendered as Unicode subscript digits (1-indexed)."""
    return "".join(chr(8272 + ord(c)) for c in str(x + 1))


def _simulate(
    intrinsic_freqs,
    intrinsic_amps,
    coupling_weights,
    phase_biases,
    convergence_coefs,
    init_phases,
    init_magnitudes,
):
    """Run a CPG simulation using ``scipy.integrate.solve_ivp``."""
    n = len(intrinsic_freqs)

    def ode_rhs(_, y, w, phi, nu, R, alpha):
        theta, r = y[:n], y[n:]
        dtheta_dt = 2 * np.pi * nu + r @ (
            w * np.sin(np.subtract.outer(theta, theta) - phi)
        )
        dr_dt = alpha * (R - r)
        return np.concatenate([dtheta_dt, dr_dt])

    y0 = np.concatenate([init_phases, init_magnitudes])
    args = (
        coupling_weights,
        phase_biases,
        intrinsic_freqs,
        intrinsic_amps,
        convergence_coefs,
    )
    sol = solve_ivp(ode_rhs, [_T[0], _T[-1]], y0, args=args, t_eval=_T)
    theta, r = sol.y[:n].T, sol.y[n:].T
    return theta, r


# ── Default parameter values ─────────────────────────────────────────────────

_DEFAULT_FREQS = np.ones(_N_OSC)
_DEFAULT_AMPS = np.array([1.0, 1.1, 1.2])
_DEFAULT_WEIGHTS = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
)
_DEFAULT_BIASES = np.deg2rad(
    np.array(
        [
            [0, 120, 0],
            [-120, 0, 120],
            [0, -120, 0],
        ]
    )
)
_DEFAULT_CONVERGENCE = np.ones(_N_OSC)

_rng = np.random.RandomState(0)
_DEFAULT_PHASES = _rng.rand(_N_OSC) * 2 * np.pi
_DEFAULT_MAGNITUDES = _rng.rand(_N_OSC) * _DEFAULT_AMPS


# ── Public API ───────────────────────────────────────────────────────────────


def interactive_plot():
    """Create and return an interactive widget for exploring CPG dynamics."""
    n = _N_OSC
    t = _T

    nu_sliders = [
        FloatLogSlider(
            value=_DEFAULT_FREQS[i],
            base=10,
            min=-2,
            max=2,
            step=0.1,
            description=f"ν{_subscript(i)}",
        )
        for i in range(n)
    ]
    R_sliders = [
        FloatSlider(
            value=_DEFAULT_AMPS[i], min=0, max=2, description=f"R{_subscript(i)}"
        )
        for i in range(n)
    ]
    w_sliders = [
        FloatSlider(
            value=_DEFAULT_WEIGHTS[i, j],
            min=0,
            max=2,
            description=f"w{_subscript(i)}{_subscript(j)}",
        )
        for i in range(n)
        for j in range(n)
    ]
    phi_sliders = [
        FloatSlider(
            value=np.rad2deg(_DEFAULT_BIASES[i, j]),
            min=-180,
            max=180,
            description=f"φ{_subscript(i)}{_subscript(j)}",
        )
        for i in range(n)
        for j in range(n)
    ]
    alpha_sliders = [
        FloatSlider(
            value=_DEFAULT_CONVERGENCE[i], min=0, max=2, description=f"α{_subscript(i)}"
        )
        for i in range(n)
    ]
    theta0_sliders = [
        FloatSlider(
            value=np.rad2deg(_DEFAULT_PHASES[i]),
            min=0,
            max=360,
            description=f"θ{_subscript(i)}(0)",
        )
        for i in range(n)
    ]
    r0_sliders = [
        FloatSlider(
            value=_DEFAULT_MAGNITUDES[i],
            min=0,
            max=1.5,
            description=f"r{_subscript(i)}(0)",
            step=0.01,
        )
        for i in range(n)
    ]

    tabs = {
        "intrinsic_freqs": nu_sliders,
        "intrinsic_amps": R_sliders,
        "coupling_weights": w_sliders,
        "phase_biases": phi_sliders,
        "convergence_coefs": alpha_sliders,
        "init_phases": theta0_sliders,
        "init_magnitudes": r0_sliders,
    }

    for key, sliders in tabs.items():
        if len(sliders) == 9:
            tabs[key] = VBox(
                [HBox([sliders[i * 3 + j] for j in range(3)]) for i in range(3)]
            )
        else:
            tabs[key] = VBox(sliders)

    tab = Tab()
    tab.children = list(tabs.values())
    tab.titles = list(tabs.keys())

    output = Output()

    with output:
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, tight_layout=True)

    theta, r = _simulate(
        _DEFAULT_FREQS,
        _DEFAULT_AMPS,
        _DEFAULT_WEIGHTS,
        _DEFAULT_BIASES,
        _DEFAULT_CONVERGENCE,
        _DEFAULT_PHASES,
        _DEFAULT_MAGNITUDES,
    )
    theta_lines = axs[0].plot(t, theta % (2 * np.pi), linewidth=1)
    axs[0].set_yticks([0, np.pi, 2 * np.pi])
    axs[0].set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    axs[0].set_ylabel("Phase")
    r_lines = axs[1].plot(t, r, linewidth=1)
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(min(s.min for s in r0_sliders), max(s.max for s in r0_sliders))

    def update(*args):
        intrinsic_freqs = np.array([slider.value for slider in nu_sliders])
        intrinsic_amps = np.array([slider.value for slider in R_sliders])
        coupling_weights = np.array([slider.value for slider in w_sliders]).reshape(
            n, n
        )
        phase_biases = np.deg2rad(
            np.array([slider.value for slider in phi_sliders])
        ).reshape(n, n)
        convergence_coefs = np.array([slider.value for slider in alpha_sliders])
        init_phases = np.deg2rad(np.array([slider.value for slider in theta0_sliders]))
        init_magnitudes = np.array([slider.value for slider in r0_sliders])
        theta, r = _simulate(
            intrinsic_freqs,
            intrinsic_amps,
            coupling_weights,
            phase_biases,
            convergence_coefs,
            init_phases,
            init_magnitudes,
        )
        for i, line in enumerate(theta_lines):
            line.set_ydata(theta.T[i] % (2 * np.pi))
        for i, line in enumerate(r_lines):
            line.set_ydata(r.T[i])
        fig.canvas.draw_idle()

    def reset(*args):
        for val, slider in zip(_DEFAULT_FREQS, nu_sliders):
            slider.value = val
        for val, slider in zip(_DEFAULT_AMPS, R_sliders):
            slider.value = val
        for val, slider in zip(_DEFAULT_WEIGHTS.ravel(), w_sliders):
            slider.value = val
        for val, slider in zip(_DEFAULT_BIASES.ravel(), phi_sliders):
            slider.value = np.rad2deg(val)
        for val, slider in zip(_DEFAULT_CONVERGENCE, alpha_sliders):
            slider.value = val
        for val, slider in zip(_DEFAULT_PHASES, theta0_sliders):
            slider.value = np.rad2deg(val)
        for val, slider in zip(_DEFAULT_MAGNITUDES, r0_sliders):
            slider.value = val

    for slider in (
        nu_sliders
        + R_sliders
        + w_sliders
        + phi_sliders
        + alpha_sliders
        + theta0_sliders
        + r0_sliders
    ):
        slider.observe(update, "value")

    button = Button(description="Reset")
    button.on_click(reset)
    return VBox([tab, button, output])
