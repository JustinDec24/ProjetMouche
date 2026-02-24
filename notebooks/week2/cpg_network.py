"""CPG (Central Pattern Generator) network implementation.

This module provides a simple oscillator-based CPG network suitable for
controlling hexapod locomotion.  The formulation follows Ijspeert et al.
(2007), *Science*.
"""

from __future__ import annotations

import numpy as np


def calculate_ddt(
    theta: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    phi: np.ndarray,
    nu: np.ndarray,
    R: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the time derivatives of phase (θ) and magnitude (r).

    Parameters
    ----------
    theta : (N,) array – current phases.
    r : (N,) array – current magnitudes.
    w : (N, N) array – coupling weights.
    phi : (N, N) array – phase biases.
    nu : (N,) array – intrinsic frequencies (Hz).
    R : (N,) array – intrinsic amplitudes.
    alpha : (N,) array – convergence coefficients.

    Returns
    -------
    dtheta_dt, dr_dt : tuple of (N,) arrays.
    """
    intrinsic_term = 2 * np.pi * nu
    phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_term = (r * w * np.sin(phase_diff - phi)).sum(axis=1)
    dtheta_dt = intrinsic_term + coupling_term
    dr_dt = alpha * (R - r)
    return dtheta_dt, dr_dt


class CPGNetwork:
    """Network of coupled oscillators integrated with Euler's method.

    Parameters
    ----------
    timestep : float
        Integration timestep (seconds).
    intrinsic_freqs : (N,) array
        Intrinsic frequencies of the oscillators (Hz).
    intrinsic_amps : (N,) array
        Intrinsic amplitudes of the oscillators.
    coupling_weights : (N, N) array
        Coupling weights between oscillators.
    phase_biases : (N, N) array
        Phase biases between oscillators (radians).
    convergence_coefs : (N,) array
        Rate-of-convergence coefficients for the amplitudes.
    init_phases : (N,) array, optional
        Initial phases.  Randomly sampled if not provided.
    init_magnitudes : (N,) array, optional
        Initial magnitudes.  Zeros if not provided.
    seed : int
        Random seed for reproducible initialisation.
    """

    def __init__(
        self,
        timestep: float,
        intrinsic_freqs: np.ndarray,
        intrinsic_amps: np.ndarray,
        coupling_weights: np.ndarray,
        phase_biases: np.ndarray,
        convergence_coefs: np.ndarray,
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.timestep = timestep
        self.num_cpgs = intrinsic_freqs.size
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.coupling_weights = coupling_weights
        self.phase_biases = phase_biases
        self.convergence_coefs = convergence_coefs
        self.seed = seed

        self.reset(init_phases, init_magnitudes)

        # Validate shapes
        assert intrinsic_freqs.shape == (self.num_cpgs,)
        assert coupling_weights.shape == (self.num_cpgs, self.num_cpgs)
        assert phase_biases.shape == (self.num_cpgs, self.num_cpgs)
        assert convergence_coefs.shape == (self.num_cpgs,)
        assert self.curr_phases.shape == (self.num_cpgs,)
        assert self.curr_magnitudes.shape == (self.num_cpgs,)

    def step(self) -> None:
        """Advance the network by one timestep (Euler integration)."""
        dtheta_dt, dr_dt = calculate_ddt(
            theta=self.curr_phases,
            r=self.curr_magnitudes,
            w=self.coupling_weights,
            phi=self.phase_biases,
            nu=self.intrinsic_freqs,
            R=self.intrinsic_amps,
            alpha=self.convergence_coefs,
        )
        self.curr_phases += dtheta_dt * self.timestep
        self.curr_magnitudes += dr_dt * self.timestep

    def reset(
        self,
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
    ) -> None:
        """Reset oscillator states.

        High magnitudes combined with unfortunate initial phases can cause
        physics instabilities, so starting with zero magnitudes is safest.
        """
        if init_phases is None:
            rng = np.random.default_rng(seed=self.seed)
            self.curr_phases = rng.random(self.num_cpgs) * 2 * np.pi
        else:
            self.curr_phases = init_phases

        if init_magnitudes is None:
            self.curr_magnitudes = np.zeros(self.num_cpgs)
        else:
            self.curr_magnitudes = init_magnitudes
