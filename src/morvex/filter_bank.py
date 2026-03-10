"""Complex Morlet-wavelet filter bank."""

from __future__ import annotations

import math

import torch

from ._wavelet_base import _MorletWaveletBase

LN2 = math.log(2.0)
PI = math.pi


def _compute_morlet_center_freqs(
    n_octaves: int, resolutions: int, shape_ratio: float, sampling_freq: float
) -> torch.Tensor:
    """Compute the center frequencies of a complex Morlet-wavelet filter bank.

    Parameters
    ----------
    n_octaves : int
        Number of octaves.
    resolutions : int
        Number of frequency intervals per octave. The total number of wavelets
        in the filter bank will be `n_octaves * resolutions + 1`.
    shape_ratio : float
        Shape ratio of the wavelet.
    sampling_freq : float
        Sampling frequency of the wavelet.

    Returns
    -------
    center_freqs : ndarray of shape (n_center_freqs,)
        Center frequencies of the wavelets.
    """
    if n_octaves <= 0 or resolutions <= 0:
        raise ValueError("Number of octaves and intervals must be positive.")

    if shape_ratio <= 0:
        raise ValueError(
            f"Shape ratio must be a positive value, but got {shape_ratio}."
        )

    if sampling_freq <= 0:
        raise ValueError("Sampling frequency must be positive.")

    n_cf = n_octaves * resolutions + 1
    ratios = torch.linspace(-(n_octaves + 1), -1, n_cf)
    center_freqs = torch.exp2(ratios) * sampling_freq
    freq_widths = (4.0 * LN2 * center_freqs) / (PI * shape_ratio)
    mask = (center_freqs + 0.5 * freq_widths) < (0.5 * sampling_freq)
    return center_freqs[mask]


class MorletFilterBank(_MorletWaveletBase):
    """Complex Morlet-wavelet filter bank with constant-Q properties."""

    def __init__(
        self,
        n_octaves: int,
        resolution: int,
        shape_ratio: float,
        time_duration: float,
        sampling_freq: float,
    ) -> None:
        """Initialise the complex Morlet-wavelet filter bank.

        Parameters
        ----------
        n_octaves : int
            Number of octaves.
        resolution : int
            Number of frequency intervals per octave. The total number of
            wavelets in the filter bank will be `n_octaves * resolutions + 1`.
        shape_ratio : float
            Shape ratio of the wavelets.
        time_duration : float
            Time duration of the wavelets, common for all wavelets in the
            filter bank. It should be long enough to capture the oscillations
            of the lowest center frequency, but not too long to avoid
            unnecessary computations.
        sampling_freq : float
            Sampling frequency of the wavelets, common for all wavelets in the
            filter bank. It should be the same as the sampling frequency of the
            signals to be analysed.

        Notes
        -----
        - The unit of the `time_duration` and `sampling_freq` must be
          compatible with each other, since this is not checked internally.
          For example:

          | `duration`   | `sampling_freq` |
          |--------------|-----------------|
          | seconds      | Hz              |
          | milliseconds | kHz             |
          | microseconds | MHz             |
        """
        center_freqs = _compute_morlet_center_freqs(
            n_octaves, resolution, shape_ratio, sampling_freq
        )
        super().__init__(
            center_freqs=center_freqs,
            shape_ratios=[shape_ratio],
            time_duration=time_duration,
            sampling_freq=sampling_freq,
        )
        self.n_octaves = n_octaves
        self.resolution = resolution
        self.shape_ratio = shape_ratio

    @property
    def omega0(self) -> float:
        """Angular frequency of the mother wavelet (a.k.a. `omega0`)."""
        return self.omega0s.item()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"J={self.n_octaves}, "
            f"Q={self.resolution}, "
            f"kappa={self.shape_ratio}, "
            f"tau={self.time_duration}, "
            f"fs={self.sampling_freq})"
        )
