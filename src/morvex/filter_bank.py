"""Complex Morlet-wavelet filter bank."""

from __future__ import annotations

import math

import torch
from pydantic import BaseModel, PositiveFloat, PositiveInt, ValidationError

from ._wavelet_base import _MorletWaveletBase

LN2 = math.log(2.0)
PI = math.pi


class MorletFilterBankConfig(BaseModel):
    n_octaves: PositiveInt
    resolution: PositiveInt
    shape_ratio: PositiveFloat
    time_duration: PositiveFloat
    sampling_freq: PositiveFloat


class MorletFilterBank(_MorletWaveletBase):
    """Complex Morlet-wavelet filter bank with constant-Q properties.

    Parameters
    ----------
    n_octaves : int
        Number of octaves.
    resolution : int
        Number of frequency intervals per octave. The total number of
        wavelets in the filter bank will be `n_octaves * resolution + 1`.
    shape_ratio : float
        Shape ratio of the wavelets.
    time_duration : float
        Time duration of the wavelets, common for all wavelets in the filter
        bank. It should be long enough to capture the oscillations of the
        lowest center frequency, but not too long to avoid unnecessary
        computations.
    sampling_freq : float
        Sampling frequency of the wavelets, common for all wavelets in the
        filter bank. It should be the same as the sampling frequency of the
        signals to be analysed.

    Notes
    -----
    - The unit of the `time_duration` and `sampling_freq` must be compatible
    with each other, since this is not checked internally. For example:

    | `duration`   | `sampling_freq` |
    |--------------|-----------------|
    | seconds      | Hz              |
    | milliseconds | kHz             |
    | microseconds | MHz             |
    """

    def __init__(
        self,
        n_octaves: int,
        resolution: int,
        shape_ratio: float,
        time_duration: float,
        sampling_freq: float,
    ) -> None:
        try:
            cfg = MorletFilterBankConfig(
                n_octaves=n_octaves,
                resolution=resolution,
                shape_ratio=shape_ratio,
                time_duration=time_duration,
                sampling_freq=sampling_freq,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid filter bank configuration: {e}") from e

        center_freqs = _compute_morlet_center_freqs(
            cfg.n_octaves, cfg.resolution, cfg.shape_ratio, cfg.sampling_freq
        )
        super().__init__(
            center_freqs=center_freqs,
            shape_ratios=[shape_ratio],
            time_duration=time_duration,
            sampling_freq=sampling_freq,
        )
        self.n_octaves = n_octaves
        self.resolution = resolution

    @property
    def shape_ratio(self) -> float:
        """Shape ratio of the wavelets."""
        return self._shape_ratios.item()

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
            f"nw={len(self)}, "
            f"tau={self.time_duration:.4f}, "
            f"fs={self.sampling_freq:.4f})"
        )


def _compute_morlet_center_freqs(
    n_octaves: int, resolution: int, shape_ratio: float, sampling_freq: float
) -> torch.Tensor:
    """Compute the center frequencies of a complex Morlet-wavelet filter bank.

    Parameters
    ----------
    n_octaves : int
        Number of octaves.
    resolution : int
        Number of frequency intervals per octave. The total number of wavelets
        in the filter bank will be `n_octaves * resolution + 1`.
    shape_ratio : float
        Shape ratio of the wavelets.
    sampling_freq : float
        Sampling frequency of the wavelet.

    Returns
    -------
    center_freqs : Tensor of shape (n_center_freqs,)
        Center frequencies of the wavelets.
    """
    # No validation is done here since the function is called internally
    # after the configuration has been validated.
    n_cf = n_octaves * resolution + 1
    ratios = torch.linspace(-(n_octaves + 1), -1, n_cf)
    center_freqs = torch.exp2(ratios) * sampling_freq
    freq_widths = (4.0 * LN2 * center_freqs) / (PI * shape_ratio)
    mask = (center_freqs + 0.5 * freq_widths) < (0.5 * sampling_freq)
    return center_freqs[mask]
