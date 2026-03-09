from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .wavelet_group import MorletWaveletGroup

if TYPE_CHECKING:
    import torch


class MorletWavelet(MorletWaveletGroup):
    """Complex Morlet wavelet with constant-Q properties."""

    def __init__(
        self,
        center_freq: float,
        shape_ratio: float,
        time_duration: float,
        sampling_freq: float,
    ) -> None:
        """Initialise a Morlet wavelet.

        Parameters
        ----------
        center_freq : float
            Center frequency of the wavelet.
        shape_ratio : float
            Shape ratio of the wavelet.
        time_duration : float
            Time duration of the wavelet.
        sampling_freq : float
            Sampling frequency of the wavelet.

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
        super().__init__(
            center_freqs=[center_freq],
            shape_ratios=[shape_ratio],
            time_duration=time_duration,
            sampling_freq=sampling_freq,
        )
        self.center_freq = center_freq
        self.shape_ratio = shape_ratio

    @property
    def time_width(self) -> float:
        """Time width of the wavelet.

        Returns
        -------
        out : float
            Time width of the wavelet. It is in the same unit as the
            `time_duration` parameter.
        """
        return self.time_widths.item()

    @property
    def freq_width(self) -> float:
        """Frequency width of the wavelet.

        Returns
        -------
        out : float
            Frequency width of the wavelet. It is in the same unit as the
            `center_freq` parameter.
        """
        return self.freq_widths.item()

    @property
    def waveform(self) -> torch.Tensor:
        """Waveform of the wavelet.

        Returns
        -------
        out : torch.Tensor of shape (n_samples,)
            Waveform of the wavelet.
        """
        return self.waveforms.squeeze(0)

    @property
    def max_apec_amp(self) -> float:
        """Maximum amplitude of the wavelet spectrum.

        Returns
        -------
        out : float
            Maximum amplitude of the wavelet spectrum.
        """
        return self.max_spec_amps.item()

    def get_freq_resp(
        self, n_fft: int, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the frequency response of the wavelet.

        Parameters
        ----------
        n_fft : int
            Number of FFT points to compute the frequency response. It should
            be at least as large as the length of the wavelet waveform, but
            can be larger to get a smoother frequency response.
        normalize : bool, optional
            Whether to normalize the frequency response by the maximum
            amplitude of the wavelet spectrum. If `True`, the maximum
            amplitude will be 1. Default is `True`.

        Returns
        -------
        out : torch.Tensor of shape (n_fft,)
            Frequency response of the wavelet.
        """
        freqs, resps = super().get_freq_resps(n_fft=n_fft, normalize=normalize)
        return freqs, resps.squeeze(0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"fc={self.center_freq}, "
            f"kappa={self.shape_ratio}, "
            f"tau={self.time_duration}, "
            f"fs={self.sampling_freq:.6f})"
        )
