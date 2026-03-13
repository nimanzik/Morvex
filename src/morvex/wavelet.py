"""Complex Morlet wavelet with constant-Q properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, PositiveFloat, ValidationError

from ._wavelet_base import _MorletWaveletBase

if TYPE_CHECKING:
    from torch import Tensor


class MorletWaveletConfig(BaseModel):
    """Configuration for a Morlet wavelet."""

    center_freq: PositiveFloat
    shape_ratio: PositiveFloat
    time_duration: PositiveFloat
    sampling_freq: PositiveFloat


class MorletWavelet(_MorletWaveletBase):
    """Complex Morlet wavelet with constant-Q properties.

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
        center_freq: float,
        shape_ratio: float,
        time_duration: float,
        sampling_freq: float,
    ) -> None:
        try:
            cfg = MorletWaveletConfig(
                center_freq=center_freq,
                shape_ratio=shape_ratio,
                time_duration=time_duration,
                sampling_freq=sampling_freq,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid wavelet configuration: {e}") from e

        super().__init__(
            center_freqs=[cfg.center_freq],
            shape_ratios=[cfg.shape_ratio],
            time_duration=cfg.time_duration,
            sampling_freq=cfg.sampling_freq,
        )

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
    def waveform(self) -> Tensor:
        """Waveform of the wavelet.

        Returns
        -------
        out : Tensor of shape (n_samples,)
            Waveform of the wavelet.
        """
        return self.waveforms.squeeze(0)

    def compute_freq_resp(
        self, n_fft: int, normalize: bool = True
    ) -> tuple[Tensor, Tensor]:
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
        out : Tensor of shape (n_fft,)
            Frequency response of the wavelet.
        """
        freqs, resps = super().compute_freq_resps(n_fft=n_fft, scaled=normalize)
        return freqs, resps.squeeze(0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"fc={self.center_freq}, "
            f"kappa={self.shape_ratio}, "
            f"tau={self.time_duration}, "
            f"fs={self.sampling_freq:.6f})"
        )
