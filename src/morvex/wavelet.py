"""Complex Morlet wavelet with constant-Q properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, PositiveFloat, ValidationError
from torch import nn

from ._wavelet_group import CoeffTypeEnum, CoeffTypeLiteral, _MorletWaveletGroup

if TYPE_CHECKING:
    from numpy import floating as np_floating
    from numpy.typing import NDArray
    from torch import Tensor

    from .tapering import Taper


class MorletWaveletConfig(BaseModel):
    """Configuration for a Morlet wavelet."""

    center_freq: PositiveFloat
    shape_ratio: PositiveFloat
    time_duration: PositiveFloat
    sampling_freq: PositiveFloat


class MorletWavelet(nn.Module):
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
        super().__init__()

        try:
            cfg = MorletWaveletConfig(
                center_freq=center_freq,
                shape_ratio=shape_ratio,
                time_duration=time_duration,
                sampling_freq=sampling_freq,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid wavelet configuration: {e}") from e

        self._group = _MorletWaveletGroup(
            center_freqs=[cfg.center_freq],
            shape_ratios=[cfg.shape_ratio],
            time_duration=cfg.time_duration,
            sampling_freq=cfg.sampling_freq,
        )

    @property
    def center_freq(self) -> float:
        """Center frequency of the wavelet.

        Returns
        -------
        out : float
            Center frequency of the wavelet. It is in the same unit as the
            `sampling_freq` parameter.
        """
        return self._group.center_freqs.item()

    @property
    def shape_ratio(self) -> float:
        """Shape ratio of the wavelet.

        Returns
        -------
        out : float
            Shape ratio of the wavelet.
        """
        return self._group.shape_ratios.item()

    @property
    def time_duration(self) -> float:
        """Time duration of the wavelet."""
        return self._group.time_duration

    @property
    def sampling_freq(self) -> float:
        """Sampling frequency of the wavelet."""
        return self._group.sampling_freq

    @property
    def device(self) -> torch.device:
        """Device on which the wavelet parameters are stored."""
        return self._group.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the wavelet parameters."""
        return self._group.dtype

    @property
    def delta_t(self) -> float:
        """Sampling interval of the wavelet."""
        return self._group.delta_t

    @property
    def n_samples(self) -> int:
        """Number of samples in the wavelet."""
        return self._group.n_samples

    @property
    def times(self) -> Tensor:
        """Time points of the wavelet, centered around zero."""
        return self._group.times

    @property
    def time_width(self) -> float:
        """Time width of the wavelet.

        Returns
        -------
        out : float
            Time width of the wavelet. It is in the same unit as the
            `time_duration` parameter.
        """
        return self._group.time_widths.item()

    @property
    def freq_width(self) -> float:
        """Frequency width of the wavelet.

        Returns
        -------
        out : float
            Frequency width of the wavelet. It is in the same unit as the
            `center_freq` parameter.
        """
        return self._group.freq_widths.item()

    @property
    def omega0(self) -> float:
        """Angular frequency of the wavelet."""
        return self._group.omega0s.item()

    @property
    def scale(self) -> float:
        """Scale of the wavelet."""
        return self._group.scales.item()

    @property
    def waveform(self) -> Tensor:
        """Waveform of the wavelet.

        Returns
        -------
        out : Tensor of shape (n_samples,)
            Waveform of the wavelet.
        """
        return self._group.waveforms[0]

    def forward(
        self,
        data: Tensor | NDArray[np_floating],
        taper: Taper | None = None,
        coeff_type: CoeffTypeEnum | CoeffTypeLiteral = CoeffTypeEnum.COMPLEX,
    ) -> Tensor:
        """Compute the wavelet transform of the input signal(s).

        Returns
        -------
        coeffs : Tensor of shape (..., signal_length)
            Wavelet-transform coefficients with no singleton wavelet dimension.
        """
        coeffs = self._group(data, taper=taper, coeff_type=coeff_type)
        return coeffs.squeeze(-2)

    def compute_freq_resp(
        self, n_fft: int | None = None, scaled: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Get the frequency response of the wavelet.

        Parameters
        ----------
        n_fft : int or None, default=None
            Number of FFT points to compute the frequency response. If None,
            the next power of two greater than or equal to the waveform length
            is used.
        scaled : bool, default=False
            If True, the frequency response will be scaled (i.e.,
            non-normalised) by multiplying it with the maximum amplitude of the
            Fourier spectrum of the wavelet. This can be useful for
            visualisation purposes only, but may not be desirable for other
            applications.

        Returns
        -------
        freqs : Tensor of shape (n_freqs,)
            Frequency points.
        resp : Tensor of shape (n_freqs,)
            Frequency response of the wavelet.
        """
        freqs, resps = self._group.compute_freq_resps(n_fft=n_fft, scaled=scaled)
        return freqs, resps[0]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"fc={self.center_freq}, "
            f"kappa={self.shape_ratio}, "
            f"tau={self.time_duration}, "
            f"fs={self.sampling_freq:.6f})"
        )

    def summary(self) -> None:
        """Summarise the wavelet configuration in a rich table."""
        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(box=box.ROUNDED, show_header=True)
        table.title = f"{self.__class__.__name__} Summary"
        table.add_column("Parameter")
        table.add_column("Value")
        table.add_row("Center freq.", f"{self.center_freq}")
        table.add_row("Shape ratio (κ)", f"{self.shape_ratio}")
        table.add_row("Time duration (τ)", f"{self.time_duration}")
        table.add_row("Sampling freq.", f"{self.sampling_freq}")

        console = Console()
        console.print(table)
