"""Base module for Morlet wavelet containers."""

from __future__ import annotations

import math
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Sequence

import torch
import torch.nn as nn
from torch.fft import rfftfreq

from .tapering import Taper
from .xcorr import xcorr_via_fft

if TYPE_CHECKING:
    from numpy import floating as np_floating
    from numpy.typing import NDArray


LN2 = math.log(2.0)
PI = math.pi


class CoeffType(StrEnum):
    POWER = "power"
    MAGNITUDE = "magnitude"
    COMPLEX = "complex"


class _MorletWaveletBase(nn.Module):
    """Internal base class for Morlet wavelet containers.

    Parameters
    ----------
    center_freqs : array-like of float
        Center frequencies of the wavelets.
    shape_ratios : float or array-like of float
        Shape ratios of the wavelets. It should be either a single value
        (common for all wavelets) or an array-like object with the same
        length as `center_freqs`.
    time_duration : float
        Time duration of the wavelets, common for all wavelets in the group
        It should be long enough to capture the oscillations of the lowest
        center frequency, but not too long to avoid unnecessary computations.
    sampling_freq : float
        Sampling frequency of the wavelets, common for all wavelets in the
        group. It should be the same as the sampling frequency of the signals
        to be analysed.
    dtype : torch.dtype or None, optional
        Data type of the wavelet parameters. If `None` (default),
        :func:`torch.get_default_dtype()` is called to determine and use the
        current default floating-point dtype.

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

    _center_freqs: torch.Tensor
    _shape_ratios: torch.Tensor
    _waveforms: torch.Tensor

    def __init__(
        self,
        center_freqs: Sequence[float] | NDArray[np_floating] | torch.Tensor,
        shape_ratios: float | Sequence[float] | NDArray[np_floating] | torch.Tensor,
        time_duration: float,
        sampling_freq: float,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self._time_duration = time_duration
        self._sampling_freq = sampling_freq

        cf_tensor = _coerce_validate_center_freqs(center_freqs, sampling_freq, dtype)
        self.register_buffer("_center_freqs", cf_tensor, persistent=False)

        sr_tensor = _coerce_validate_shape_ratios(shape_ratios, cf_tensor)
        self.register_buffer("_shape_ratios", sr_tensor, persistent=False)

        self.register_buffer("_waveforms", self._compute_waveforms(), persistent=False)

    @property
    def time_duration(self) -> float:
        """Time duration of the wavelets."""
        return self._time_duration

    @property
    def sampling_freq(self) -> float:
        """Sampling frequency of the wavelets."""
        return self._sampling_freq

    @property
    def center_freqs(self) -> torch.Tensor:
        """Center frequencies of the wavelets."""
        return self._center_freqs

    @property
    def shape_ratios(self) -> torch.Tensor:
        """Shape ratios of the wavelets."""
        return self._shape_ratios

    @property
    def waveforms(self) -> torch.Tensor:
        """Return the values of the wavelets in the time domain.

        Waveforms are complex-valued and centered around zero in time.

        Returns
        -------
        waveforms : Tensor of shape (n_wavelets, wavelet_length)
            Wavelets in the time domain.
        """
        return self._waveforms

    @property
    def device(self) -> torch.device:
        """Get the device on which the wavelet parameters are stored."""
        return self._center_freqs.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the data-type of the wavelet parameters."""
        return self._center_freqs.dtype

    @property
    def delta_t(self) -> float:
        """Sampling interval of the wavelets."""
        return 1.0 / self._sampling_freq

    @property
    def n_samples(self) -> int:
        """Number of samples (time points) of the wavelets."""
        return int(round(self._time_duration * self._sampling_freq)) + 1

    @property
    def times(self) -> torch.Tensor:
        """Time points of the wavelets, centered around zero."""
        return (
            torch.arange(self.n_samples, dtype=self.dtype, device=self.device)
            * self.delta_t
            - 0.5 * self._time_duration
        )

    @property
    def time_widths(self) -> torch.Tensor:
        """Time widths of the wavelets.

        The units of the time widths are the same as the `time_duration`
        parameter.

        Returns
        -------
        out : Tensor of shape (n_wavelets,)
            Time widths of the wavelets.
        """
        return self._shape_ratios / self._center_freqs

    @property
    def freq_widths(self) -> torch.Tensor:
        """Frequency widths (bandwidths) of the wavelets.

        The units of the frequency widths are the same as the `sampling_freq`
        parameter.

        Returns
        -------
        out : Tensor of shape (n_wavelets,)
            Frequency widths of the wavelets.
        """
        return (4.0 * LN2) / (PI * self.time_widths)

    @property
    def omega0s(self) -> torch.Tensor:
        """Angular frequencies of the wavelets (a.k.a. `omega0`s).

        Returns
        -------
        out : Tensor of shape (n_wavelets,) or (1,)
            Angular frequencies of the wavelets.
        """
        return (self._shape_ratios * PI) / math.sqrt(2.0 * LN2)

    @property
    def scales(self) -> torch.Tensor:
        """Scales of the wavelets."""
        return (self.omega0s * self._sampling_freq) / (2.0 * PI * self._center_freqs)

    def _compute_waveforms(self) -> torch.Tensor:
        """Compute the values of the wavelets (waveforms) in the time domain.

        Waveforms are complex-valued and centered around zero time.

        Returns
        -------
        waveforms : Tensor of shape (n_wavelets, wavelet_length)
            Wavelets in the time domain.
        """
        t = self.times  # (wavelet_length,)
        tw = self.time_widths[:, None]  # (n_wavelets, 1)
        cf = self._center_freqs[:, None]  # (n_wavelets, 1)

        gaussian = torch.exp(-4.0 * LN2 * torch.square(t / tw))
        oscillation = torch.exp(1j * 2.0 * PI * cf * t)

        return gaussian * oscillation

    def forward(
        self,
        data: torch.Tensor | NDArray[np_floating],
        taper: Taper | None = None,
        coeff_type: CoeffType
        | Literal["power", "magnitude", "complex"] = CoeffType.POWER,
    ) -> torch.Tensor:
        """Compute the wavelet transform of the input signal(s).

        Parameters
        ----------
        data : Tensor or ndarray of shape (..., signal_length)
            Input signal(s) to be analysed.
        taper : Taper or None, default=None
            Tapering module to apply to the input signal(s) before computing
            the wavelet transform. If None, a default Hann taper with a
            maximum fade length of 5% of the signal length will be applied.
        coeff_type : {'power', 'magnitude', 'complex'}, default='power'
            The type of the wavelet-transform coefficients to return:
                - `'power'`: squared magnitude of the coefficients.
                - `'magnitude'`: absolute magnitude of the coefficients.
                - `'complex'`: complex-valued coefficients.

        Returns
        -------
        coeffs : Tensor of shape (..., n_wavelets, signal_length)
            Wavelet-transformed coefficients of the input signal(s). The shape
            of the output tensor depends on the shape of the input signal(s):

            | Input shape | Output shape   |
            |-------------|----------------|
            | `(L,)`      | `(M, L)`       |
            | `(C, L)`    | `(C, M, L)`    |
            | `(B, C, L)` | `(B, C, M, L)` |

            where `L` is the length of the signal(s) in samples, `M` is the
            number of wavelets, `C` is the number of channels, and `B` is the
            batch size.
        """
        coeff_type = CoeffType(coeff_type)

        if taper is None:
            taper = _get_default_taper(data.shape[-1], self.dtype, self.device)

        x_in = _preprocess_input(data, taper, self.dtype, self.device)

        # Cross-correlate + normalise by the scales
        # Detach fixed buffers so that autograd does not track them
        waveforms = self.waveforms.detach()
        scales = self.scales.detach()
        coeffs = xcorr_via_fft(x_in, waveforms) / torch.sqrt(scales[:, None])

        if coeff_type == CoeffType.POWER:
            return (coeffs * coeffs.conj()).real
        elif coeff_type == CoeffType.MAGNITUDE:
            return coeffs.abs()
        else:  # "complex"
            return coeffs

    @torch.no_grad()
    def compute_freq_resps(
        self, n_fft: int | None = None, scaled: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the frequency responses of the wavelets.

        Parameters
        ----------
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_samples`
            will be used.
        scaled : bool, default=True
            Whether to return the scaled responses. If False, the responses
            are normalised to one.

        Returns
        -------
        freqs : Tensor of shape (n_freqs,)
            Frequency points.
        resps : Tensor of shape (n_wavelets, n_freqs)
            Frequency responses of the wavelets.
        """
        if n_fft is None:
            n_fft = int(2 ** math.ceil(math.log2(self.n_samples)))

        n_fft = max(n_fft, self.n_samples)
        rfreqs = rfftfreq(n=n_fft, d=self.delta_t, device=self.device, dtype=self.dtype)

        phase_diffs = 2.0 * PI * (rfreqs - self._center_freqs[:, None])
        resps = torch.exp(
            -1.0 * torch.square(self.time_widths[:, None] * phase_diffs) / (16.0 * LN2)
        )

        if scaled:
            # Maximum amplitudes of the Fourier spectra of the wavelets
            max_spec_amps = 0.5 * math.sqrt(PI / LN2) * self.time_widths
            resps = resps * max_spec_amps[:, None]

        return rfreqs, resps

    def __len__(self) -> int:
        """Return the number of wavelets in the group."""
        return self._center_freqs.numel()

    def __repr__(self) -> str:
        """Return a string representation of the wavelet group."""
        return (
            f"{self.__class__.__name__}("
            f"nw={len(self)}, "
            f"tau={self.time_duration:.4f}, "
            f"fs={self.sampling_freq:.4f})"
        )


def _coerce_validate_center_freqs(
    center_freqs: Sequence[float] | NDArray[np_floating] | torch.Tensor,
    sampling_freq: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Coerce and validate a sequence of center frequencies and return a tensor."""  # noqa: W505
    dtype = dtype or torch.get_default_dtype()
    center_freqs = torch.as_tensor(center_freqs, dtype=dtype)
    nyquist = sampling_freq * 0.5
    if center_freqs.numel() == 0:
        raise ValueError("Center frequencies must be non-empty.")
    if center_freqs.ndim != 1:
        raise ValueError("Center frequencies must be 1-dimensional.")
    if not torch.all(torch.isfinite(center_freqs)):
        raise ValueError("Center frequencies must be finite values.")
    if torch.any(center_freqs <= 0.0) or torch.any(center_freqs >= nyquist):
        raise ValueError(
            f"Center frequencies must be in the range (0, {nyquist}) (exclusive)."
        )
    return center_freqs


def _coerce_validate_shape_ratios(
    shape_ratios: float | Sequence[float] | NDArray[np_floating] | torch.Tensor,
    center_freqs: torch.Tensor,
) -> torch.Tensor:
    """Coerce and validate a sequence of shape ratios and return a tensor."""
    shape_ratios = torch.as_tensor(shape_ratios, dtype=center_freqs.dtype)
    if not torch.all(torch.isfinite(shape_ratios)):
        raise ValueError("Shape ratios must be finite values.")
    if torch.any(shape_ratios <= 0.0):
        raise ValueError("Shape ratios must be positive values.")
    if shape_ratios.numel() != 1 and shape_ratios.shape != center_freqs.shape:
        raise ValueError(
            "Shape ratios must be either a scalar, or an array-like with the "
            "same length as center frequencies."
        )
    return shape_ratios


@lru_cache(maxsize=32)
def _get_default_taper(
    n_samples: int, dtype: torch.dtype, device: torch.device
) -> Taper:
    """Return a cached default Hann taper for the given signal length."""
    return Taper(
        window_type="hann",
        n_samples=n_samples,
        max_percentage=0.05,
        side="both",
    ).to(dtype=dtype, device=device)


def _preprocess_input(
    data: torch.Tensor | NDArray[np_floating],
    taper: Taper,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Preprocess input data by coercing to tensor, demeaning, and tapering."""
    x_in = torch.as_tensor(data, dtype=dtype, device=device)

    # Demean + taper
    # Note that 'as_tensor' returns a view if dtype/device match. DO NOT
    # demean x_in in-place unless a clone has been made first.
    x_in = x_in - x_in.mean(dim=-1, keepdim=True)
    x_in = taper(x_in)

    return x_in
