from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Sequence

import torch
import torch.nn as nn
from torch.fft import rfftfreq

from .tapering import Taper
from .xcorr import xcorr_via_fft

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from numpy import floating as np_floating
    from numpy.typing import NDArray
    from plotly.graph_objects import Figure as PlotlyFigure


LN2 = math.log(2.0)
PI = math.pi


class MorletWaveletGroup(nn.Module):
    """Base class for a group of Morlet wavelets."""

    _center_freqs: torch.Tensor
    _shape_ratios: torch.Tensor

    def __init__(
        self,
        center_freqs: Sequence[float] | NDArray[np_floating] | torch.Tensor,
        shape_ratios: Sequence[float] | NDArray[np_floating] | torch.Tensor,
        time_duration: float,
        sampling_freq: float,
    ) -> None:
        """Initialise the Morlet wavelet group.

        Parameters
        ----------
        center_freqs : array-like of float
            Center frequencies of the wavelets.
        shape_ratios : array-like of float
            Shape ratios of the wavelets. It should be an array-like object
            object with either a single value (common for all wavelets) or the
            same length as the `center_freqs`.
        time_duration : float
            Time duration of the wavelets, common for all wavelets in the
            group. It should be long enough to capture the oscillations of
            the lowest center frequency, but not too long to avoid unnecessary
            computations.
        sampling_freq : float
            Sampling frequency of the wavelets, common for all wavelets in the
            group. It should be the same as the sampling frequency of the
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
        super().__init__()

        self.time_duration = time_duration
        self.sampling_freq = sampling_freq
        self.register_buffer(
            name="_center_freqs", tensor=torch.atleast_1d(torch.as_tensor(center_freqs))
        )
        self.register_buffer(
            name="_shape_ratios", tensor=torch.atleast_1d(torch.as_tensor(shape_ratios))
        )

        self._validate_center_freqs()
        self._validate_shape_ratios()

    @property
    def nyquist_freq(self) -> float:
        """Nyquist frequency of the wavelets."""
        return 0.5 * self.sampling_freq

    def _validate_center_freqs(self) -> None:
        """Ensure the center frequencies are valid."""
        desc = "Center frequencies"
        if self._center_freqs.numel() == 0:
            raise ValueError(f"{desc} must contain at least one value.")

        if self._center_freqs.ndim != 1:
            raise ValueError(f"{desc} must be a 1D array-like object.")

        if not torch.all(self._center_freqs > 0.0):
            raise ValueError(f"{desc} must be positive values.")

        if not torch.all(self._center_freqs < self.nyquist_freq):
            raise ValueError(
                f"{desc} must be less than the Nyquist "
                f"frequency of {self.nyquist_freq}."
            )

    def _validate_shape_ratios(self) -> None:
        """Ensure the shape ratios are valid."""
        desc = "Shape ratios"
        if not torch.all(self._shape_ratios > 0.0):
            raise ValueError(f"{desc} must be positive values.")

        if (
            self._shape_ratios.numel() != 1
            and self._shape_ratios.shape != self._center_freqs.shape
        ):
            raise ValueError(
                f"{desc} must be either a length-1 array-like object or "
                f"have the same length as the center frequencies."
            )

    def __len__(self) -> int:
        """Return the number of wavelets in the group."""
        return self._center_freqs.numel()

    @property
    def center_freqs(self) -> torch.Tensor:
        """Center frequencies of the wavelets."""
        return self._center_freqs

    @property
    def shape_ratios(self) -> torch.Tensor:
        """Shape ratios of the wavelets."""
        return self._shape_ratios

    @property
    def device(self) -> torch.device:
        """Get the device on which the wavelet parameters are stored."""
        return self._center_freqs.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the wavelet parameters."""
        return self._center_freqs.dtype

    @property
    def delta_t(self) -> float:
        """Sampling interval of the wavelets."""
        return 1.0 / self.sampling_freq

    @property
    def n_t(self) -> int:
        """Number of time samples of the wavelets."""
        return int(round(self.time_duration * self.sampling_freq)) + 1

    @property
    def times(self) -> torch.Tensor:
        """Time points of the wavelets, centered around zero."""
        return (
            torch.arange(self.n_t, dtype=self.dtype, device=self.device) * self.delta_t
            - 0.5 * self.time_duration
        )

    @property
    def time_widths(self) -> torch.Tensor:
        """Time widths of the wavelets.

        Returns
        -------
        out : torch.Tensor of shape (n_center_freqs,)
            Time widths of the wavelets. They are in the same units as the
            `time_duration` parameter.
        """
        return self.shape_ratios / self.center_freqs

    @property
    def freq_widths(self) -> torch.Tensor:
        """Frequency widths (bandwidths) of the wavelets.

        Returns
        -------
        out : torch.Tensor of shape (n_center_freqs,)
            Frequency widths of the wavelets. They are in the same units as
            the `sampling_freq`.
        """
        return (4.0 * LN2) / (PI * self.time_widths)

    @property
    def omega0s(self) -> torch.Tensor:
        """Angular frequencies of the wavelets (a.k.a. `omega0`s)."""
        return (self.shape_ratios * PI) / math.sqrt(2.0 * LN2)

    @property
    def scales(self) -> torch.Tensor:
        """Scales of the wavelets."""
        return (self.omega0s * self.sampling_freq) / (2.0 * PI * self.center_freqs)

    @property
    def waveforms(self) -> torch.Tensor:
        """Return the values of the wavelets in the time domain.

        Returns
        -------
        waveforms : torch.Tensor of shape (n_center_freqs, n_times)
            Wavelets in the time domain. They are complex-valued and centered
            around zero in time.
        """
        # Gaussian envelope
        t_over_dt = self.times / self.time_widths[:, None]
        gaussian = torch.exp(-4.0 * LN2 * t_over_dt**2)

        # Oscillatory component
        oscillation = torch.exp(1j * 2.0 * PI * self.center_freqs[:, None] * self.times)

        return gaussian * oscillation

    @property
    def max_spec_amps(self) -> torch.Tensor:
        """Maximum amplitudes of the Fourier spectra of the wavelets."""
        return 0.5 * math.sqrt(PI / LN2) * self.time_widths

    def forward(
        self,
        data: torch.Tensor | NDArray[np_floating],
        taper: Taper | None = None,
        coeff_type: Literal["power", "magnitude", "complex"] = "power",
    ) -> torch.Tensor:
        """Compute the wavelet transform of the input signal(s).

        Parameters
        ----------
        data : torch.Tensor or ndarray of shape (..., n_times)
            Input signal(s) to be analyzed.
        taper : Taper or None, default=None
            Tapering module to apply to the input signal(s) before computing
            the wavelet transform. If None, a default Hann taper with a
            maximum fade length of 5% of the signal length will be applied.
        coeff_type : {'power', 'magnitude', 'complex'}, default='power'
            Specifies the type of the wavelet coefficients to return:
                - `'power'`: squared magnitude of the coefficients.
                - `'magnitude'`: absolute magnitude of the coefficients.
                - `'complex'`: complex-valued coefficients.

        Returns
        -------
        coeffs : torch.Tensor of shape (..., n_center_freqs, n_times)
            Wavelet-transformed coefficients of the input signal(s). The shape
            of the output tensor depends on the shape of the input signal(s):

            | Input shape | Output shape   |
            |-------------|----------------|
            | `(L,)`      | `(F, L)`       |
            | `(B, L)`    | `(B, F, L)`    |
            | `(C, L)`    | `(C, F, L)`    |
            | `(B, C, L)` | `(B, C, F, L)` |

            where `L` is the number of time samples in the input signal(s), `F`
            is the number of wavelets (i.e., center frequencies) in the group,
            `B` is the batch size, and `C` is the number of channels.
        """
        if coeff_type not in {"power", "magnitude", "complex"}:
            raise ValueError(
                f"Coefficient type must be one of ('power', 'magnitude', 'complex'), "
                f"but got '{coeff_type}'."
            )

        x_in = torch.as_tensor(data, dtype=self.dtype, device=self.device)

        if taper is None:
            taper = Taper(
                window_type="hann",
                n_samples=x_in.shape[-1],
                max_percentage=0.05,
                side="both",
            ).to(device=self.device, dtype=self.dtype)

        # Deman + taper
        x_in = x_in - x_in.mean(dim=-1, keepdim=True)
        x_in = taper(x_in)

        # Cross-correlate + normalise by the scales
        coeffs = xcorr_via_fft(x_in, self.waveforms) / torch.sqrt(self.scales[:, None])

        if coeff_type == "power":
            return torch.square(torch.abs(coeffs))
        elif coeff_type == "magnitude":
            return torch.abs(coeffs)
        else:  # coeff_type == "complex":
            return coeffs

    def transform(self, *args, **kwargs) -> torch.Tensor:
        """Alias for the forward method."""
        return self.forward(*args, **kwargs)

    def get_freq_resps(
        self, n_fft: int | None = None, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the frequency responses of the wavelets.

        Parameters
        ----------
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.
        normalize : bool, default=True
            Whether to return the normalised responses.

        Returns
        -------
        freqs : torch.Tensor of shape (n_freqs,)
            Frequency points.
        resps : torch.Tensor of shape (n_center_freqs, n_freqs)
            Frequency responses of the wavelets.
        """
        if n_fft is None:
            n_fft = int(2 ** math.ceil(math.log2(self.n_t)))

        n_fft = max(n_fft, self.n_t)
        rfreqs = rfftfreq(n=n_fft, d=self.delta_t, device=self.device, dtype=self.dtype)

        phase_diffs = 2.0 * PI * (rfreqs - self.center_freqs[:, None])
        resps = torch.exp(
            -1.0 * torch.square(self.time_widths[:, None] * phase_diffs) / (16.0 * LN2)
        )

        if not normalize:
            resps = resps * self.max_spec_amps[:, None]

        return rfreqs, resps

    def plot_freq_resps_mpl(
        self,
        ax: MplAxes,
        n_fft: int | None = None,
        normalize: bool = True,
        color: str | None = None,
        auto_xlabel: bool = True,
        auto_ylabel: bool = True,
        auto_title: bool = True,
    ) -> MplAxes:
        """Plot the frequency responses of the wavelets using Matplotlib.

        Parameters
        ----------
        ax : Axes
            The Matplotlib axes to plot the frequency responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.
        normalize : bool, default=True
            Whether to plot the normalized responses.
        color : str or None, default=None
            Color to use for plotting the frequency responses. If None, the
            default color cycle of Matplotlib will be used.
        auto_xlabel : bool, default=True
            Whether to automatically set the x-axis label.
        auto_ylabel : bool, default=True
            Whether to automatically set the y-axis label.
        auto_title : bool, default=True
            Whether to automatically set the title.

        Returns
        -------
        ax : Axes
            Matplotlib axes displaying the frequency responses.
        """
        freqs_, resps_ = self.get_freq_resps(n_fft, normalize=normalize)

        freqs = freqs_.detach().cpu().numpy()
        resps = resps_.detach().cpu().numpy()

        for resp in resps:
            ax.plot(freqs, resp, color=color)

        if auto_xlabel:
            ax.set_xlabel("Frequency")
        if auto_ylabel:
            ax.set_ylabel("Amplitude, normalised" if normalize else "Amplitude")
        if auto_title:
            ax.set_title("Wavelets Frequency Responses")
        return ax

    def plot_freq_resps_plotly(
        self,
        fig: PlotlyFigure,
        n_fft: int | None = None,
        normalize: bool = True,
        color: str | None = None,
        auto_xlabel: bool = True,
        auto_ylabel: bool = True,
        auto_title: bool = True,
    ) -> PlotlyFigure:
        """Plot the frequency responses of the wavelets using Plotly.

        Parameters
        ----------
        fig : PlotlyFigure
            The Plotly figure to plot the frequency responses.
        normalize : bool, default=True
            Whether to plot the normalized responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.
        color : str or None, default=None
            Color to use for plotting the frequency responses. If None, the
            default color cycle of Plotly will be used.
        auto_xlabel : bool, default=True
            Whether to automatically set the x-axis label.
        auto_ylabel : bool, default=True
            Whether to automatically set the y-axis label.
        auto_title : bool, default=True
            Whether to automatically set the title.

        Returns
        -------
        fig: PlotlyFigure
            Plotly figure displaying the frequency responses.
        """
        from plotly import graph_objects as go

        freqs_, resps_ = self.get_freq_resps(n_fft, normalize=normalize)

        # Convert to numpy for plotting
        freqs = freqs_.detach().cpu().numpy()
        resps = resps_.detach().cpu().numpy()

        for resp in resps:
            fig.add_trace(go.Scatter(x=freqs, y=resp, showlegend=False))

        if auto_xlabel:
            fig.update_xaxes(title_text="Frequency")
        if auto_ylabel:
            fig.update_yaxes(
                title_text="Amplitude, normalised" if normalize else "Amplitude"
            )
        if auto_title:
            fig.update_layout(title="Wavelets Frequency Responses")

        return fig
