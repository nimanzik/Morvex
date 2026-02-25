from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

Ln2 = math.log(2.0)
PI = math.pi


class _MorletWaveletGroup(nn.Module):
    """Base class for single and multi-scale complex Morlet wavelets.

    Parameters
    ----------
    center_freqs : float or array-like of float
        Center frequencies of the wavelets.
    shape_ratios : float or array-like of float
        Shape ratios of the wavelets (a.k.a. number of cycles).
    duration : float
        Time duration of the wavelets.
    sampling_freq : float
        Sampling frequency of the wavelets (should be the same as the signals
        to be analyzed).
    learnable : bool, default=False
        If True, shape ratios become learnable parameters that can be optimised
        via backpropagation.

    Raises
    ------
    ValueError
        - If the center frequencies are not positive or exceed the Nyquist.
        - If the shape ratios are not positive or have an incompatible shape
          with the center frequencies.

    Notes
    -----
    - The unit of the `duration` and `sampling_freq` must be compatible with
      each other, since this is not checked internally:

      | `duration`   | `sampling_freq` |
      |--------------|-----------------|
      | seconds      | Hz              |
      | milliseconds | kHz             |
      | microseconds | MHz             |
    """

    _center_freqs: torch.Tensor
    _shape_ratios: torch.Tensor

    def __init__(
        self,
        center_freqs: float | Sequence[float] | NDArray[np.float] | torch.Tensor,
        shape_ratios: float | Sequence[float] | NDArray[np.float] | torch.Tensor,
        duration: float,
        sampling_freq: float,
        learnable: bool = False,
    ) -> None:
        """Initialize the complex Morlet-wavelet group."""
        super().__init__()

        self.duration = duration
        self.sampling_freq = sampling_freq

        if isinstance(center_freqs, int | float):
            center_freqs = [center_freqs]

        if isinstance(shape_ratios, int | float):
            shape_ratios = [shape_ratios]

        center_freqs_tensor = torch.atleast_1d(torch.as_tensor(center_freqs))
        shape_ratios_tensor = torch.atleast_1d(torch.as_tensor(shape_ratios))

        # Center freqs: buffer (not learnable)
        self.register_buffer("_center_freqs", center_freqs_tensor)

        # Shape rations: parameter (learnable) or buffer (not learnable)
        if learnable:
            self._shape_ratios = nn.Parameter(shape_ratios_tensor)
        else:
            self.register_buffer("_shape_ratios", shape_ratios_tensor)

        self._validate_center_freqs()
        self._validate_shape_ratios()

    @property
    def nyquist_freq(self) -> float:
        """Nyquist frequency of the wavelets."""
        return 0.5 * self.sampling_freq

    def _validate_center_freqs(self) -> None:
        """Check the center frequencies of the wavelets."""
        title = "Center frequencies"
        if self._center_freqs.numel() == 0:
            raise ValueError(f"{title} cannot be empty or zero-length array.")

        if self._center_freqs.ndim != 1:
            raise ValueError(f"{title} must be a 1D array-like object.")

        if not torch.all(self._center_freqs > 0.0):
            raise ValueError(f"{title} must be positive values.")

        if not torch.all(self._center_freqs < self.nyquist_freq):
            raise ValueError(
                f"{title} must be less than the "
                f"Nyquist frequency of {self.nyquist_freq}."
            )

    def _validate_shape_ratios(self) -> None:
        """Check the shape ratios of the wavelets."""
        title = "Shape ratios"
        if not torch.all(self._shape_ratios > 0.0):
            raise ValueError(f"{title} must be positive values.")

        if (
            self._shape_ratios.numel() != 1
            and self._shape_ratios.shape != self._center_freqs.shape
        ):
            raise ValueError(
                f"{title} must be either a scalar or a 1D array-like "
                "object with the same length as the center frequencies."
            )

    def __len__(self) -> int:
        """Return the number of wavelets in the group."""
        return self._center_freqs.numel()

    @property
    def center_freqs(self) -> torch.Tensor:
        """Center frequencies of the wavelets.

        The unit of the center frequencies is the same as the `sampling_freq`.
        """
        return self._center_freqs

    @property
    def shape_ratios(self) -> torch.Tensor:
        """Shape ratios of the wavelets."""
        if self._shape_ratios.numel() == 1 and len(self) > 1:
            return self._shape_ratios.expand(len(self))
        return self._shape_ratios

    def _get_device(self) -> torch.device:
        """Get the device on which the wavelet parameters are allocated."""
        return self._center_freqs.device

    def _get_dtype(self) -> torch.dtype:
        """Get the data type of the wavelet parameters."""
        return self._center_freqs.dtype

    @property
    def delta_t(self) -> float:
        """Sampling interval of the wavelets.

        Unit is the same as the `duration`.
        """
        return 1.0 / self.sampling_freq

    @property
    def n_t(self) -> int:
        """Number of time samples of the wavelets."""
        return int(round(self.duration * self.sampling_freq)) + 1

    @property
    def times(self) -> torch.Tensor:
        """Time samples of the wavelets.

        The time samples are centered around zero, so they range from
        `-duration/2` to `duration/2` (inclusive). Unit is the same as the
        `duration` and `delta_t`.

        Returns
        -------
        times : torch.Tensor of shape (n_times,)
            Time samples of the wavelets.
        """
        device = self._get_device()
        dtype = self._get_dtype()
        return (
            torch.arange(self.n_t, dtype=dtype, device=device) * self.delta_t
            - 0.5 * self.duration
        )

    @property
    def time_widths(self) -> torch.Tensor:
        """Time widths of the wavelets.

        Unit is the same as the `duration` and `delta_t`.

        Returns
        -------
        time_widths : torch.Tensor of shape (n_center_freqs,)
            Time widths of the wavelets.
        """
        return self.shape_ratios / self.center_freqs

    @property
    def freq_widths(self) -> torch.Tensor:
        """Frequency widths (bandwidths) of the wavelets.

        Unit is the same as the `sampling_freq`.

        Returns
        -------
        freq_widths : torch.Tensor of shape (n_center_freqs,)
            Frequency widths of the wavelets.
        """
        return (4.0 * Ln2) / (PI * self.time_widths)

    @property
    def omega0s(self) -> torch.Tensor:
        """Angular frequencies of the wavelets (Scipy's `omega0`)."""
        return (self.shape_ratios * PI) / math.sqrt(2.0 * Ln2)

    @property
    def scales(self) -> torch.Tensor:
        """Scales of the wavelets."""
        return (self.omega0s * self.sampling_freq) / (2.0 * PI * self.center_freqs)

    @property
    def waveforms(self) -> torch.Tensor:
        """Return the values of the wavelets in the time domain.

        Returns
        -------
        waveforms : complex torch.Tensor of shape (n_center_freqs, n_times)
            Wavelets in the time domain.
        """
        # Gaussian envelope
        gaussian = torch.exp(-4.0 * Ln2 * (self.times / self.time_widths[:, None]) ** 2)
        # Oscillatory part
        oscillation = torch.exp(1j * 2.0 * PI * self.center_freqs[:, None] * self.times)

        return gaussian * oscillation

    @property
    def spectral_max_amps(self) -> torch.Tensor:
        """Maximum amplitudes of the Fourier spectra of the wavelets."""
        return 0.5 * math.sqrt(PI / Ln2) * self.time_widths

    def forward(
        self,
        data: torch.Tensor | NDArray,
        demean: bool = True,
        tukey_alpha: float | None = 0.1,
        coeff_type: Literal["power", "magnitude", "complex"] = "power",
    ) -> torch.Tensor:
        """Compute the wavelet transform of the input signal(s).

        Parameters
        ----------
        data : torch.Tensor or ndarray of shape (..., n_times)
            Input signal(s) to be analyzed.
        demean : bool, default=True
            Whether to demean the input signal(s) before computing the
            wavelet transform.
        tukey_alpha : float or None, default=0.1
            Alpha parameter for the Tukey window. If None, no windowing is
            applied.
        mode : {'power', 'magnitude', 'complex'}, default='power'
            Specifies the type of the returned values:
                - `'power'`: squared magnitude of the coefficients.
                - `'magnitude'`: absolute magnitude of the coefficients.
                - `'complex'`: complex-valued coefficients.

        Returns
        -------
        coeffs : torch.Tensor of shape (..., n_center_freqs, n_times)
            Wavelet-transform coefficients.

        Notes
        -----
        The shape of the output depends on the shape of the input signal(s):
            - `F`: number of center frequencies (wavelets)
            - `B`: batch size
            - `C`: number of channels
            - `L`: number of time points

            | Input shape | Output shape   |
            |-------------|----------------|
            | `(L,)`      | `(F, L)`       |
            | `(B, L)`    | `(B, F, L)`    |
            | `(C, L)`    | `(C, F, L)`    |
            | `(B, C, L)` | `(B, C, F, L)` |
        """
        if coeff_type not in {"power", "magnitude", "complex"}:
            raise ValueError(
                f"Invalid coeff_type: {coeff_type}. "
                "Expected one of 'power', 'magnitude', or 'complex'."
            )

        device = self._get_device()
        dtype = self._get_dtype()

        x_in = torch.as_tensor(data, dtype=dtype, device=device)

        if demean:
            x_in = x_in - x_in.mean(dim=-1, keepdim=True)

        if tukey_alpha is not None:
            window = tukey_window(x_in.shape[-1], tukey_alpha, device=device)
            x_in = x_in * window

        wt_coeffs = _cwt_via_fft(x_in, self.waveforms, hermitian=True)
        # Normalize by the scales
        wt_coeffs = wt_coeffs / torch.sqrt(self.scales[:, None])

        if coeff_type == "power":
            wt_coeffs = torch.square(torch.abs(wt_coeffs))
        elif coeff_type == "magnitude":
            wt_coeffs = torch.abs(wt_coeffs)
        # If 'mode=complex', do nothing. The coeffs are already complex.
        return wt_coeffs

    def magnitude_responses(
        self,
        normalize: bool = True,
        n_fft: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the frequency responses of the wavelets.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to return the normalized responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.

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

        # Compute frequency points
        rfreqs = torch.fft.rfftfreq(
            n=n_fft, d=self.delta_t, device=self.device, dtype=self.dtype
        )

        # Compute responses
        phase_diffs = 2.0 * PI * (rfreqs - self.center_freqs[:, None])
        resps = torch.exp(
            -1.0 * torch.square(self.time_widths[:, None] * phase_diffs) / (16.0 * Ln2)
        )

        if not normalize:
            resps = resps * self.spectral_max_amps[:, None]

        return rfreqs, resps

    def plot_responses(
        self,
        ax: MplAxes,
        normalize: bool = True,
        n_fft: int | None = None,
        auto_xlabel: bool = True,
        auto_ylabel: bool = True,
        auto_title: bool = True,
    ) -> MplAxes:
        """Plot the frequency responses of the wavelets using Matplotlib.

        Parameters
        ----------
        ax : Axes
            The Matplotlib axes to plot the frequency responses.
        normalize : bool, default=True
            Whether to plot the normalized responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.
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
        freqs, resps = self.magnitude_responses(normalize=normalize, n_fft=n_fft)

        # Convert to numpy for plotting
        freqs_np = freqs.detach().cpu().numpy()
        resps_np = resps.detach().cpu().numpy()

        for resp in resps_np:
            ax.plot(freqs_np, resp)

        if auto_xlabel:
            ax.set_xlabel("Frequency [Hz]")
        if auto_ylabel:
            ax.set_ylabel("Magnitude, normalized" if normalize else "Magnitude")
        if auto_title:
            ax.set_title("Wavelets Frequency Responses")
        return ax

    def plot_responses_plotly(
        self,
        fig: PlotlyFigure,
        normalize: bool = True,
        n_fft: int | None = None,
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

        freqs, resps = self.magnitude_responses(normalize=normalize, n_fft=n_fft)

        # Convert to numpy for plotting
        freqs_np = freqs.detach().cpu().numpy()
        resps_np = resps.detach().cpu().numpy()

        for resp in resps_np:
            fig.add_trace(go.Scatter(x=freqs_np, y=resp, showlegend=False))

        if auto_xlabel:
            fig.update_xaxes(title_text="Frequency [Hz]")
        if auto_ylabel:
            fig.update_yaxes(
                title_text="Magnitude, normalized" if normalize else "Magnitude"
            )
        if auto_title:
            fig.update_layout(title="Wavelets Frequency Responses")
        return fig
