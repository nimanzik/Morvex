"""Plotting time-frequency (TF) plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeGuard

import numpy as np
from cmcrameri import cm
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray
    from plotly.graph_objects import Figure as PlotlyFigure


PlotBackend: TypeAlias = Literal["matplotlib", "plotly"]


@dataclass
class FreqRespPlotConfig:
    n_fft: int | None = None
    scaled: bool = False
    color: str | None = None
    auto_xlabel: bool = True
    auto_ylabel: bool = True
    auto_title: bool = True
    backend: PlotBackend | None = None


def plot_freq_resps(
    filter_bank,
    plot_obj: MplAxes | PlotlyFigure,
    n_fft: int | None = None,
    scaled: bool = False,
    color: str | None = None,
    auto_xlabel: bool = True,
    auto_ylabel: bool = True,
    auto_title: bool = True,
    backend: PlotBackend | None = None,
) -> MplAxes | PlotlyFigure:
    """Plot wavelet frequency responses using Matplotlib or Plotly.

    Parameters
    ----------
    filter_bank : FilterBank
        The filter bank to plot the frequency responses.
    plot_obj : Axes or PlotlyFigure
        Matplotlib axes or Plotly figure to draw on.
    n_fft : int or None, default=None
        Number of FFT points to use for computing the frequency responses.
        If None, the next power of two greater than or equal to `n_t`
        will be used.
    scaled : bool, default=False
        Whether to plot the scaled responses. If False, the responses
        are normalised to one.
    color : str or None, default=None
        Line color for all responses.
    auto_xlabel : bool, default=True
        Whether to automatically set the x-axis label.
    auto_ylabel : bool, default=True
        Whether to automatically set the y-axis label.
    auto_title : bool, default=True
        Whether to automatically set the title.
    backend : {"matplotlib", "plotly"} or None, default=None
        Plotting backend. If None, inferred from `plot_obj`.

    Returns
    -------
    out : Axes or PlotlyFigure
        The same plotting object passed in `plot_obj`.
    """
    backend = backend or _infer_plot_backend(plot_obj)
    cfg = FreqRespPlotConfig(
        n_fft=n_fft,
        scaled=scaled,
        color=color,
        auto_xlabel=auto_xlabel,
        auto_ylabel=auto_ylabel,
        auto_title=auto_title,
        backend=backend,
    )

    freqs, resps = _fetch_freq_resps(filter_bank, n_fft=cfg.n_fft, scaled=cfg.scaled)

    if cfg.backend == "matplotlib":
        if not _is_mpl_axes(plot_obj):
            raise TypeError(
                "`plot_obj` must be a Matplotlib Axes when backend='matplotlib'."
            )
        return _plot_freq_resps_mpl(plot_obj, freqs=freqs, resps=resps, config=cfg)

    if cfg.backend == "plotly":
        if not _is_plotly_figure(plot_obj):
            raise TypeError("`plot_obj` must be a Plotly Figure when backend='plotly'.")
        return _plot_freq_resps_plotly(plot_obj, freqs=freqs, resps=resps, config=cfg)

    raise ValueError("`backend` must be one of ('matplotlib', 'plotly').")


def _fetch_freq_resps(
    filter_bank, n_fft: int | None, scaled: bool
) -> tuple[NDArray, NDArray]:
    freqs_, resps_ = filter_bank.compute_freq_resps(n_fft, scaled=scaled)
    return freqs_.detach().cpu().numpy(), resps_.detach().cpu().numpy()


def _is_mpl_axes(obj: Any) -> TypeGuard[MplAxes]:
    return hasattr(obj, "plot") and hasattr(obj, "set_xlabel")


def _is_plotly_figure(obj: Any) -> TypeGuard[PlotlyFigure]:
    return hasattr(obj, "add_trace") and hasattr(obj, "update_xaxes")


def _infer_plot_backend(plot_obj: Any) -> PlotBackend:
    if _is_mpl_axes(plot_obj):
        return "matplotlib"
    if _is_plotly_figure(plot_obj):
        return "plotly"
    raise TypeError(
        "Could not infer backend from `plot_obj`. "
        "Pass backend as 'matplotlib' or 'plotly'."
    )


def _plot_freq_resps_mpl(
    ax: MplAxes, freqs: NDArray, resps: NDArray, config: FreqRespPlotConfig
) -> MplAxes:
    for resp in resps:
        ax.plot(freqs, resp, color=config.color)

    if config.auto_xlabel:
        ax.set_xlabel("Frequency")
    if config.auto_ylabel:
        ax.set_ylabel("Amplitude" if config.scaled else "Amplitude (normalised)")
    if config.auto_title:
        ax.set_title("Wavelets Frequency Responses")
    return ax


def _plot_freq_resps_plotly(
    fig: PlotlyFigure, freqs: NDArray, resps: NDArray, config: FreqRespPlotConfig
) -> PlotlyFigure:
    from plotly import graph_objects as go

    line_kwargs = {"line": {"color": config.color}} if config.color is not None else {}
    for resp in resps:
        fig.add_trace(go.Scatter(x=freqs, y=resp, showlegend=False, **line_kwargs))

    if config.auto_xlabel:
        fig.update_xaxes(title_text="Frequency")
    if config.auto_ylabel:
        fig.update_yaxes(
            title_text="Amplitude" if config.scaled else "Amplitude (normalised)"
        )
    if config.auto_title:
        fig.update_layout(title="Wavelets Frequency Responses")
    return fig


def plot_time_freq_plane(
    ax: MplAxes,
    freqs: NDArray,
    times: NDArray,
    xgram: NDArray,
    label: str,
    log_scale: bool = False,
    cmap: str | Colormap | None = None,
    auto_xlabel: bool = True,
    auto_ylabel: bool = True,
    auto_cbar_label: bool = True,
) -> MplAxes:
    """Plot time-frequency (TF) plane.

    Parameters
    ----------
    ax : Axes
        The Matplotlib axes to plot the TF plane.
    freqs : ndarray
        The array of sample frequencies.
    times : ndarray
        The array of segment times.
    xgram : ndarray
        The TF representation of the signal.
    label : str
        The mode of the TF representation (e.g., 'psd', 'magnitude'). This is
        used as the label for the colorbar.
    log_scale : bool, default=False
        Whether to plot the TF plane in decibel (dB) scale.
    cmap : str or Colormap or None, default=None
        The colormap to use for the TF plane.

    Returns
    -------
    ax : MplAxes
        Matplotlib axes displaying the TF plane.
    """
    if xgram.ndim != 2:
        raise ValueError(
            f"Time-frequency representation input must be 2D, but got {xgram.ndim}D."
        )

    if freqs.shape + times.shape != xgram.shape:
        raise ValueError(
            f"The shapes of `freqs`, `times`, and input time-frequency representation "
            f"do not match: {freqs.shape} + {times.shape} != {xgram.shape}."
        )

    # Downsample the frequency dimension if it is too large
    freqs, xgram = _downsample_freq_dim(freqs, xgram)

    # Convert to decibel scale if needed
    if log_scale:
        xgram = _to_decibel(xgram)

    # Get the min and max values for the colorbar
    v_min, v_max = _get_vrange(xgram)
    cmap = cmap or cm.cmaps["lipari"]

    if xgram.shape[-1] > 1000:
        # Use `imshow` for large data
        im = ax.imshow(
            xgram,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=(times[0], times[-1], 0, len(freqs) - 1),
            vmin=v_min,
            vmax=v_max,
        )
        cbar = ax.figure.colorbar(mappable=im, ax=ax)
        y_ticks = np.linspace(0, len(freqs) - 1, num=8, dtype=int)
        y_tick_labels = [f"{freqs[i]:.0f}" for i in y_ticks]
        ax.set_yticks(ticks=y_ticks, labels=y_tick_labels)
    else:
        # Use `pcolormesh` for small data
        pc = ax.pcolormesh(
            times,
            freqs,
            xgram,
            cmap=cmap,
            shading="gouraud",
            vmin=v_min,
            vmax=v_max,
        )
        cbar = ax.figure.colorbar(mappable=pc, ax=ax)
    if auto_xlabel:
        ax.set_xlabel("Time")
    if auto_ylabel:
        ax.set_ylabel("Frequency")

    if auto_cbar_label:
        clabel = label.upper() if label == "psd" else label.capitalize()
        cbar.set_label(f"{clabel} (dB)" if log_scale else clabel)

    return ax


def _downsample_plan(n_original: int, n_max: int) -> tuple[int, int]:
    n_mean = (n_original - 1) // n_max + 1
    n_ds = n_original // n_mean
    return n_ds, n_mean


def _downsample_freq_dim(freqs: NDArray, xgram: NDArray) -> tuple[NDArray, NDArray]:
    max_num_freqs = 500

    n_freqs, n_times = xgram.shape
    n_freqs_ds, n_freqs_mean = _downsample_plan(n_freqs, max_num_freqs)

    if n_freqs_mean == 1:
        return freqs, xgram

    n_freqs_trim = n_freqs_ds * n_freqs_mean
    return (
        freqs[:n_freqs_trim:n_freqs_mean],
        xgram[:n_freqs_trim].reshape(n_freqs_ds, n_freqs_mean, n_times).mean(axis=1),
    )


def _to_decibel(a: NDArray, tiny: float = 1e-10) -> NDArray:
    return 10 * np.log10(np.fmax(a, tiny))


def _get_vrange(xgram: NDArray) -> tuple[float, float]:
    mu = xgram.mean()
    sigma = xgram.std(ddof=1)
    v_min = max(mu - 3 * sigma, xgram.min())
    v_max = min(mu + 3 * sigma, xgram.max())
    return v_min, v_max
