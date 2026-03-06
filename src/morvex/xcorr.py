"""Cross-correlation via FFT."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.fft import next_fast_len

_CACHED_NEXT_FAST_LEN: dict[int, int] = {}


def _reverse_and_conj(
    x: torch.Tensor, dims: list[int] | tuple[int, ...] | None = None
) -> torch.Tensor:
    """Reverse `x` along specified dimensions and take the complex conjugate.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dims : sequence of int or None, default=None
        Dimensions along which to reverse. If None, input is reversed in all
        dimensions.

    Returns
    -------
    x_dagger : torch.Tensor
        The reversed and conjugated tensor.
    """
    if dims is None:
        dims = list(range(x.ndim))

    # Torch's flip is a copy and not a view
    x_rev = torch.flip(x, dims=dims)

    return torch.conj(x_rev) if x.is_complex() else x_rev


def _get_centered(x: torch.Tensor, new_shape: tuple[int, ...]) -> torch.Tensor:
    """Return the center newshape portion of a tensor.

    Adapted from: https://github.com/scipy/scipy/blob/main/scipy/signal/_signaltools.py#L411

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    new_shape : tuple of int
        Desired shape of the output tensor.

    Returns
    -------
    x_centered : torch.Tensor
        Centered tensor with the new shape.
    """
    output_shape = np.asarray(new_shape)
    current_shape = np.asarray(x.shape)
    start_idx = (current_shape - output_shape) // 2
    end_idx = start_idx + output_shape
    slice_idxs = [slice(start_idx[k], end_idx[k]) for k in range(len(end_idx))]
    return x[tuple(slice_idxs)]


@lru_cache(maxsize=256)
def _next_fast_len(n: int, real: bool) -> int:
    """Cache results of next_fast_len for performance and thread safety.

    See `scipy.fft.next_fast_len` for documentation.
    """
    return next_fast_len(n, real=real)


def xcorr_via_fft(data: torch.Tensor, waveforms: torch.Tensor) -> torch.Tensor:
    """Apply cross-correlation between `data` and `waveforms` using FFT.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples)
        Input data to be analysed.
    waveforms : torch.Tensor of shape (n_waveforms, n_samples)
        Waveforms (of a wavelet group, for example) to be cross-correlated
        with `data`.

    Returns
    -------
    coeffs : torch.Tensor of shape (..., n_waveforms, n_samples)
        Coefficients of the cross-correlation between `data` and each waveform
        in `waveforms` group.
    """
    is_complex = data.is_complex() or waveforms.is_complex()

    # xcorr -> conv(mode='full')
    n_conv = data.shape[-1] + waveforms.shape[-1] - 1
    n_fft = _next_fast_len(n_conv, real=not is_complex)

    if is_complex:
        fft_, ifft_ = torch.fft.fft, torch.fft.ifft
    else:
        fft_, ifft_ = torch.fft.rfft, torch.fft.irfft

    # github.com/scipy/scipy/blob/main/scipy/signal/_signaltools.py#L257
    filter_spectra = fft_(_reverse_and_conj(waveforms, dims=[-1]), n=n_fft)

    # FFT(data) and add dimension for wavelets: (..., n_fft) -> (..., 1, n_fft)
    data_spectra = fft_(data, n=n_fft).unsqueeze(-2)

    coeffs = ifft_(filter_spectra * data_spectra, n=n_fft)[..., :n_conv]

    # Center with respect to the mode-'full' convolution
    final_shape = coeffs.shape[:-1] + (data.shape[-1],)
    return _get_centered(coeffs, final_shape)
