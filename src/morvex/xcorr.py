"""Cross-correlation via FFT."""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import numpy as np
import torch
from scipy.fft import next_fast_len

__all__ = ["xcorr_via_fft"]


@lru_cache(maxsize=256)
def _next_fast_len(n: int, real: bool) -> int:
    """Cache results of `next_fast_len` for performance.

    See `scipy.fft.next_fast_len` for more details.

    Parameters
    ----------
    n : int
        Length of the input sequence (to start searching from).
    real : bool
        Set to True if FFT involves real-valued input or output (i.e., rfft
        and irfft), False for complex-valued FFT (i.e., fft and ifft).
    """
    return cast("int", next_fast_len(n, real=real))


def _get_centered(x: torch.Tensor, new_shape: tuple[int, ...]) -> torch.Tensor:
    """Return the center newshape portion of a tensor.

    Adapted from: https://github.com/scipy/scipy/blob/main/scipy/signal/_signaltools.py#L411

    Parameters
    ----------
    x : Tensor
        Input tensor.
    new_shape : tuple of int
        Desired shape of the output tensor.

    Returns
    -------
    x_out : Tensor
        Centered tensor with the new shape.
    """
    if len(new_shape) != x.ndim:
        raise ValueError(
            f"`new_shape` must have the same number of dimensions as `x`, but "
            f"got {len(new_shape)} and {x.ndim}"
        )

    if any(s > x.shape[k] for k, s in enumerate(new_shape)):
        raise ValueError(
            f"Each dimension in `new_shape` must be less than or equal to the "
            f"corresponding dimension in `x`, but got {new_shape} and {x.shape}"
        )

    output_shape = np.asarray(new_shape)
    current_shape = np.asarray(x.shape)
    start_idx = (current_shape - output_shape) // 2
    end_idx = start_idx + output_shape
    slice_idxs = [slice(start_idx[k], end_idx[k]) for k in range(len(end_idx))]
    return x[tuple(slice_idxs)]


def xcorr_via_fft(data: torch.Tensor, waveforms: torch.Tensor) -> torch.Tensor:
    """Apply cross-correlation between `data` and `waveforms` using FFT.

    Parameters
    ----------
    data : Tensor of shape (..., signal_length)
        Input data to be analysed.
    waveforms : Tensor of shape (n_waveforms, waveform_length)
        Waveforms (of a wavelet group, for example) to be cross-correlated
        with `data`.

    Returns
    -------
    coeffs : Tensor of shape (..., n_waveforms, signal_length)
        Coefficients of the cross-correlation between `data` and each waveform
        in `waveforms` group.
    """
    if waveforms.ndim != 2:
        raise ValueError(
            f"`waveforms` must be a 2D tensor of shape "
            f"(n_waveforms, waveform_length), but got shape {waveforms.shape}"
        )

    if data.ndim < 1:
        raise ValueError(
            f"`data` must be at least 1D tensor of shape (..., signal_length), "
            f"but got shape {data.shape}"
        )

    if data.device != waveforms.device:
        raise ValueError(
            f"`data` and `waveforms` must be on the same device, but got "
            f"{data.device} and {waveforms.device}"
        )

    if data.real.dtype != waveforms.real.dtype:
        raise ValueError(
            f"dtype precision mismatch between `data` and `waveforms`: "
            f"got {data.real.dtype} and {waveforms.real.dtype}"
        )

    is_complex = data.is_complex() or waveforms.is_complex()

    # xcorr -> conv(mode='full')
    n_conv = data.shape[-1] + waveforms.shape[-1] - 1
    n_fft = _next_fast_len(n_conv, real=not is_complex)

    if is_complex:
        _fft, _ifft = torch.fft.fft, torch.fft.ifft
    else:
        _fft, _ifft = torch.fft.rfft, torch.fft.irfft

    filter_spectra = _fft(waveforms.flip(dims=(-1,)).conj(), n=n_fft)

    # FFT(data) and add dimension for wavelets: (..., n_fft) -> (..., 1, n_fft)
    data_spectra = _fft(data, n=n_fft).unsqueeze(-2)

    coeffs = _ifft(filter_spectra * data_spectra, n=n_fft)[..., :n_conv]

    # Center with respect to the mode-'full' convolution
    final_shape = coeffs.shape[:-1] + (data.shape[-1],)
    return _get_centered(coeffs, final_shape)
