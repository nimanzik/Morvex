"""Cross-correlation via FFT."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.fft import next_fast_len

_CACHED_NEXT_FAST_LEN: dict[int, int] = {}


@lru_cache(maxsize=256)
def _next_fast_len(n: int, real: bool) -> int:
    """Cache results of next_fast_len for performance and thread safety.

    See `scipy.fft.next_fast_len` for documentation.

    Parameters
    ----------
    n : int
        Length of the input sequence (to start searching from).
    real : bool
        Set to True if FFT involves real-valued input or output (i.e., rfft
        and irfft), False for complex-valued FFT (i.e., fft and ifft).
    """
    return next_fast_len(n, real=real)


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
    x_out : torch.Tensor
        Centered tensor with the new shape.
    """
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
    data : torch.Tensor of shape (..., signal_length)
        Input data to be analysed.
    waveforms : torch.Tensor of shape (n_waveforms, waveform_length)
        Waveforms (of a wavelet group, for example) to be cross-correlated
        with `data`.

    Returns
    -------
    coeffs : torch.Tensor of shape (..., n_waveforms, signal_length)
        Coefficients of the cross-correlation between `data` and each waveform
        in `waveforms` group.
    """
    if waveforms.ndim != 2:
        raise ValueError(
            f"`waveforms` must be a 2D tensor of shape "
            f"(n_waveforms, waveform_length), but got shape {waveforms.shape}",
        )

    is_complex = data.is_complex() or waveforms.is_complex()

    # xcorr -> conv(mode='full')
    n_conv = data.shape[-1] + waveforms.shape[-1] - 1
    n_fft = _next_fast_len(n_conv, real=not is_complex)

    if is_complex:
        fft_, ifft_ = torch.fft.fft, torch.fft.ifft
    else:
        fft_, ifft_ = torch.fft.rfft, torch.fft.irfft

    filter_spectra = torch.conj(fft_(waveforms, n=n_fft))

    # FFT(data) and add dimension for wavelets: (..., n_fft) -> (..., 1, n_fft)
    data_spectra = fft_(data, n=n_fft).unsqueeze(-2)

    # For iFFT, 'n' controls the output length, not the transform length.
    coeffs = ifft_(filter_spectra * data_spectra, n=n_fft)[..., :n_conv]

    # Center with respect to the mode-'full' convolution
    final_shape = coeffs.shape[:-1] + (data.shape[-1],)
    return _get_centered(coeffs, final_shape)
