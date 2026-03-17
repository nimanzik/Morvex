# Morvex

**A lightweight Python library for continuous wavelet transform (CWT) using complex Morlet wavelets, built on PyTorch with GPU support.**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/nimanzik/Morvex/actions/workflows/ci.yml/badge.svg)](https://github.com/nimanzik/Morvex/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!CAUTION]
> This project is under active development and may undergo significant changes.

## Overview

This Python library provides an implementation of the Morlet wavelet transform
for time-frequency analysis. The implementation follows the original,
physically intuitive formulation by [Jean Morlet](https://en.wikipedia.org/wiki/Jean_Morlet),
which defines wavelet shape through a **shape ratio** parameter – the Gaussian
time width as an integer multiple of the wavelet's dominant period. This
preserves wavelet shape across frequencies and makes the constant-Q property explicit.

Built entirely on PyTorch, Morvex runs on both CPU and GPU with no code changes
required.

## Features

- **Morlet wavelet transform** with constant-Q filter bank
- **GPU acceleration** via PyTorch: move to GPU with a single `.to(device=...`)
- **Batch processing** — transform multiple signals in one call
- **Flexible output** — power, magnitude, or complex coefficients
- **Built-in visualisation** with Matplotlib and Plotly backends
- **Configurable tapering** to reduce edge artefacts

## Installation

With [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv add git+https://github.com/nimanzik/Morvex
```

With pip (from source):

```bash
git clone https://github.com/nimanzik/Morvex.git
cd Morvex
pip install .
```

### PyTorch with CUDA

Morvex provides optional extras for installing PyTorch with the appropriate backend:

```bash
# CPU-only
uv add git+https://github.com/nimanzik/Morvex --extra torch-cpu

# CUDA 12.8
uv add git+https://github.com/nimanzik/Morvex --extra torch-cu128
```

## Quick start

### Single wavelet

```python
from morvex import MorletWavelet

wavelet = MorletWavelet(
    center_freq=5.0,      # Center frequency
    shape_ratio=5.0,      # Shape ratio (kappa)
    time_duration=2.0,    # Duration in seconds
    sampling_freq=100.0,  # Sampling frequency in Hz
)

print(wavelet.time_width)   # Gaussian time width
print(wavelet.freq_width)   # Frequency bandwidth
print(wavelet.waveform)     # Complex-valued waveform tensor
```

### Filter bank and CWT

```python
import torch
from morvex import MorletFilterBank

# Build a constant-Q filter bank
fb = MorletFilterBank(
    n_octaves=4,          # Number of octaves
    resolution=8,         # Filters per octave
    shape_ratio=5.0,      # Shape ratio (kappa)
    time_duration=2.0,    # Duration in seconds
    sampling_freq=1000.0, # Sampling frequency in Hz
)

# Compute the wavelet transform
signal = torch.randn(8000)                    # 8 seconds of signal
coeffs = fb(signal, coeff_type="power")       # Scalogram (power)
# coeffs shape: (n_wavelets, 8000)
```

### Batch processing

The forward pass supports arbitrary leading dimensions:

```python
batch = torch.randn(16, 3, 8000)             # (batch, channels, time)
coeffs = fb(batch, coeff_type="magnitude")    # (16, 3, n_wavelets, 8000)
```

### GPU acceleration

Since Morvex is a standard `torch.nn.Module`, moving to GPU is straightforward:

```python
fb = fb.cuda()
signal = signal.cuda()
coeffs = fb(signal, coeff_type="power")       # Computed on GPU
```

### Visualisation

Plot the frequency responses of the filter bank:

```python
import matplotlib.pyplot as plt
from morvex.plotting import plot_freq_resps

fig, ax = plt.subplots()
plot_freq_resps(fb, ax)
plt.show()
```

Plot a scalogram (time-frequency plane):

```python
import numpy as np
from morvex.plotting import plot_time_freq_plane

freqs = fb.center_freqs.numpy()
times = np.arange(coeffs.shape[-1]) / fb.sampling_freq
scalogram = coeffs.numpy()

fig, ax = plt.subplots()
plot_time_freq_plane(ax, freqs, times, scalogram, label="power", log_scale=True)
plt.show()
```

## API reference

### `MorletWavelet`

A single complex Morlet wavelet defined by its center frequency, shape ratio, time duration, and sampling frequency.

| Parameter | Type | Description |
|---|---|---|
| `center_freq` | `float` | Center frequency of the wavelet |
| `shape_ratio` | `float` | Shape ratio (kappa) — controls the time-frequency trade-off |
| `time_duration` | `float` | Time duration of the wavelet |
| `sampling_freq` | `float` | Sampling frequency |

### `MorletFilterBank`

A constant-Q filter bank of complex Morlet wavelets, spaced logarithmically across octaves.

| Parameter | Type | Description |
|---|---|---|
| `n_octaves` | `int` | Number of octaves to cover |
| `resolution` | `int` | Number of wavelets per octave |
| `shape_ratio` | `float` | Shape ratio (kappa) — common for all wavelets |
| `time_duration` | `float` | Time duration, common for all wavelets |
| `sampling_freq` | `float` | Sampling frequency |

**Forward pass**: `fb(data, taper=None, coeff_type="power")` returns wavelet coefficients.

| `coeff_type` | Output |
|---|---|
| `"power"` | Squared magnitude (real-valued) |
| `"magnitude"` | Absolute magnitude (real-valued) |
| `"complex"` | Complex-valued coefficients |

> [!NOTE]
> The units of `time_duration` and `sampling_freq` must be compatible (e.g., seconds and Hz, milliseconds and kHz).

## Troubleshooting

Report issues or bugs on [GitHub Issues](https://github.com/nimanzik/Morvex/issues).
