# Morvex

**Continuous Wavelet Transform (CWT) using Morlet wavelets filter bank, built on PyTorch with GPU support.**

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
- **GPU acceleration** via PyTorch: move to GPU with a single `.to(device=...)`
- **Batch processing**: transform multiple signals in one call

## Installation

### Pfrerequisite

Install [uv](https://github.com/astral-sh/uv) Python packagfe manager.

### Minimal dependencies: without PyTorch

If PyTorch is already installed in your virtual environment:

```bash
uv add git+https://github.com/nimanzik/Morvex
```

### Full dependencies: PyTorch with CUDA or CPU only

Morvex provides optional extras for installing PyTorch with the appropriate backend:

- CUDA 12.8

```bash
uv add git+https://github.com/nimanzik/Morvex --extra torch-cu128
```

- CPU-only

```bash
uv add git+https://github.com/nimanzik/Morvex --extra torch-cpu
```

## Quick start

> [!NOTE]
> The units of `time_duration` and `sampling_freq` must be compatible (e.g., seconds and Hz, milliseconds and kHz etc).

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

## Troubleshooting

Report issues or bugs on [GitHub Issues](https://github.com/nimanzik/Morvex/issues).
