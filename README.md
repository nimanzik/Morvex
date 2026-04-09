# Morvex

**Continuous Wavelet Transform (CWT) using Morlet wavelets filter bank,**
**built on PyTorch with GPU support.**

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
French geophysicist and pioneer of wavelet theory, which defines wavelet shape
through a **shape ratio** parameter $\kappa$ – the Gaussian time width as an integer
multiple of the wavelet's dominant period. This preserves wavelet shape across
frequencies and makes the constant-Q property explicit.

Built entirely on PyTorch, Morvex runs on both GPU and CPU with no code changes
required.

## Installation

### Pfrerequisite

Install [uv](https://github.com/astral-sh/uv) Python packagfe manager.

### Core installation (without PyTorch)

If you already have PyTorch installed, and need only the core functionality of
Morvex, run the following command:

```bash
uv add git+https://github.com/nimanzik/Morvex
```

### Full installation (with PyTorch)

Morvex provides optional extras for different PyTorch configurations (CUDA
versions and CPU-only). Install the appropriate extra based on your setup:

- CPU-only

```bash
uv add git+https://github.com/nimanzik/Morvex --extra torch-cpu
```

- CUDA 13.0

```bash
uv add git+https://github.com/nimanzik/Morvex --extra torch-cu130
```

- CUDA 12.8

```bash
uv add git+https://github.com/nimanzik/Morvex --extra torch-cu128
```

- CUDA 12.6

```bash
uv add git+https://github.com/nimanzik/Morvex --extra torch-cu126
```

## Quick start

> [!NOTE]
> The units of `time_duration` and `sampling_freq` must be compatible
> (e.g., seconds and Hz, milliseconds and kHz etc).

### Filter bank and CWT

```python
import torch

from morvex import MorletFilterBank

# Build a constant-Q filter bank
fbank = MorletFilterBank(
    n_octaves=4,           # Number of octaves
    resolution=8,          # Number of filters per octave
    shape_ratio=5.0,       # Shape ratio (kappa)
    time_duration=2.0,     # Wavelet time duration (here in seconds)
    sampling_freq=1000.0,  # Sampling frequency (here in Hz)
)

# Compute the wavelet transform (scalogram) of an 8-second signal
signal = torch.randn(8000)
scalogram = fbank(signal, coeff_type="power")  # shape: (n_wavelets, 8000) 
```

### Batch processing

The forward pass supports arbitrary leading dimensions for batch processing. For
example, to compute the CWT of a batch of 16 stereo signals (3 channels, 8
seconds each):

```python
signals = torch.randn(16, 3, 8000)
scalogram = fbank(
    signals,
    coeff_type="magnitude",
)  # shape: (16, 3, n_wavelets, 8000)
```

### GPU acceleration

Since Morvex is a standard `torch.nn.Module`, moving the filter bank and input
data to GPU is straightforward:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fbank = fbank.to(device)
signal = signal.to(device)
scalogram = fbank(signal, coeff_type="power")  # Computed on GPU if available
```

### Visualisation

To display the frequency responses of the filter bank:

```python
import matplotlib.pyplot as plt

from morvex.plotting import plot_freq_resps

fig, ax = plt.subplots()
plot_freq_resps(fbank, plot_obj=ax, color="skyblue")
plt.show()
```

To display the scalogram in the time-frequency plane:

```python
import numpy as np

from morvex.plotting import plot_time_freq_plane

scalogram = scalogram.cpu().numpy()  # Move to CPU for plotting
freqs = fbank.center_freqs.cpu().numpy()
times = np.arange(scalogram.shape[-1]) / fbank.sampling_freq

fig, ax = plt.subplots()
plot_time_freq_plane(ax, freqs, times, scalogram, log_scale=True)
plt.show()
```

## Troubleshooting

Report issues or bugs on [GitHub Issues](https://github.com/nimanzik/Morvex/issues).
