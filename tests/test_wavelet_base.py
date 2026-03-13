from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from morvex._wavelet_base import (
    LN2,
    PI,
    CoeffType,
    _coerce_validate_center_freqs,
    _coerce_validate_shape_ratios,
    _get_default_taper,
    _MorletWaveletBase,
    _preprocess_input,
)
from morvex.tapering import Taper


class TestCoerceValidateCenterFreqs:
    def test_accepts_valid_input(self) -> None:
        center_freqs = _coerce_validate_center_freqs(
            center_freqs=[5.0, 12.5],
            sampling_freq=100.0,
            dtype=torch.float64,
        )

        assert center_freqs.dtype == torch.float64
        assert torch.allclose(
            center_freqs, torch.tensor([5.0, 12.5], dtype=torch.float64)
        )

    @pytest.mark.parametrize(
        ("center_freqs", "message"),
        [
            ([], "non-empty"),
            ([[1.0, 2.0]], "1-dimensional"),
            ([1.0, float("nan")], "finite"),
            ([float("inf")], "finite"),
            ([float("-inf")], "finite"),
            ([0.0, 1.0], "range"),
            ([-5.0], "range"),
            ([50.0], "range"),
        ],
    )
    def test_rejects_invalid_values(self, center_freqs: list, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            _coerce_validate_center_freqs(
                center_freqs=center_freqs, sampling_freq=100.0
            )


class TestCoerceValidateShapeRatios:
    def test_accepts_scalar(self) -> None:
        kappa = 7.0
        center_freqs = torch.tensor([5.0, 10.0])
        shape_ratios = _coerce_validate_shape_ratios(kappa, center_freqs)

        assert shape_ratios.numel() == 1
        assert shape_ratios.dtype == center_freqs.dtype
        assert shape_ratios.item() == pytest.approx(kappa)

    def test_accepts_per_wavelet_array(self) -> None:
        center_freqs = torch.tensor([5.0, 10.0], dtype=torch.float64)
        shape_ratios = _coerce_validate_shape_ratios(
            shape_ratios=[5.0, 9.0], center_freqs=center_freqs
        )

        assert shape_ratios.shape == (2,)
        assert shape_ratios.dtype == torch.float64
        assert shape_ratios[0].item() == pytest.approx(5.0)
        assert shape_ratios[1].item() == pytest.approx(9.0)

    @pytest.mark.parametrize(
        ("shape_ratios", "message"),
        [
            ([1.0, float("inf")], "finite"),
            ([float("nan"), 2.0], "finite"),
            ([0.0, 1.0], "positive"),
            ([-1.0, 2.0], "positive"),
            ([1.0, 2.0, 3.0], "same length"),
        ],
    )
    def test_rejects_invalid_values(self, shape_ratios: list, message: str) -> None:
        center_freqs = torch.tensor([5.0, 10.0])

        with pytest.raises(ValueError, match=message):
            _coerce_validate_shape_ratios(
                shape_ratios=shape_ratios, center_freqs=center_freqs
            )


class TestGetDefaultTaper:
    def test_returns_cached_instance_for_equal_arguments(self) -> None:
        device = torch.device("cpu")

        t1 = _get_default_taper(100, torch.float32, device)
        t2 = _get_default_taper(100, torch.float32, device)
        t3 = _get_default_taper(101, torch.float32, device)

        assert t1 is t2
        assert t1 is not t3


class TestPreprocessInput:
    @pytest.mark.parametrize("data_len", [13, 73, 109, 512, 4096])
    def test_demeans_without_mutating_original_tensor(self, data_len: int) -> None:
        data = torch.rand(data_len, dtype=torch.float64)
        data_before = data.clone()
        taper = Taper("hann", n_samples=data_len, max_percentage=0.10, side="both")

        output = _preprocess_input(
            data=data, taper=taper, dtype=torch.float64, device=torch.device("cpu")
        )

        expected = taper(data_before - data_before.mean())

        assert output.dtype == torch.float64
        assert output.mean().item() == pytest.approx(0.0, abs=0.1)
        assert torch.allclose(output, expected)
        assert torch.equal(data, data_before)

    def test_handles_2d_input(self) -> None:
        """Preprocess should work on (C, L) shaped inputs."""
        data = torch.rand(3, 64, dtype=torch.float64)
        taper = Taper("hann", n_samples=64, max_percentage=0.10, side="both")

        output = _preprocess_input(
            data=data, taper=taper, dtype=torch.float64, device=torch.device("cpu")
        )

        assert output.shape == (3, 64)
        for ch in range(3):
            assert output[ch].mean().item() == pytest.approx(0.0, abs=0.1)

    def test_handles_3d_input(self) -> None:
        """Preprocess should work on (B, C, L) shaped inputs."""
        data = torch.rand(2, 3, 64, dtype=torch.float64)
        taper = Taper("hann", n_samples=64, max_percentage=0.10, side="both")

        output = _preprocess_input(
            data=data, taper=taper, dtype=torch.float64, device=torch.device("cpu")
        )

        assert output.shape == (2, 3, 64)

    def test_handles_numpy_input(self) -> None:
        data_np = np.random.randn(64).astype(np.float64)
        taper = Taper("hann", n_samples=64, max_percentage=0.10, side="both")

        output = _preprocess_input(
            data=data_np, taper=taper, dtype=torch.float64, device=torch.device("cpu")
        )

        assert isinstance(output, torch.Tensor)
        assert output.shape == (64,)
        assert output.dtype == torch.float64


class TestMorletWaveletBase:
    def test_properties_and_waveforms_have_expected_shapes(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0, 20.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=100.0,
            dtype=torch.float32,
        )

        assert len(wavelets) == 2
        assert wavelets.n_samples == 101
        assert wavelets.waveforms.shape == (2, 101)
        assert torch.is_complex(wavelets.waveforms)
        assert wavelets.times[0].item() == pytest.approx(-0.5)
        assert wavelets.times[-1].item() == pytest.approx(0.5)
        assert torch.allclose(
            wavelets.time_widths, torch.tensor([0.7, 0.35], dtype=torch.float32)
        )
        assert repr(wavelets) == "_MorletWaveletBase(nw=2, tau=1.0000, fs=100.0000)"

    def test_forward_returns_consistent_coeff_types(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[8.0, 16.0],
            shape_ratios=7.0,
            time_duration=0.5,
            sampling_freq=128.0,
        )
        data = torch.randn(256)

        coeffs_complex = wavelets(data, coeff_type=CoeffType.COMPLEX)
        coeffs_magnitude = wavelets(data, coeff_type="magnitude")
        coeffs_power = wavelets(data, coeff_type="power")

        assert coeffs_complex.shape == (2, 256)
        assert coeffs_magnitude.shape == (2, 256)
        assert coeffs_power.shape == (2, 256)
        assert torch.allclose(
            coeffs_complex.abs(), coeffs_magnitude, atol=1e-6, rtol=1e-5
        )
        assert torch.allclose(
            coeffs_magnitude.square(), coeffs_power, atol=1e-6, rtol=1e-5
        )
        assert torch.all(coeffs_power >= 0.0)

    def test_compute_freq_resps_scales_by_expected_amplitude(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[6.0, 12.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=64.0,
        )

        freqs, resps_scaled = wavelets.compute_freq_resps(n_fft=16, scaled=True)
        _, resps_unscaled = wavelets.compute_freq_resps(n_fft=16, scaled=False)

        assert freqs.shape[0] == 33
        assert resps_scaled.shape == (2, 33)
        assert resps_unscaled.shape == (2, 33)

        expected_scale = 0.5 * math.sqrt(PI / LN2) * wavelets.time_widths
        for i, center_freq in enumerate(wavelets.center_freqs):
            peak_idx = torch.argmin(torch.abs(freqs - center_freq))
            ratio = resps_scaled[i, peak_idx] / resps_unscaled[i, peak_idx]
            assert ratio.item() == pytest.approx(expected_scale[i].item(), rel=1e-6)

    def test_forward_with_invalid_coeff_type_raises(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0],
            shape_ratios=7.0,
            time_duration=0.5,
            sampling_freq=100.0,
        )
        data = torch.randn(64)

        with pytest.raises(ValueError):
            wavelets(data, coeff_type="invalid_type")

    def test_forward_with_explicit_taper(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0, 20.0],
            shape_ratios=7.0,
            time_duration=0.5,
            sampling_freq=100.0,
        )
        data = torch.randn(128)
        taper = Taper("hann", n_samples=128, max_percentage=0.10, side="both")

        coeffs = wavelets(data, taper=taper, coeff_type="power")

        assert coeffs.shape == (2, 128)
        assert torch.all(coeffs >= 0.0)

    def test_forward_batched_2d_input(self) -> None:
        """Test forward with (C, L) shaped input."""
        wavelets = _MorletWaveletBase(
            center_freqs=[8.0, 16.0],
            shape_ratios=7.0,
            time_duration=0.5,
            sampling_freq=128.0,
        )
        n_channels, signal_len = 3, 256
        data = torch.randn(n_channels, signal_len)

        coeffs = wavelets(data, coeff_type=CoeffType.COMPLEX)

        assert coeffs.shape == (n_channels, 2, signal_len)
        assert torch.is_complex(coeffs)

    def test_forward_batched_3d_input(self) -> None:
        """Test forward with (B, C, L) shaped input."""
        wavelets = _MorletWaveletBase(
            center_freqs=[8.0, 16.0],
            shape_ratios=7.0,
            time_duration=0.5,
            sampling_freq=128.0,
        )
        batch, n_channels, signal_len = 4, 3, 256
        data = torch.randn(batch, n_channels, signal_len)

        coeffs_power = wavelets(data, coeff_type="power")
        coeffs_mag = wavelets(data, coeff_type="magnitude")

        assert coeffs_power.shape == (batch, n_channels, 2, signal_len)
        assert coeffs_mag.shape == (batch, n_channels, 2, signal_len)
        assert torch.all(coeffs_power >= 0.0)
        assert torch.allclose(coeffs_mag.square(), coeffs_power, atol=1e-6, rtol=1e-5)

    def test_forward_with_numpy_input(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0],
            shape_ratios=7.0,
            time_duration=0.5,
            sampling_freq=100.0,
        )
        data_np = np.random.randn(128).astype(np.float64)

        coeffs = wavelets(data_np, coeff_type="power")

        assert coeffs.shape == (1, 128)
        assert torch.all(coeffs >= 0.0)

    def test_per_wavelet_shape_ratios(self) -> None:
        """Happy path: per-wavelet shape_ratios array."""
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0, 20.0],
            shape_ratios=[5.0, 9.0],
            time_duration=1.0,
            sampling_freq=100.0,
        )

        assert wavelets.shape_ratios.shape == (2,)
        assert wavelets.shape_ratios[0].item() == pytest.approx(5.0)
        assert wavelets.shape_ratios[1].item() == pytest.approx(9.0)
        assert wavelets.waveforms.shape == (2, 101)

    def test_delta_t(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=200.0,
        )
        assert wavelets.delta_t == pytest.approx(1.0 / 200.0)

    def test_device_property(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=100.0,
        )
        assert wavelets.device == torch.device("cpu")

    def test_dtype_property(self) -> None:
        for dt in (torch.float32, torch.float64):
            wavelets = _MorletWaveletBase(
                center_freqs=[10.0],
                shape_ratios=7.0,
                time_duration=1.0,
                sampling_freq=100.0,
                dtype=dt,
            )
            assert wavelets.dtype == dt

    def test_freq_widths(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0, 20.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=100.0,
            dtype=torch.float64,
        )
        expected = (4.0 * LN2) / (PI * wavelets.time_widths)

        assert wavelets.freq_widths.shape == (2,)
        assert torch.allclose(wavelets.freq_widths, expected)

    def test_omega0s(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0, 20.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=100.0,
            dtype=torch.float64,
        )
        expected = (wavelets.shape_ratios * PI) / math.sqrt(2.0 * LN2)

        assert torch.allclose(wavelets.omega0s, expected)

    def test_scales(self) -> None:
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0, 20.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=100.0,
            dtype=torch.float64,
        )
        expected = (wavelets.omega0s * wavelets.sampling_freq) / (
            2.0 * PI * wavelets.center_freqs
        )

        assert wavelets.scales.shape == (2,)
        assert torch.allclose(wavelets.scales, expected)

    def test_len(self) -> None:
        for n in (1, 3, 5):
            freqs = [5.0 + i * 3.0 for i in range(n)]
            wavelets = _MorletWaveletBase(
                center_freqs=freqs,
                shape_ratios=7.0,
                time_duration=1.0,
                sampling_freq=100.0,
            )
            assert len(wavelets) == n

    def test_compute_freq_resps_default_n_fft(self) -> None:
        """n_fft=None should use next power of 2 >= n_samples."""
        wavelets = _MorletWaveletBase(
            center_freqs=[10.0],
            shape_ratios=7.0,
            time_duration=1.0,
            sampling_freq=100.0,
        )
        freqs, resps = wavelets.compute_freq_resps(n_fft=None, scaled=False)
        expected_n_fft = int(2 ** math.ceil(math.log2(wavelets.n_samples)))
        expected_n_freqs = expected_n_fft // 2 + 1

        assert freqs.shape == (expected_n_freqs,)
        assert resps.shape == (1, expected_n_freqs)

    def test_compute_freq_resps_peaks_near_center_freq(self) -> None:
        """Frequency response should peak near each wavelet's center freq."""
        center_freqs = [8.0, 20.0, 35.0]
        wavelets = _MorletWaveletBase(
            center_freqs=center_freqs,
            shape_ratios=7.0,
            time_duration=2.0,
            sampling_freq=100.0,
        )
        freqs, resps = wavelets.compute_freq_resps(n_fft=1024, scaled=False)
        freq_resolution = freqs[1].item() - freqs[0].item()

        for i, cf in enumerate(center_freqs):
            peak_idx = torch.argmax(resps[i])
            peak_freq = freqs[peak_idx].item()
            assert abs(peak_freq - cf) <= freq_resolution + 1e-6
