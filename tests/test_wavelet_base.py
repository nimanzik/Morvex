from __future__ import annotations

import math

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
            ([0.0, 1.0], "range"),
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

    @pytest.mark.parametrize(
        ("shape_ratios", "message"),
        [
            ([1.0, float("inf")], "finite"),
            ([0.0, 1.0], "positive"),
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
