"""Tests for the singular Morlet wavelet interface."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from morvex._transform_engine import _MorletTransformEngine
from morvex.wavelet import MorletWavelet


@pytest.fixture()
def wavelet() -> MorletWavelet:
    return MorletWavelet(
        center_freq=10.0, shape_ratio=5.0, time_duration=1.0, sampling_freq=100.0
    )


class TestMorletWaveletComposition:
    def test_uses_composition_instead_of_engine_inheritance(
        self, wavelet: MorletWavelet
    ) -> None:
        assert isinstance(wavelet, nn.Module)
        assert not isinstance(wavelet, _MorletTransformEngine)
        assert isinstance(wavelet._engine, _MorletTransformEngine)

    def test_wrapped_engine_is_a_registered_child_module(
        self, wavelet: MorletWavelet
    ) -> None:
        children = dict(wavelet.named_children())

        assert children == {"_engine": wavelet._engine}
        assert wavelet.to(device="cpu") is wavelet
        assert wavelet.device.type == "cpu"


class TestMorletWaveletProperties:
    def test_exposes_singular_properties(self, wavelet: MorletWavelet) -> None:
        assert wavelet.center_freq == pytest.approx(10.0)
        assert wavelet.shape_ratio == pytest.approx(5.0)
        assert wavelet.time_width == pytest.approx(0.5)
        assert wavelet.waveform.shape == (wavelet.n_samples,)
        assert wavelet.times.shape == (wavelet.n_samples,)
        assert isinstance(wavelet.omega0, float)
        assert isinstance(wavelet.scale, float)


class TestMorletWaveletForward:
    @pytest.mark.parametrize("input_shape", [(200,), (3, 200), (2, 3, 200)])
    def test_removes_wavelet_dimension(
        self, wavelet: MorletWavelet, input_shape: tuple[int, ...]
    ) -> None:
        coeffs = wavelet(torch.randn(input_shape))

        assert coeffs.shape == input_shape

    def test_matches_wrapped_engine(self, wavelet: MorletWavelet) -> None:
        signal = torch.randn(200)

        actual = wavelet(signal)
        expected = wavelet._engine(signal)[0]

        assert torch.allclose(actual, expected)


class TestMorletWaveletFrequencyResponse:
    def test_is_singular(self, wavelet: MorletWavelet) -> None:
        freqs, response = wavelet.compute_freq_resp(n_fft=256)

        assert freqs.ndim == 1
        assert response.shape == freqs.shape
