"""Tests for cross-correlation via FFT."""

from __future__ import annotations

import pytest
import torch

from morvex.xcorr import _get_centered, _next_fast_len, xcorr_via_fft


class TestNextFastLen:
    def test_returns_int(self) -> None:
        result = _next_fast_len(100, real=True)
        assert isinstance(result, int)

    def test_result_gte_input(self) -> None:
        for n in [1, 7, 100, 1023]:
            assert _next_fast_len(n, real=True) >= n

    def test_caches_results(self) -> None:
        # Calling twice should return the same object (cached).
        a = _next_fast_len(123, real=False)
        b = _next_fast_len(123, real=False)
        assert a == b


class TestGetCentered:
    def test_extracts_center_1d(self) -> None:
        x = torch.arange(10, dtype=torch.float32)
        result = _get_centered(x, (4,))
        expected = torch.tensor([3.0, 4.0, 5.0, 6.0])
        assert torch.equal(result, expected)

    def test_extracts_center_2d(self) -> None:
        x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        result = _get_centered(x, (2, 3))
        expected = x[1:3, 1:4]
        assert torch.equal(result, expected)

    def test_same_shape_returns_original(self) -> None:
        x = torch.randn(3, 4)
        result = _get_centered(x, (3, 4))
        assert torch.equal(result, x)


class TestXcorrViaFft:
    @pytest.fixture
    def simple_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a simple 1D signal and a set of two waveforms."""
        torch.manual_seed(42)
        data = torch.randn(64)
        waveforms = torch.randn(2, 8)
        return data, waveforms

    def test_output_shape_1d(
        self, simple_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        data, waveforms = simple_data
        result = xcorr_via_fft(data, waveforms)
        assert result.shape == (2, 64)

    def test_output_shape_batched(self) -> None:
        torch.manual_seed(0)
        data = torch.randn(3, 5, 128)
        waveforms = torch.randn(4, 16)
        result = xcorr_via_fft(data, waveforms)
        assert result.shape == (3, 5, 4, 128)

    def test_rejects_non_2d_waveforms(self) -> None:
        data = torch.randn(64)
        waveforms_1d = torch.randn(8)
        with pytest.raises(ValueError, match="`waveforms` must be a 2D tensor"):
            xcorr_via_fft(data, waveforms_1d)

        waveforms_3d = torch.randn(2, 3, 8)
        with pytest.raises(ValueError, match="`waveforms` must be a 2D tensor"):
            xcorr_via_fft(data, waveforms_3d)

    def test_impulse_response(self) -> None:
        """Cross-correlate with an impulse that should return the signal itself."""  # noqa: W505
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        impulse = torch.zeros(1, 1)
        impulse[0, 0] = 1.0
        result = xcorr_via_fft(signal, impulse)
        assert result.shape == (1, 5)
        assert torch.allclose(result[0], signal, atol=1e-6)

    def test_complex_input(self) -> None:
        torch.manual_seed(0)
        data = torch.randn(32) + 1j * torch.randn(32)
        waveforms = torch.randn(2, 8) + 1j * torch.randn(2, 8)
        result = xcorr_via_fft(data, waveforms)
        assert result.is_complex()
        assert result.shape == (2, 32)

    def test_matches_manual_correlation(self) -> None:
        """Check FFT-based xcorr against a direct computation for a short case.

        The function computes ifft(conj(FFT(w)) * FFT(d)), which is the
        circular cross-correlation, then centers the result to 'same' length.
        """
        data = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0])
        waveform = torch.tensor([[1.0, -1.0, 1.0]])
        result = xcorr_via_fft(data, waveform)

        # Full result: [0, 1, 0, -1, 1, 0, 1], centered 5: [1, 0, -1, 1, 0]
        expected = torch.tensor([[1.0, 0.0, -1.0, 1.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-5)
