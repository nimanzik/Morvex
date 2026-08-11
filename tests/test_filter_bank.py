"""Tests for the MorletFilterBank and related utilities."""

from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from morvex.filter_bank import (
    MorletFilterBank,
    MorletFilterBankConfig,
    _compute_morlet_center_freqs,
)

PI = math.pi
LN2 = math.log(2.0)


@pytest.fixture()
def default_config() -> dict:
    return dict(
        n_octaves=4,
        resolution=8,
        shape_ratio=1.0,
        time_duration=2.0,
        sampling_freq=100.0,
    )


@pytest.fixture()
def filter_bank(default_config: dict) -> MorletFilterBank:
    return MorletFilterBank(**default_config)


class TestComputeMorletCenterFreqs:
    def test_output_is_1d_tensor(self) -> None:
        cfs = _compute_morlet_center_freqs(
            n_octaves=3, resolution=4, shape_ratio=1.0, sampling_freq=100.0
        )
        assert cfs.ndim == 1
        assert cfs.numel() > 0

    def test_total_count_upper_bound(self) -> None:
        """Number of center freqs is at most n_octaves * resolution + 1."""
        n_oct, res = 4, 8
        cfs = _compute_morlet_center_freqs(
            n_octaves=n_oct, resolution=res, shape_ratio=1.0, sampling_freq=100.0
        )
        assert cfs.numel() <= n_oct * res + 1

    def test_frequencies_are_positive(self) -> None:
        cfs = _compute_morlet_center_freqs(
            n_octaves=4, resolution=8, shape_ratio=1.0, sampling_freq=100.0
        )
        assert torch.all(cfs > 0)

    def test_frequencies_below_nyquist(self) -> None:
        """All center freqs + half bandwidth must stay below Nyquist."""
        fs = 100.0
        shape_ratio = 1.0
        cfs = _compute_morlet_center_freqs(
            n_octaves=4, resolution=8, shape_ratio=shape_ratio, sampling_freq=fs
        )
        freq_widths = (4.0 * LN2 * cfs) / (PI * shape_ratio)
        assert torch.all((cfs + 0.5 * freq_widths) < 0.5 * fs)

    def test_frequencies_are_sorted_ascending(self) -> None:
        cfs = _compute_morlet_center_freqs(
            n_octaves=5, resolution=4, shape_ratio=1.0, sampling_freq=200.0
        )
        assert torch.all(cfs[1:] > cfs[:-1])

    def test_higher_sampling_freq_scales_center_freqs(self) -> None:
        cfs_low = _compute_morlet_center_freqs(
            n_octaves=3, resolution=4, shape_ratio=1.0, sampling_freq=100.0
        )
        cfs_high = _compute_morlet_center_freqs(
            n_octaves=3, resolution=4, shape_ratio=1.0, sampling_freq=200.0
        )
        # Center freqs scale linearly with sampling_freq
        assert cfs_high.max() > cfs_low.max()


class TestMorletFilterBankConfig:
    def test_valid_config(self, default_config: dict) -> None:
        cfg = MorletFilterBankConfig(**default_config)
        assert cfg.n_octaves == 4
        assert cfg.resolution == 8

    @pytest.mark.parametrize(
        "field, invalid_value",
        [
            ("n_octaves", 0),
            ("n_octaves", -1),
            ("resolution", 0),
            ("shape_ratio", -1.0),
            ("time_duration", 0.0),
            ("sampling_freq", -10.0),
        ],
    )
    def test_rejects_invalid_values(
        self, default_config: dict, field: str, invalid_value: object
    ) -> None:
        default_config[field] = invalid_value
        with pytest.raises(ValidationError):
            MorletFilterBankConfig(**default_config)


class TestMorletFilterBankInit:
    def test_basic_construction(self, filter_bank: MorletFilterBank) -> None:
        assert filter_bank.n_octaves == 4
        assert filter_bank.resolution == 8
        assert len(filter_bank) > 0

    def test_from_config(self, default_config: dict) -> None:
        cfg = MorletFilterBankConfig(**default_config)
        fb = MorletFilterBank.from_config(cfg)
        assert fb.n_octaves == cfg.n_octaves
        assert fb.resolution == cfg.resolution

    @pytest.mark.parametrize(
        "field, invalid_value",
        [
            ("n_octaves", 0),
            ("resolution", -1),
            ("shape_ratio", 0.0),
            ("time_duration", -1.0),
            ("sampling_freq", 0.0),
        ],
    )
    def test_rejects_invalid_params(
        self, default_config: dict, field: str, invalid_value: object
    ) -> None:
        default_config[field] = invalid_value
        with pytest.raises(ValueError, match="Invalid filter bank configuration"):
            MorletFilterBank(**default_config)


class TestMorletFilterBankProperties:
    def test_shape_ratio_scalar(self, filter_bank: MorletFilterBank) -> None:
        assert isinstance(filter_bank.shape_ratio, float)
        assert filter_bank.shape_ratio == 1.0

    def test_omega0_scalar(self, filter_bank: MorletFilterBank) -> None:
        assert isinstance(filter_bank.omega0, float)
        expected = (1.0 * PI) / math.sqrt(2.0 * LN2)
        assert filter_bank.omega0 == pytest.approx(expected, abs=1e-6)

    def test_waveforms_shape(self, filter_bank: MorletFilterBank) -> None:
        wf = filter_bank.waveforms
        assert wf.ndim == 2
        assert wf.shape[0] == len(filter_bank)
        assert wf.shape[1] == filter_bank.n_samples

    def test_center_freqs_count(self, filter_bank: MorletFilterBank) -> None:
        # Must be <= n_octaves * resolution + 1 (Nyquist mask may remove some)
        assert len(filter_bank) <= filter_bank.n_octaves * filter_bank.resolution + 1
        assert len(filter_bank) > 0


class TestMorletFilterBankForward:
    def test_output_shape_1d(self, filter_bank: MorletFilterBank) -> None:
        signal = torch.randn(1000)
        coeffs = filter_bank(signal)
        assert coeffs.shape == (len(filter_bank), 1000)

    def test_output_shape_2d(self, filter_bank: MorletFilterBank) -> None:
        signal = torch.randn(3, 1000)
        coeffs = filter_bank(signal)
        assert coeffs.shape == (3, len(filter_bank), 1000)

    def test_output_shape_3d(self, filter_bank: MorletFilterBank) -> None:
        signal = torch.randn(2, 3, 1000)
        coeffs = filter_bank(signal)
        assert coeffs.shape == (2, 3, len(filter_bank), 1000)

    def test_power_coeffs_are_non_negative(self, filter_bank: MorletFilterBank) -> None:
        signal = torch.randn(500)
        coeffs = filter_bank(signal, coeff_type="power")
        assert torch.all(coeffs >= 0)

    def test_complex_coeffs_are_complex(self, filter_bank: MorletFilterBank) -> None:
        signal = torch.randn(500)
        coeffs = filter_bank(signal, coeff_type="complex")
        assert coeffs.is_complex()

    def test_magnitude_coeffs_are_non_negative(
        self, filter_bank: MorletFilterBank
    ) -> None:
        signal = torch.randn(500)
        coeffs = filter_bank(signal, coeff_type="magnitude")
        assert torch.all(coeffs >= 0)
