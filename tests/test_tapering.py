"""Unit tests for morvex.tapering module."""

from __future__ import annotations

from typing import get_args

import pytest
import torch
from pydantic import ValidationError

from morvex.tapering import Taper, TaperConfig, WindowType, build_taper_window


@pytest.fixture
def n_samples() -> int:
    return 100


@pytest.fixture
def default_window(n_samples: int) -> torch.Tensor:
    return build_taper_window("hann", n_samples)


class TestTaperConfig:
    def test_valid_config(self) -> None:
        cfg = TaperConfig(window_type="hann", n_samples=100)
        assert cfg.window_type == "hann"
        assert cfg.n_samples == 100
        assert cfg.side == "both"

    def test_invalid_window_type(self) -> None:
        with pytest.raises(ValidationError):
            TaperConfig(window_type="invalid", n_samples=100)  # ty: ignore[invalid-argument-type]

    def test_n_samples_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TaperConfig(window_type="hann", n_samples=0)

    def test_max_percentage_bounds(self) -> None:
        TaperConfig(window_type="hann", n_samples=100, max_percentage=0.5)
        with pytest.raises(ValidationError):
            TaperConfig(window_type="hann", n_samples=100, max_percentage=0.0)
        with pytest.raises(ValidationError):
            TaperConfig(window_type="hann", n_samples=100, max_percentage=0.6)

    def test_frozen(self) -> None:
        cfg = TaperConfig(window_type="hann", n_samples=100)
        with pytest.raises(ValidationError):
            cfg.n_samples = 200  # ty: ignore[invalid-assignment]


class TestBuildTaperWindow:
    def test_output_shape(self, n_samples: int) -> None:
        w = build_taper_window("hann", n_samples)
        assert w.shape == (n_samples,)

    def test_center_is_ones_both(self, n_samples: int) -> None:
        w = build_taper_window("hann", n_samples, max_fade_len=10)
        # Center region should be all ones
        assert torch.allclose(w[10:90], torch.ones(80))

    def test_edges_start_at_zero_both(self, n_samples: int) -> None:
        w = build_taper_window("hann", n_samples, max_fade_len=10)
        assert w[0].item() == pytest.approx(0.0, abs=1e-6)
        assert w[-1].item() == pytest.approx(0.0, abs=1e-6)

    def test_side_left(self, n_samples: int) -> None:
        w = build_taper_window("hann", n_samples, max_fade_len=10, side="left")
        assert w[0].item() == pytest.approx(0.0, abs=1e-6)
        # Right side should be all ones
        assert torch.allclose(w[10:], torch.ones(90))

    def test_side_right(self, n_samples: int) -> None:
        w = build_taper_window("hann", n_samples, max_fade_len=10, side="right")
        assert w[-1].item() == pytest.approx(0.0, abs=1e-6)
        # Left side should be all ones
        assert torch.allclose(w[:90], torch.ones(90))

    def test_max_percentage_limits_fade(self) -> None:
        # 10% of 100 = 10 samples fade
        w = build_taper_window("hann", 100, max_percentage=0.1)
        assert torch.allclose(w[10:90], torch.ones(80))

    def test_max_fade_len_limits_fade(self) -> None:
        w = build_taper_window("hann", 100, max_fade_len=5)
        assert torch.allclose(w[5:95], torch.ones(90))

    def test_both_constraints_uses_minimum(self) -> None:
        # max_percentage=0.1 -> 10, max_fade_len=5 -> min is 5
        w = build_taper_window("hann", 100, max_percentage=0.1, max_fade_len=5)
        assert torch.allclose(w[5:95], torch.ones(90))

    @pytest.mark.parametrize("window_type", get_args(WindowType))
    def test_all_window_types(self, window_type: WindowType, n_samples: int) -> None:
        w = build_taper_window(window_type, n_samples, max_fade_len=10)
        assert w.shape == (n_samples,)
        # All values should be in [0, 1] (with float tolerance)
        assert w.min() >= -1e-6
        assert w.max() <= 1.0 + 1e-6

    def test_kaiser_beta_affects_shape(self, n_samples: int) -> None:
        w1 = build_taper_window("kaiser", n_samples, kaiser_beta=2.0)
        w2 = build_taper_window("kaiser", n_samples, kaiser_beta=14.0)
        assert not torch.allclose(w1, w2)

    def test_invalid_params_raise_valueerror(self) -> None:
        with pytest.raises(ValueError):
            build_taper_window("hann", 0)
        with pytest.raises(ValueError):
            build_taper_window("hann", 100, max_percentage=0.8)

    def test_values_in_unit_range(self, default_window: torch.Tensor) -> None:
        assert default_window.min() >= 0.0
        assert default_window.max() <= 1.0 + 1e-6


class TestTaper:
    def test_forward_shape(self, n_samples: int) -> None:
        taper = Taper("hann", n_samples, max_fade_len=10)
        x = torch.ones(1, n_samples)
        out = taper(x)
        assert out.shape == x.shape

    def test_forward_applies_window(self, n_samples: int) -> None:
        taper = Taper("hann", n_samples, max_fade_len=10)
        x = torch.ones(n_samples)
        out = taper(x)
        assert torch.allclose(out, taper.weight)

    def test_forward_dimension_mismatch_raises(self) -> None:
        taper = Taper("hann", 100)
        x = torch.ones(50)
        with pytest.raises(ValueError, match="does not match"):
            taper(x)

    def test_forward_batched(self, n_samples: int) -> None:
        taper = Taper("hann", n_samples, max_fade_len=10)
        x = torch.ones(4, 2, n_samples)
        out = taper(x)
        assert out.shape == (4, 2, n_samples)

    def test_from_config(self) -> None:
        cfg = TaperConfig(
            window_type="hamming", n_samples=64, max_fade_len=8, side="left"
        )
        taper = Taper.from_config(cfg)
        assert taper.n_samples == 64
        assert taper.side == "left"
        assert taper.weight.shape == (64,)

    def test_weight_is_buffer(self, n_samples: int) -> None:
        taper = Taper("hann", n_samples)
        # weight should be a registered buffer, not a parameter
        assert "weight" in dict(taper.named_buffers())
        assert "weight" not in dict(taper.named_parameters())

    def test_extra_repr(self, n_samples: int) -> None:
        taper = Taper("hann", n_samples)
        assert f"({n_samples},)" in taper.extra_repr()
