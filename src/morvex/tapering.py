"""Tapering functions for smoothing signals at the edges."""

from __future__ import annotations

from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ValidationError

WindowType: TypeAlias = Literal["bartlett", "blackman", "hann", "hamming", "kaiser"]


class TaperConfig(BaseModel, frozen=True, extra="forbid"):
    window_type: WindowType
    n_samples: int = Field(..., gt=0)
    max_percentage: float | None = Field(default=None, gt=0, le=0.5)
    max_fade_len: int | None = Field(default=None, gt=0)
    side: Literal["left", "right", "both"] = Field(default="both")
    kaiser_beta: float = Field(default=14.0)


def build_taper_window(
    window_type: WindowType,
    n_samples: int,
    *,
    max_percentage: float | None = None,
    max_fade_len: int | None = None,
    side: Literal["left", "right", "both"] = "both",
    kaiser_beta: float = 14.0,
) -> torch.Tensor:
    """Generate a tapering window based on the specified parameters.

    Parameters
    ----------
    window_type : {"bartlett", "blackman", "hann", "hamming", "kaiser"}
        Type of window function to use for the fade region.
    n_samples : int
        Total length of the output window in samples.
    max_percentage : float or None, default=None
        Maximum length of the fade region expressed as a fraction of
        `n_samples`. Must be between 0 (exclusive) and 0.5 (inclusive). If
        None, no percentage-based limiting is applied.
    max_fade_len : int or None, default=None
        Maximum length of the fade region in samples. If None, no fixed-length
        limiting is applied.
    side : {"left", "right", "both"}, default="both"
        Side(s) of the window to apply tapering. Default is "both".
    kaiser_beta : float, default=14.0
        Beta parameter for the Kaiser window. Higher values produce narrower
        transitions with more stopband attenuation. Only relevant if
        `window_type="kaiser"`. Otherwise, this parameter is ignored.

    Returns
    -------
    window : Tensor of shape (n_samples,)
        The tapering window.

    Raises
    ------
    ValueError
        If any parameter fails validation.
    """
    try:
        cfg = TaperConfig(
            window_type=window_type,
            n_samples=n_samples,
            max_percentage=max_percentage,
            max_fade_len=max_fade_len,
            side=side,
            kaiser_beta=kaiser_beta,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid tapering configuration: {e}") from e

    try:
        window_fn = getattr(torch, f"{cfg.window_type}_window")
    except AttributeError as e:
        raise ValueError(f"Unsupported window type: {cfg.window_type}") from e

    max_half_lens = [cfg.n_samples // 2]

    if cfg.max_percentage is not None:
        max_half_lens.append(int(cfg.n_samples * cfg.max_percentage))

    if cfg.max_fade_len is not None:
        max_half_lens.append(cfg.max_fade_len)

    fade_len = min(max_half_lens)
    if fade_len == 0:
        return torch.ones(cfg.n_samples)

    window_kwargs: dict[str, float | bool] = {"periodic": False}
    if cfg.window_type == "kaiser":
        window_kwargs["beta"] = cfg.kaiser_beta

    fade_in_out = window_fn(2 * fade_len + 1, **window_kwargs)

    if cfg.side == "left":
        return torch.cat([fade_in_out[:fade_len], torch.ones(cfg.n_samples - fade_len)])
    elif cfg.side == "right":
        return torch.cat(
            [torch.ones(cfg.n_samples - fade_len), fade_in_out[-fade_len:]]
        )
    else:  # cfg.side == "both"
        return torch.cat(
            [
                fade_in_out[:fade_len],
                torch.ones(cfg.n_samples - 2 * fade_len),
                fade_in_out[-fade_len:],
            ]
        )


class Taper(nn.Module):
    """A PyTorch module that applies a tapering window to its input.

    Parameters
    ----------
    window_type : {"bartlett", "blackman", "hann", "hamming", "kaiser"}
        Type of window function to use for the fade region.
    n_samples : int
        Total length of the output window in samples.
    max_percentage : float or None, default=None
        Maximum length of the fade region expressed as a fraction of
        `n_samples`. Must be between 0 (exclusive) and 0.5 (inclusive). If
        None, no percentage-based limiting is applied.
    max_fade_len : int or None, default=None
        Maximum length of the fade region in samples. If None, no fixed-length
        limiting is applied.
    side : {"left", "right", "both"}, default="both"
        Side(s) of the window to apply tapering. Default is "both".
    kaiser_beta : float, default=14.0
        Beta parameter for the Kaiser window. Higher values produce
        narrower transitions with more stopband attenuation. Only relevant if
        `window_type="kaiser"`. Otherwise, this parameter is ignored.

    Notes
    -----
    - If both `max_percentage` and `max_fade_len` are provided, the actual
      fade length will be the minimum of the two constraints.
    """

    weight: torch.Tensor

    def __init__(
        self,
        window_type: WindowType,
        n_samples: int,
        *,
        max_percentage: float | None = None,
        max_fade_len: int | None = None,
        side: Literal["left", "right", "both"] = "both",
        kaiser_beta: float = 14.0,
    ):
        """Initialise the Taper module."""
        super().__init__()
        self.window_type = window_type
        self.n_samples = n_samples
        self.max_percentage = max_percentage
        self.max_fade_len = max_fade_len
        self.side = side
        self.kaiser_beta = kaiser_beta
        self.register_buffer(
            name="weight",
            tensor=build_taper_window(
                window_type=self.window_type,
                n_samples=self.n_samples,
                max_percentage=self.max_percentage,
                max_fade_len=self.max_fade_len,
                side=self.side,
                kaiser_beta=self.kaiser_beta,
            ),
        )

    @classmethod
    def from_config(cls, config: TaperConfig) -> Taper:
        return cls(**config.model_dump())

    def extra_repr(self) -> str:
        return f"weight_shape={tuple(self.weight.shape)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tapering window to the input along the last dimension.

        Parameters
        ----------
        x : Tensor of shape (..., n_samples)
            Input tensor to be tapered. The last dimension must match the
            length of the tapering window.

        Returns
        -------
        out : Tensor of shape (..., n_samples)
            The tapered output, obtained by element-wise multiplication of the
            input with the tapering window.
        """
        if (last_dim := x.shape[-1]) != self.n_samples:
            raise ValueError(
                f"The last dimension of the input tensor does not match tapering "
                f"window length: expected {self.n_samples}, but got {last_dim}."
            )
        return x * self.weight
