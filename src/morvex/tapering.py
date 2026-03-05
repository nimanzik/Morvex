from __future__ import annotations

from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

WindowType: TypeAlias = Literal["bartlett", "blackman", "hann", "hamming", "kaiser"]

_TAPER_PARAMS_DOC = """\
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
    max_fade_length : int or None, default=None
        Maximum length of the fade region in samples. If None, no fixed-length
        limiting is applied.
    side : {"left", "right", "both"}, default="both"
        Side(s) of the window to apply tapering. Default is "both".
    kaiser_beta : float, default=14.0
        Beta parameter for the Kaiser window. Higher values produce
        narrower transitions with more stopband attenuation. Only relevant if
        `window_type="kaiser"`. Otherwise, this parameter is ignored."""


def get_taper_window(
    window_type: WindowType,
    n_samples: int,
    *,
    max_percentage: float | None = None,
    max_fade_length: int | None = None,
    side: Literal["left", "right", "both"] = "both",
    kaiser_beta: float = 14.0,
) -> torch.Tensor:
    try:
        window_fn = getattr(torch, f"{window_type}_window")
    except AttributeError as e:
        raise ValueError(f"Unsupported window type: {window_type}") from e

    max_half_lens = [n_samples // 2]

    if max_percentage is not None:
        max_half_lens.append(int(n_samples * max_percentage))

    if max_fade_length is not None:
        max_half_lens.append(max_fade_length)

    fade_len = min(max_half_lens)
    if fade_len == 0:
        return torch.ones(n_samples)

    window_kwargs = {"periodic": False}
    if window_type == "kaiser":
        window_kwargs["beta"] = kaiser_beta

    fade_in_out = window_fn(2 * fade_len + 1, **window_kwargs)

    if side == "left":
        return torch.cat([fade_in_out[:fade_len], torch.ones(n_samples - fade_len)])
    elif side == "right":
        return torch.cat([torch.ones(n_samples - fade_len), fade_in_out[-fade_len:]])
    else:  # side == "both"
        return torch.cat(
            [
                fade_in_out[:fade_len],
                torch.ones(n_samples - 2 * fade_len),
                fade_in_out[-fade_len:],
            ]
        )


get_taper_window.__doc__ = f"""Create a tapering window of a specified type and length.

{_TAPER_PARAMS_DOC}

    Returns
    -------
    out : torch.Tensor of shape (n_samples,)
        The taper window, with the maximum value normalized to 1.
"""


class Taper(nn.Module):
    __doc__ = f"""A PyTorch module that applies a tapering window to its input.

{_TAPER_PARAMS_DOC}
    """

    weight: torch.Tensor

    def __init__(
        self,
        window_type: WindowType,
        n_samples: int,
        *,
        max_percentage: float | None = None,
        max_fade_length: int | None = None,
        side: Literal["left", "right", "both"] = "both",
        kaiser_beta: float = 14.0,
    ):
        super().__init__()
        self.register_buffer(
            name="weight",
            tensor=get_taper_window(
                window_type,
                n_samples,
                max_percentage=max_percentage,
                max_fade_length=max_fade_length,
                side=side,
                kaiser_beta=kaiser_beta,
            ),
        )

    def extra_repr(self) -> str:
        return f"weight_shape={tuple(self.weight.shape)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tapering window to the input along the last dimension.

        Parameters
        ----------
        x : torch.Tensor of shape (..., n_samples)
            Input tensor to be tapered. The last dimension must match the
            length of the tapering window.

        Returns
        -------
        out : torch.Tensor of shape (..., n_samples)
            The tapered output, obtained by element-wise multiplication of the
            input with the tapering window.
        """
        return x * self.weight


class TaperConfig(BaseModel):
    window_type: WindowType = Field(
        ...,
        description="Type of window function to use for the fade region.",
    )
    n_samples: int = Field(
        ...,
        gt=0,
        description="Total length of the output window in samples.",
    )
    max_percentage: float | None = Field(
        default=None,
        gt=0,
        le=0.5,
        description=(
            "Maximum length of the fade region expressed as a fraction of "
            "`n_samples`. Must be between 0 (exclusive) and 0.5 (inclusive). "
            "If None, no percentage-based limiting is applied."
        ),
    )
    max_fade_length: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum length of the fade region in samples. If None, no "
            "fixed-length limiting is applied."
        ),
    )
    side: Literal["left", "right", "both"] = Field(
        default="both",
        description="Side(s) of the window to apply tapering. Default is 'both'.",
    )
    kaiser_beta: float = Field(
        default=14.0,
        description=(
            "Beta parameter for the Kaiser window. Higher values produce "
            "narrower transitions with more stopband attenuation. Only relevant "
            "if `window_type='kaiser'`. Otherwise, this parameter is ignored."
        ),
    )

    def create_taper_module(self) -> Taper:
        """Create a Taper module based on the configuration."""
        return Taper(
            window_type=self.window_type,
            n_samples=self.n_samples,
            max_percentage=self.max_percentage,
            max_fade_length=self.max_fade_length,
            side=self.side,
            kaiser_beta=self.kaiser_beta,
        )
