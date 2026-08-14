from typing import Literal

type PlotBackend = Literal["matplotlib", "plotly"]

type TransformOutput = Literal["complex", "magnitude", "power"]

type WindowType = Literal["bartlett", "blackman", "hann", "hamming", "kaiser"]
