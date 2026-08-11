from typing import Literal, TypeAlias

PlotBackend: TypeAlias = Literal["matplotlib", "plotly"]

WindowType: TypeAlias = Literal["bartlett", "blackman", "hann", "hamming", "kaiser"]
