import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from morvex import MorletFilterBank
    from morvex.plotting import plot_freq_resps, plot_time_freq_plane

    plt.style.use("bmh")
    return (
        MorletFilterBank,
        Path,
        dataclass,
        np,
        plot_freq_resps,
        plot_time_freq_plane,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Load example data: fin-whale song recording signal

    First, we load an example signal data, which is a recording of a fin-whale song. The data is stored in a NumPy `.npz` file, and contains the time points, acoustic amplitude values, and the sampling frequency of the recording.

    We define a `Signal` dataclass to encapsulate the information above and provide instance properties for signal time-duration, `duration`, and length, `num_samples`.
    """)
    return


@app.cell
def _(dataclass, np):
    @dataclass
    class Signal:
        times: np.ndarray
        values: np.ndarray
        fs: float

        @property
        def duration(self):
            return self.times.max() - self.times.min()

        @property
        def num_samples(self):
            return len(self.values)

        def __repr__(self):
            return (
                f"Signal duration: {self.duration} s | "
                f"num. of samples: {self.num_samples} | "
                f"sampling frequency: {self.fs}"
            )

    return (Signal,)


@app.cell
def _(mo):
    mo.md(r"""
    Now, let's load our example signal and display it. We can also print out the signal information to confirm that it has been loaded correctly.
    """)
    return


@app.cell
def _(Path, Signal, np):
    npz_file = Path(__file__).parents[1] / "data" / "fin_whale_song.npz"
    npz_data = np.load(npz_file)

    signal = Signal(
        times=npz_data["times"],
        values=npz_data["values"],
        fs=npz_data["fs"],
    )

    print(signal)
    return (signal,)


@app.cell
def _(plt, signal):
    fig_sig, ax_sig = plt.subplots(figsize=(10, 6))
    ax_sig.plot(signal.times, signal.values, linewidth=1)
    ax_sig.set(
        xlabel="Time [s]",
        ylabel="Acoustic Amplitude",
        title="Fin-Whale song recording | Bandpass filtered 12–30 Hz",
    )
    # mo.mpl.interactive(fig_sig)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Apply the CWT

    ### Compute scalogram

    Now, we are ready to apply the CWT to our signal using the Morlet filter bank. We specify that we want to compute the magnitude of the coefficients, which will give us a scalogram that represents the time-frequency representation of the signal.
    """)
    return


@app.cell
def _(MorletFilterBank, signal):
    filt_bank = MorletFilterBank(
        n_octaves=2,
        resolution=8,
        shape_ratio=5.0,
        time_duration=1.5,
        sampling_freq=signal.fs,
    )

    coeff_type = "magnitude"
    scalogram = filt_bank(signal.values, coeff_type=coeff_type).detach().cpu().numpy()

    print(filt_bank)
    print(f"Scalogram array shape: {scalogram.shape}")
    return coeff_type, filt_bank, scalogram


@app.cell
def _(mo):
    mo.md(r"""
    ### Display the scalogram
    """)
    return


@app.cell
def _(coeff_type, filt_bank, plot_time_freq_plane, plt, scalogram, signal):
    center_freqs = filt_bank.center_freqs.detach().cpu().numpy()

    fig_sgram, ax_sgram = plt.subplots(figsize=(10, 6))
    plot_time_freq_plane(
        ax=ax_sgram,
        freqs=center_freqs,
        times=signal.times,
        xgram=scalogram,
        label=coeff_type,
    )

    ax_sgram.grid(False)
    # mo.mpl.interactive(fig_sgram)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Display the frequency responses of the filterbank

    We can also visualize the filter responses in the frequency domain by plotting the impulse responses of the filters. This can help us understand how each filter responds to a brief input signal and how it captures different frequency components of the signal.

    In the figure below, we also show the minimum and maximum frequencies of the filterbank, which correspond to the lowest and highest frequencies that the filters can capture. The maximum frequency is determined by the Nyquist frequency,  while the minimum frequency is determined by the number of octaves in the filterbank.
    """)
    return


@app.cell
def _(filt_bank, plot_freq_resps, plt, signal):
    fig_resps, ax_resps = plt.subplots(figsize=(10, 6))
    plot_freq_resps(filt_bank, ax_resps, n_fft=512, color="#3465a4")

    # --- Mark lower frequency passband and the Nyquist ---
    f_nyq = signal.fs / 2
    f_low = f_nyq / (2 ** filt_bank.n_octaves)

    for f in (f_low, f_nyq):
        ax_resps.axvline(f, color="#cc0000", linestyle="--", linewidth=1.5)
        ax_resps.text(
            f - 0.5,
            0.975,
            f"{f:.1f} Hz",
            color="#cc0000",
            fontsize=10,
            ha="right",
            va="top",
            transform=ax_resps.axes.get_xaxis_transform(),
            bbox=dict(boxstyle="round", fc="white", ec="#cc0000", alpha=0.75),
        )

    # mo.mpl.interactive(fig_resps)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
