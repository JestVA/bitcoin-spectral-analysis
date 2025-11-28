#!/usr/bin/env python3
"""
Spectral analysis for BTC-USD (daily) using yfinance.

Dependencies:
    pip install yfinance numpy pandas scipy matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import welch


# ------------------ CONFIGURABLE PARAMETERS ------------------ #

TICKER = "BTC-USD"
PERIOD = "730d"  # max available for 1h data in yfinance (730 days = 2 years)
INTERVAL = "1h"  # working with hourly data
FS = 24.0  # sampling frequency: 24 samples / day (1 sample per hour)

# window settings for spectrogram (in hours)
WINDOW_HOURS = 128 * 24  # window length in hours (equivalent to 128 days)
STEP_HOURS = 16 * 24  # window displacement in hours (equivalent to 16 days)
N_PER_SEG = 256  # for Welch inside the window


# ------------------ UTILITY FUNCTIONS ------------------ #


def download_prices(ticker: str, period: str, interval: str) -> pd.Series:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if "Close" not in data.columns:
        raise RuntimeError("Cannot find Close column in downloaded data.")
    close_data = data["Close"].dropna()
    # Ensure it's a Series, not a DataFrame
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.squeeze()
    return close_data


def compute_log_returns(prices: pd.Series) -> pd.Series:
    logp = np.log(prices)
    r = logp.diff().dropna()
    return r


def global_psd(log_returns: pd.Series):
    # protection: we want at least a few dozen points
    if len(log_returns) < 32:
        raise RuntimeError(
            f"Too few log-returns ({len(log_returns)}) for PSD. "
            "Check period/interval or if you accidentally truncated the series."
        )

    r = log_returns.values
    # Ensure we have a 1D array
    if r.ndim > 1:
        r = r.flatten()

    print(f"[DEBUG] len(r) = {len(r)}, shape = {r.shape}")

    # No need to manually subtract mean - welch does this with detrend="constant"
    freqs, psd = welch(
        r,
        fs=FS,
        nperseg=min(256, len(r)),
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def sliding_spectrogram(log_returns: pd.Series):
    r = log_returns.values
    # Ensure we have a 1D array
    if r.ndim > 1:
        r = r.flatten()
    n = len(r)

    if n < WINDOW_HOURS:
        raise RuntimeError("Too few data points for the selected window size.")

    starts = np.arange(0, n - WINDOW_HOURS + 1, STEP_HOURS)
    spec_list = []
    time_axis = []

    for s in starts:
        e = s + WINDOW_HOURS
        segment = r[s:e]
        # No need to manually subtract mean - welch does this with detrend="constant"
        freqs, psd = welch(
            segment,
            fs=FS,
            nperseg=min(N_PER_SEG, len(segment)),
            detrend="constant",
            scaling="density",
        )
        spec_list.append(psd)
        time_axis.append(log_returns.index[s + WINDOW_HOURS // 2])

    spec = np.vstack(spec_list).T  # shape: [freq, time_window]
    return freqs, np.array(time_axis), spec


def cutoff_frequency(freqs, psd, power_ratio=0.95):
    if len(freqs) < 2:
        raise RuntimeError(
            f"Too few frequencies in PSD (len(freqs)={len(freqs)}). "
            "Likely the input vector to welch had length 1."
        )

    df = freqs[1] - freqs[0]
    cumulative = np.cumsum(psd) * df
    total = cumulative[-1]
    target = power_ratio * total
    idx = np.searchsorted(cumulative, target)
    if idx >= len(freqs):
        idx = len(freqs) - 1
    return freqs[idx], total


# ------------------ MAIN ------------------ #


def main():
    print(f"Downloading {PERIOD} of hourly data for {TICKER} from yfinance...")
    prices = download_prices(TICKER, PERIOD, INTERVAL)
    print(f"Number of data points: {len(prices)}")

    # Calculate time span
    time_span_hours = len(prices)
    time_span_days = time_span_hours / 24
    print(f"Time span: {time_span_days:.1f} days ({time_span_hours} hours)")

    log_returns = compute_log_returns(prices)
    print(f"Number of log-returns: {len(log_returns)}")

    # ---- PSD global ----
    freqs, psd = global_psd(log_returns)
    f_cut, total_power = cutoff_frequency(freqs, psd, power_ratio=0.95)

    dt_opt_hours = 1.0 / (2.0 * f_cut) if f_cut > 0 else np.inf

    print("\n=== GLOBAL PSD SUMMARY ===")
    print(f"Total power (arbitrary units): {total_power:.3e}")
    print(f"f_cutoff (95% of power): {f_cut:.4f} cycles/day")
    print(f"Optimal Nyquist step ~ 1/(2*f_cutoff): {dt_opt_hours:.2f} hours")
    if dt_opt_hours < 1:
        print(f"That is, a sampling faster than ~ {dt_opt_hours * 60:.1f} minutes")

    # Global PSD plot (log-log is usually clearer)
    plt.figure(figsize=(8, 4))
    plt.loglog(freqs[1:], psd[1:])  # skip 0 Hz for log-scale
    plt.axvline(f_cut, color="r", linestyle="--", label=f"f_cutoff={f_cut:.4f}")
    plt.title(f"{TICKER} - PSD log-return (hourly, {PERIOD})")
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Power spectral density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("btc_psd_global.png", dpi=150)
    print("Saved global PSD plot to btc_psd_global.png")

    # ---- Sliding-window spectrogram ----
    print("\nComputing spectrogram (sliding windows)...")
    freqs_s, time_axis, spec = sliding_spectrogram(log_returns)

    plt.figure(figsize=(10, 4))
    # low frequencies are more interesting; we can cut at, e.g., 5 cycles/day
    max_f = 5.0  # increased for hourly data
    mask = freqs_s <= max_f

    im = plt.imshow(
        10 * np.log10(spec[mask, :] + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[
            0,
            len(time_axis),
            freqs_s[mask][0],
            freqs_s[mask][-1],
        ],
        cmap="viridis",
    )
    plt.colorbar(im, label="PSD (dB, arbitrary units)")
    window_days = WINDOW_HOURS / 24
    plt.title(f"{TICKER} - Log-return spectrogram ({window_days:.0f}-day windows)")
    plt.xlabel("Window (time index)")
    plt.ylabel("Frequency (cycles/day)")
    plt.tight_layout()
    plt.savefig("btc_spectrogram.png", dpi=150)
    print("Saved spectrogram to btc_spectrogram.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
