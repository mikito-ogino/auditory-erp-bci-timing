#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an ERP PDF report from one recording made by bci_server.py.

Usage:
    python3 pi_bci/make_erp_report.py <recording.csv> [-o OUTDIR] [--focus-ch N]

The CSV is the one the Pi writes (`Timestamp, Ch1..Ch8, Label, Seq`), holding
unfiltered µV. Everything the report shows is derived here; the CSV is not modified.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt

# ---- 図の体裁（冒頭で一括管理）----
FONT_SIZE = 10          # 全文字共通サイズ
LABEL_X = -0.14         # サブプロットラベルのx位置（axes座標）
LABEL_Y = 1.10          # サブプロットラベルのy位置（axes座標）
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})

# ---- 解析パラメータ ----
FS = 250.0              # Timestamp からレートが読めなかったときの既定
KNOWN_RATES = (250.0, 500.0, 1000.0, 2000.0)
BAND = (0.5, 30.0)      # model.py の filter_fixed プリセットと同じ帯域
PRE_MS = 100.0          # 刺激前（ベースライン区間でもある）
POST_MS = 500.0         # 刺激後（SOA=500 ms なので次刺激まで）
POLARITY = -1.0         # Mac の plot_erp.py と同じ向き（陰性成分が上）に揃える
CHANNELS = [f"Ch{i}" for i in range(1, 9)]

STIM_NAME = {1: "LEFT (524 Hz)", 2: "CENTER (708 Hz)", 3: "RIGHT (932 Hz)", 4: "SILENCE"}
STIM_COLOR = {1: "#3a7afe", 2: "#2faa55", 3: "#d24b4b", 4: "#9aa0aa"}
PANEL = "abcdefghijklmnop"


def panel_label(ax, i):
    ax.text(LABEL_X, LABEL_Y, f"({PANEL[i]})", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold")


def detect_fs(df, default=FS):
    """Timestamp 列から実サンプリングレートを推定し、既知の値に丸める。

    Cyton は 250 Hz、Polymate Pocket は 1000 Hz。実測をそのまま使うと 999.6 Hz の
    ような端数になって時間軸がずれるので、アンプが出しうるレートに寄せる。
    """
    if "Timestamp" not in df.columns or len(df) < 2:
        return default
    dt = float(np.median(np.diff(df["Timestamp"].to_numpy(dtype=float))))
    if not np.isfinite(dt) or dt <= 0:
        return default
    return min(KNOWN_RATES, key=lambda k: abs(k - 1.0 / dt))


def load(csv_path):
    """CSVを読み、帯域制限したEEGと刺激オンセット（行番号）を返す。"""
    df = pd.read_csv(csv_path)
    eeg = df[CHANNELS].to_numpy(dtype=float)
    fs = detect_fs(df)

    sos = butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    filt = sosfiltfilt(sos, eeg, axis=0) * POLARITY

    labels = np.nan_to_num(df["Label"].to_numpy(dtype=float), nan=0.0)
    onsets = np.where(np.diff(labels) != 0)[0] + 1
    onsets = onsets[labels[onsets] != 0]
    return df, filt, onsets, labels[onsets].astype(int), fs


def epoch(filt, onsets, tones, fs=FS):
    """ベースライン補正したエポック (n_epoch, n_time, n_ch) と時間軸 [ms] を作る。"""
    pre = int(round(PRE_MS / 1000.0 * fs))
    post = int(round(POST_MS / 1000.0 * fs))
    keep, chunks = [], []
    for i, on in enumerate(onsets):
        if on - pre < 0 or on + post > filt.shape[0]:
            continue
        seg = filt[on - pre:on + post]
        chunks.append(seg - seg[:pre].mean(axis=0))     # 刺激前平均をベースラインにする
        keep.append(i)
    epochs = np.asarray(chunks)
    t = (np.arange(-pre, post) / fs) * 1000.0
    return epochs, t, tones[keep]


def decorate(ax, t):
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Amplitude (µV)")


def page_all_channels(pdf, epochs, t, tones):
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    axes = axes.ravel()
    for ch, ax in enumerate(axes):
        for code in sorted(STIM_NAME):
            sel = tones == code
            if not sel.any():
                continue
            ax.plot(t, epochs[sel, :, ch].mean(axis=0),
                    color=STIM_COLOR[code], lw=1.4, label=STIM_NAME[code])
        decorate(ax, t)
        ax.set_title(CHANNELS[ch])
        panel_label(ax, ch)
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    pdf.savefig(fig)
    plt.close(fig)


def page_focus(pdf, epochs, t, tones, ch_idx):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    # (a) 刺激ごとの平均 ± 標準誤差
    ax = axes[0]
    for code in sorted(STIM_NAME):
        sel = tones == code
        if not sel.any():
            continue
        x = epochs[sel, :, ch_idx]
        m, se = x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(sel.sum())
        ax.plot(t, m, color=STIM_COLOR[code], lw=1.6, label=f"{STIM_NAME[code]}  n={sel.sum()}")
        ax.fill_between(t, m - se, m + se, color=STIM_COLOR[code], alpha=0.18, lw=0)
    decorate(ax, t)
    ax.set_title(f"{CHANNELS[ch_idx]} — mean ± SEM")
    panel_label(ax, 0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    # (b) LEFT の単一試行を並べたヒートマップ
    ax = axes[1]
    sel = tones == 1
    single = epochs[sel, :, ch_idx]
    vmax = np.abs(single).max() if single.size else 1.0
    im = ax.imshow(single, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=(t[0], t[-1], single.shape[0] - 0.5, -0.5),
                   interpolation="nearest")
    ax.axvline(0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("LEFT trial #")
    ax.set_title(f"{CHANNELS[ch_idx]} — single LEFT trials")
    panel_label(ax, 1)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Amplitude (µV)")
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    # (c) LEFT と他刺激平均の差分
    ax = axes[2]
    left = epochs[tones == 1, :, ch_idx].mean(axis=0)
    other = epochs[tones != 1, :, ch_idx].mean(axis=0)
    ax.plot(t, left - other, color="#3a7afe", lw=1.6, label="LEFT − mean(others)")
    decorate(ax, t)
    ax.set_title(f"{CHANNELS[ch_idx]} — difference wave")
    panel_label(ax, 2)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    fig.tight_layout(rect=(0, 0, 0.86, 1))
    pdf.savefig(fig)
    plt.close(fig)


def page_peaks(pdf, epochs, t, tones):
    """刺激後 0–400 ms のピーク振幅をチャンネル×刺激で俯瞰する。"""
    win = (t >= 0) & (t <= 400)
    codes = sorted(STIM_NAME)
    grid = np.zeros((len(CHANNELS), len(codes)))
    for ci in range(len(CHANNELS)):
        for k, code in enumerate(codes):
            sel = tones == code
            m = epochs[sel, :, ci].mean(axis=0)[win]
            grid[ci, k] = m[np.argmax(np.abs(m))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    vmax = np.abs(grid).max()
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([STIM_NAME[c].split(" ")[0] for c in codes])
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("Stimulus")
    ax.set_ylabel("Channel")
    ax.set_title("Peak amplitude, 0–400 ms")
    panel_label(ax, 0)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=FONT_SIZE - 1)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Amplitude (µV)")
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    ax = axes[1]
    width = 0.2
    x = np.arange(len(CHANNELS))
    for k, code in enumerate(codes):
        ax.bar(x + (k - 1.5) * width, np.abs(grid[:, k]), width,
               color=STIM_COLOR[code], label=STIM_NAME[code])
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_xlabel("Channel")
    ax.set_ylabel("|Peak amplitude| (µV)")
    ax.set_title("Peak magnitude by channel")
    panel_label(ax, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    fig.tight_layout(rect=(0, 0, 0.82, 1))
    pdf.savefig(fig)
    plt.close(fig)


def page_info(pdf, csv_path, df, tones, n_epochs, timing, fs=FS):
    lines = [
        "Recording",
        f"    file                 {os.path.basename(csv_path)}",
        f"    samples              {len(df)}",
        f"    channels             {len(CHANNELS)}",
        f"    stimuli labelled     {len(tones)}",
        f"    epochs used          {n_epochs}",
        "    per stimulus         " + ",  ".join(
            f"{STIM_NAME[c].split(' ')[0]}={int((tones == c).sum())}" for c in sorted(STIM_NAME)),
        "",
        "Processing (applied in this report only — the CSV is untouched)",
        f"    band-pass            {BAND[0]}–{BAND[1]} Hz, 4th-order Butterworth, zero-phase",
        f"    epoch                {-PRE_MS:.0f} to +{POST_MS:.0f} ms around onset",
        f"    baseline             {-PRE_MS:.0f} to 0 ms, per epoch per channel",
        f"    sampling rate        {fs:.0f} Hz (Timestamp 列から実測)",
    ]
    if timing:
        def fmt(key):
            s = timing.get(key)
            return (f"mean {s['mean']:.3f}  sd {s['sd']:.3f}  "
                    f"min {s['min']:.3f}  max {s['max']:.3f}") if s else "n/a"
        lines += [
            "",
            "Measured timing (from the _timing.json sidecar, in ms)",
            f"    effective rate       {timing.get('effective_fs')} Hz",
            f"    sample interval      {fmt('sample_interval_ms')}",
            f"    audio SOA            {fmt('audio_soa_ms')}",
            f"    label SOA            {fmt('label_soa_ms')}",
            f"    dropped samples      {timing.get('dropped_samples')}",
        ]

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.text(0.06, 0.94, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=FONT_SIZE)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--outdir", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--focus-ch", type=int, default=1, help="detail page channel (1-8)")
    args = ap.parse_args()

    df, filt, onsets, tones, fs = load(args.csv)
    epochs, t, tones = epoch(filt, onsets, tones, fs)

    timing = None
    side = os.path.splitext(args.csv)[0] + "_timing.json"
    if os.path.exists(side):
        with open(side) as f:
            timing = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    pdf_path = os.path.join(args.outdir, f"{stem}_ERP_report.pdf")

    with PdfPages(pdf_path) as pdf:
        page_all_channels(pdf, epochs, t, tones)
        page_focus(pdf, epochs, t, tones, args.focus_ch - 1)
        page_peaks(pdf, epochs, t, tones)
        page_info(pdf, args.csv, df, tones, len(epochs), timing, fs)

    print(f"epochs: {len(epochs)}  " +
          "  ".join(f"{STIM_NAME[c].split(' ')[0]}={int((tones == c).sum())}"
                    for c in sorted(STIM_NAME)))
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
