#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Condition-averaged ERPs for the browser page (the Mac main.py ERP plot, ported).

`main.py` shows the averaged ERP of the recording that just finished, through
`plot_erp.ERPPlotWidget` / `render_all_channels_png`. The Pi has no Qt and no
matplotlib in the server process, so the averaging happens here and the browser
draws it on a canvas — same content, no plotting stack on the acquisition host.

Input is one recording written by `bci_server.py`
(`Timestamp, Ch1..Ch8[, EXT1, EXT2], Label, Seq`, unfiltered µV); the file is only read.
サンプリングレートは Timestamp 列から実測する（Cyton 250 Hz / Polymate 1000 Hz）。
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

FS = 250.0                      # レートが読めなかったときの既定
KNOWN_RATES = (250.0, 500.0, 1000.0, 2000.0)
CHANNELS = [f"Ch{i}" for i in range(1, 9)]
PRE_MS = 100.0                  # 刺激前。ベースライン区間でもある

# 極性。Mac の plot_erp.py は既定（negative_peak_on=False）で帯域制限後の振幅を
# 反転させているので、同じ向きに見えるようここでも反転させる。
POLARITY = -1.0

# 表示条件。帯域は make_erp_report.py と同じ 0.5–30 Hz、窓は Mac の plot_erp.py と
# 同じ 1 秒（後続刺激と重なる）。
BAND = (0.5, 30.0)
POST_MS = 1000.0
VIEW_LABEL = "0.5–30 Hz / −100–1000 ms"

# 刺激コード → 表示名と色。色は index.html の凡例と揃える。
STIM_NAME = {1: "Left (524 Hz)", 2: "Center (708 Hz)",
             3: "Right (932 Hz)", 4: "Silence"}
STIM_COLOR = {1: "#3a7afe", 2: "#2faa55", 3: "#d24b4b", 4: "#9aa0aa"}


class ErpError(Exception):
    """計算できない理由をそのままブラウザに出すための例外。"""


def detect_fs(df, default=FS):
    """Timestamp 列から実サンプリングレートを推定し、既知の値に丸める。

    実測をそのまま使うと 999.6 Hz のような端数になって時間軸のラベルが
    ずれるので、アンプが出しうるレートに寄せる。
    """
    if "Timestamp" not in df.columns or len(df) < 2:
        return default
    dt = float(np.median(np.diff(df["Timestamp"].to_numpy(dtype=float))))
    if not np.isfinite(dt) or dt <= 0:
        return default
    return min(KNOWN_RATES, key=lambda k: abs(k - 1.0 / dt))


def target_of(path):
    """ファイル名末尾のターゲットキー（"1"/"2"/"3"）。聴覚BCIの記録以外は None。"""
    key = os.path.basename(os.path.splitext(path)[0]).rsplit("_", 1)[-1]
    return key if key in ("1", "2", "3") else None


def compute(csv_path):
    """1本の記録から刺激ごとの平均ERP（平均±標準誤差）を作る。

    戻り値はそのまま jsonify できる dict。mean / sem は [channel][time] の順で、
    振幅は µV、時間軸は刺激オンセットを 0 とする ms。
    """
    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError) as exc:
        raise ErpError(f"読み込みに失敗: {exc}") from exc

    missing = [c for c in CHANNELS + ["Label"] if c not in df.columns]
    if missing:
        # Ne-utsuri の記録にはラベル列が無いので、ここで弾かれるのが正しい
        raise ErpError("刺激ラベルのある記録ではありません（列 "
                       + ", ".join(missing) + " が見つかりません）")

    fs = detect_fs(df)
    eeg = df[CHANNELS].to_numpy(dtype=float)
    pre = int(round(PRE_MS / 1000.0 * fs))
    post = int(round(POST_MS / 1000.0 * fs))
    if eeg.shape[0] < pre + post + 1:
        raise ErpError("記録が短すぎます")

    sos = butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    filt = sosfiltfilt(sos, eeg, axis=0) * POLARITY

    # ラベルが変わった行が刺激オンセット（merge_labels が ffill で埋めているため）
    labels = np.nan_to_num(df["Label"].to_numpy(dtype=float), nan=0.0)
    onsets = np.where(np.diff(labels) != 0)[0] + 1
    onsets = onsets[labels[onsets] != 0]
    if onsets.size == 0:
        raise ErpError("刺激ラベルが入っていません")

    chunks, tones = [], []
    for on in onsets:
        if on - pre < 0 or on + post > filt.shape[0]:
            continue                        # 窓が記録の外へ出るエポックは捨てる
        seg = filt[on - pre:on + post]
        chunks.append(seg - seg[:pre].mean(axis=0))     # 刺激前平均をベースラインに
        tones.append(int(labels[on]))
    if not chunks:
        raise ErpError("窓に収まるエポックがありません")

    epochs = np.asarray(chunks)             # (n_epoch, n_time, n_ch)
    tones = np.asarray(tones)
    t = (np.arange(-pre, post) / fs) * 1000.0

    target = target_of(csv_path)
    series = []
    for code in sorted(STIM_NAME):
        sel = tones == code
        n = int(sel.sum())
        if n == 0:
            continue
        x = epochs[sel]                     # (n, time, ch)
        mean = x.mean(axis=0).T             # (ch, time)
        # n=1 のときは標準誤差が定義できない。帯を出さず 0 にする。
        sem = (x.std(axis=0, ddof=1).T / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
        series.append({
            "code": code,
            "name": STIM_NAME[code],
            "color": STIM_COLOR[code],
            "is_target": target is not None and code == int(target),
            "n": n,
            "mean": np.round(mean, 2).tolist(),
            "sem": np.round(sem, 2).tolist(),
        })

    return {
        "ok": True,
        "file": os.path.basename(csv_path),
        "target": target,
        "view_label": VIEW_LABEL,
        "band": list(BAND),
        "pre_ms": PRE_MS,
        "post_ms": POST_MS,
        "fs": fs,
        "channels": CHANNELS,
        "t": np.round(t, 2).tolist(),
        "series": series,
        "epochs": int(len(chunks)),
        "dropped": int(onsets.size - len(chunks)),
    }
