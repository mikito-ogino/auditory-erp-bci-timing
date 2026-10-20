#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ne-utsuri 押下（噛み込み）検出の特徴量 — 学習側 (neutsuri_train.py / neutsuri_search.py) と
Pi の実行側 (neutsuri.py) が同じ関数を使う。ここを変えたら必ず学習し直す。

噛み込みは押下の ±0.2 s に 20–45 Hz と 60–125 Hz の広帯域バースト（咬筋の筋電）
として現れる。従来の「2 秒窓にどれか押下を含むか」というラベルではこの短い
バーストが窓の中で薄まって識別できなかったので、ここでは

  * 短い窓（既定 0.4 s）の帯域ごとの対数パワー（8 ch × 5 帯域 = 40 次元）を
  * 直前 60 秒の中央値 / MAD で因果的に正規化し（セッション・電極のばらつき吸収）
  * 線形分類器に入れる

という構成にする。50 Hz / 100 Hz の電源ノイズは帯域から外してある。

v3 で足したもの（いずれも既定は v2 と同じ挙動になるようにしてある）:

  * 帯域セットを名前で選べる（BAND_SETS）。
  * 特徴量をコンポーネントの組み合わせで指定する（FEATURE_SETS）。
    bp … 帯域対数パワー / bpcmr … 帯域ごとにチャンネル平均を引いたもの（共通成分除去）
    hjorth … Hjorth 3 種 / ll … 対数 line length / rms … 対数 RMS
  * 正規化の統計量を選べる（make_normalizer: none / med_mad / med_iqr / mean_std）。
  * 直前フレームを連結する LagStacker（バーストの立ち上がり形状を見る）。

純 numpy / scipy のみ。pyserial 等は読み込まない（学習側で import しやすくするため）。
"""

from collections import deque

import numpy as np

SAMPLING_RATE = 250
CHANNEL_NUM = 8

# 窓の既定値。実行側はこの値をモデル pkl の "window_samples" で上書きする。
WINDOW_S_DEFAULT = 0.4
HOP_S = 0.1                     # 判定ティック（10 Hz）。neutsuri.PREDICT_INTERVAL と同じ

# 帯域 [Hz]。50 Hz と 100 Hz の電源ノイズを避けて切る。
BANDS_DEFAULT = ((4.0, 12.0), (12.0, 20.0), (20.0, 45.0), (55.0, 95.0), (105.0, 124.0))

# 名前付きの帯域セット（探索の候補）。値は BANDS_DEFAULT と同じ形式。
BAND_SETS = {
    "default5": BANDS_DEFAULT,
    "emg3": ((20.0, 45.0), (55.0, 95.0), (105.0, 124.0)),
    "emg5": ((20.0, 30.0), (30.0, 45.0), (55.0, 75.0), (75.0, 95.0), (105.0, 124.0)),
    "fine8": ((4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 30.0), (30.0, 45.0),
              (55.0, 75.0), (75.0, 95.0), (105.0, 124.0)),
    "wide9": ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 45.0),
              (55.0, 70.0), (70.0, 95.0), (105.0, 124.0)),
    "fine12": ((1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 26.0),
               (26.0, 33.0), (33.0, 45.0), (55.0, 70.0), (70.0, 85.0), (85.0, 95.0),
               (105.0, 124.0)),
}

# 因果正規化（直前の履歴で中央値 / MAD）
NORM_HISTORY_S_DEFAULT = 60.0
NORM_WARMUP_FRAMES = 20         # これだけ溜まるまでは出力 0（= 中立）
NORM_MAD_SCALE = 1.4826         # MAD → 標準偏差相当
NORM_IQR_SCALE = 1.349          # IQR → 標準偏差相当
NORM_KINDS = ("none", "med_mad", "med_iqr", "mean_std")
EPS = 1e-12


# ==== 1 窓から計算する成分 ====
def band_log_power(window, fs=SAMPLING_RATE, bands=BANDS_DEFAULT):
    """(N, C) の窓から帯域ごとの log10 パワーを返す。形は (C * len(bands),)。

    平均を引いた後 Hann 窓をかけて FFT。並びは ch1 の全帯域、ch2 の全帯域、…。
    """
    return band_log_power_2d(window, fs, bands).reshape(-1)


def band_log_power_2d(window, fs=SAMPLING_RATE, bands=BANDS_DEFAULT):
    """band_log_power と同じ計算で、形を (C, len(bands)) のまま返す。"""
    x = np.asarray(window, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    win = np.hanning(n)
    spec = np.fft.rfft(x * win[:, None], axis=0)
    psd = (np.abs(spec) ** 2) / (np.sum(win ** 2) * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    out = np.empty((x.shape[1], len(bands)))
    for b, (lo, hi) in enumerate(bands):
        m = (freqs >= lo) & (freqs <= hi)
        out[:, b] = np.log10(psd[m].sum(axis=0) + EPS)
    return out


def band_log_power_cmr(window, fs=SAMPLING_RATE, bands=BANDS_DEFAULT):
    """帯域ごとにチャンネル平均を引いた対数パワー（共通成分除去）。(C * len(bands),)。

    電極インピーダンスやアンプのゲインが日によって動いても、帯域内の
    「どのチャンネルが相対的に強いか」は咬筋バーストの空間分布を反映して残る。
    """
    p = band_log_power_2d(window, fs, bands)
    return (p - p.mean(axis=0, keepdims=True)).reshape(-1)


def hjorth(window):
    """チャンネルごとの Hjorth (log activity, mobility, complexity)。(3*C,)。"""
    x = np.asarray(window, dtype=np.float64)
    dx = np.diff(x, axis=0)
    ddx = np.diff(dx, axis=0)
    v0 = np.var(x, axis=0) + EPS
    v1 = np.var(dx, axis=0) + EPS
    v2 = np.var(ddx, axis=0) + EPS
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return np.concatenate([np.log10(v0), mob, comp])


def line_length(window):
    """チャンネルごとの log10 平均 |差分|。(C,)。短い高周波バーストで素直に上がる。"""
    x = np.asarray(window, dtype=np.float64)
    return np.log10(np.abs(np.diff(x, axis=0)).mean(axis=0) + EPS)


def log_rms(window):
    """チャンネルごとの log10 標準偏差（平均を引いた後）。(C,)。"""
    x = np.asarray(window, dtype=np.float64)
    return np.log10(np.sqrt(np.var(x, axis=0)) + EPS)


# 成分名 → 1 窓を受け取る関数。bp 系だけ bands を使う。
FEATURE_COMPONENTS = {
    "bp": lambda w, fs, bands: band_log_power(w, fs, bands),
    "bpcmr": lambda w, fs, bands: band_log_power_cmr(w, fs, bands),
    "hjorth": lambda w, fs, bands: hjorth(w),
    "ll": lambda w, fs, bands: line_length(w),
    "rms": lambda w, fs, bands: log_rms(w),
}

# 名前 → 成分の並び。モデル pkl に名前を保存し、実行側はそれで組み立てる。
FEATURE_SETS = {
    "bp": ("bp",),
    "bp+hjorth": ("bp", "hjorth"),
    "bp+ll": ("bp", "ll"),
    "bp+hjorth+ll": ("bp", "hjorth", "ll"),
    "bp+bpcmr": ("bp", "bpcmr"),
    "bp+bpcmr+ll": ("bp", "bpcmr", "ll"),
    "bp+bpcmr+hjorth+ll": ("bp", "bpcmr", "hjorth", "ll"),
    "bpcmr": ("bpcmr",),
    "bpcmr+ll": ("bpcmr", "ll"),
    "bp+rms+ll": ("bp", "rms", "ll"),
}


def feature_dim(feature_set, n_bands, n_ch=CHANNEL_NUM):
    """特徴ベクトルの次元（Lag 連結前）。"""
    per = {"bp": n_ch * n_bands, "bpcmr": n_ch * n_bands, "hjorth": 3 * n_ch,
           "ll": n_ch, "rms": n_ch}
    return sum(per[c] for c in FEATURE_SETS[feature_set])


def frame_features(window, fs=SAMPLING_RATE, bands=BANDS_DEFAULT, feature_set="bp"):
    """1 窓 (N, C) → 特徴ベクトル。nan / inf は 0 に落とす（平坦な電極対策）。"""
    with np.errstate(all="ignore"):
        parts = [FEATURE_COMPONENTS[c](window, fs, bands) for c in FEATURE_SETS[feature_set]]
        f = parts[0] if len(parts) == 1 else np.concatenate(parts)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


# ==== 因果正規化 ====
class CausalNormalizer:
    """直前 history 秒の特徴ベクトルの中央値 / MAD で z 化する。

    押下は 1 秒に 1 回程度・バーストは 0.3 秒程度なので、履歴の中央値は
    ほぼ「噛んでいないとき」の値になる。セッションごとの電極インピーダンスや
    アンプのゲイン差をここで吸収し、モデルをセッション横断で使えるようにする。
    実行側では 1 ティック（0.1 s）ごとに update() を呼ぶ。

    kind で中心 / 広がりの統計量を選ぶ:
      med_mad  … 中央値と MAD（既定。v2 モデルはこれ）
      med_iqr  … 中央値と四分位範囲 / 1.349
      mean_std … 平均と標準偏差（外れ値に弱いが軽い）
    """

    def __init__(self, history_s=NORM_HISTORY_S_DEFAULT, hop_s=HOP_S,
                 warmup=NORM_WARMUP_FRAMES, kind="med_mad"):
        if kind not in NORM_KINDS:
            raise ValueError(f"未知の正規化: {kind}")
        self.kind = kind
        self.maxlen = max(warmup, int(round(history_s / hop_s)))
        self.warmup = warmup
        self.hist = deque(maxlen=self.maxlen)

    def reset(self):
        self.hist.clear()

    @property
    def ready(self):
        return len(self.hist) >= self.warmup

    def update(self, x):
        """x を履歴に加えてから、その x の正規化値を返す。"""
        x = np.asarray(x, dtype=np.float64)
        self.hist.append(x)
        if len(self.hist) < self.warmup:
            return np.zeros_like(x)
        h = np.asarray(self.hist)
        if self.kind == "mean_std":
            center = h.mean(axis=0)
            scale = h.std(axis=0)
        elif self.kind == "med_iqr":
            center = np.median(h, axis=0)
            q1, q3 = np.percentile(h, [25.0, 75.0], axis=0)
            scale = (q3 - q1) / NORM_IQR_SCALE
        else:
            center = np.median(h, axis=0)
            scale = np.median(np.abs(h - center), axis=0) * NORM_MAD_SCALE
        return (x - center) / (scale + 1e-6)


def make_normalizer(kind, history_s=NORM_HISTORY_S_DEFAULT, hop_s=HOP_S,
                    warmup=NORM_WARMUP_FRAMES):
    """kind が "none" なら None を返す。それ以外は CausalNormalizer。"""
    if kind in (None, "none", False):
        return None
    return CausalNormalizer(history_s, hop_s, warmup, kind=kind)


# ==== 直前フレームの連結 ====
class LagStacker:
    """現フレームと lag_step ごとの過去フレームを連結する（バーストの時間形状）。

    履歴が足りないうちは一番古いフレームで埋める（学習側の clip と同じ）。
    n_lags=1 なら何もしない。
    """

    def __init__(self, n_lags=1, lag_step=1):
        self.n_lags = max(1, int(n_lags))
        self.lag_step = max(1, int(lag_step))
        self.need = (self.n_lags - 1) * self.lag_step + 1
        self.hist = deque(maxlen=self.need)

    def reset(self):
        self.hist.clear()

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.n_lags == 1:
            return x
        self.hist.append(x)
        h = list(self.hist)
        out = []
        for k in range(self.n_lags):
            i = len(h) - 1 - k * self.lag_step
            out.append(h[max(0, i)])
        return np.concatenate(out)


# ==== オフライン（学習・評価）用 ====
def frame_ends(n_samples, window_samples, hop_samples):
    """オフラインで窓の終端インデックス列を作る（実行側のティックに相当）。"""
    return np.arange(window_samples, n_samples + 1, hop_samples)


def extract_frames(eeg, window_samples, hop_samples, fs=SAMPLING_RATE,
                   bands=BANDS_DEFAULT, feature_set="bp", normalizer=None,
                   stacker=None):
    """連続 EEG (N, C) を 10 Hz の特徴フレーム列にする。

    戻り値 (ends, X)。ends[k] は k 番目のフレームの終端（排他的）サンプル番号、
    X[k] はその窓の特徴（normalizer / stacker があれば実行側と同じ順で適用）。
    実行側と同じ順序で update() を呼ぶので、オフライン評価が実機と一致する。
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    ends = frame_ends(eeg.shape[0], window_samples, hop_samples)
    X = []
    for e in ends:
        f = frame_features(eeg[e - window_samples:e], fs, bands, feature_set)
        if normalizer is not None:
            f = normalizer.update(f)
        if stacker is not None:
            f = stacker.update(f)
        X.append(f)
    return ends, np.asarray(X)
