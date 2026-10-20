#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ne-utsuri モード — ESP32のピエゾ入力とEEGの同時記録、および判定の優先度混合。

Mac 側 `Ne-utsuri/main.py` の機能を Pi のブラウザUIへ移植したもの。聴覚BCI
(`bci_server.Experiment`) とは独立したモードで、刺激提示もラベル付与も行わない。
やることは3つだけ:

  1. ESP32（CP2102 経由でも C6 等の内蔵USBでも）から届く押下イベント "BTN" を
     EEG と同じ CSV に並べて残す。
  2. EEG から押下確率を 10 Hz で推定し、確率カーブの山を1回の押下に変換する。
     既定のモデル (neutsuri_train.py が書く v2) は 0.3 秒窓の帯域対数パワーを直前
     60 秒の中央値 / MAD で正規化して LDA に入れる。噛み込みは押下の ±0.2 秒に
     20–45 / 60–125 Hz の筋電バーストとして出るので、これを拾う。旧 2 秒窓・
     116 次元・RandomForest の pkl (v1) もそのまま読める。
  3. 「脳波予測」と「実センサー」のどちらを採用するかを 0–100% の優先度で混ぜ、
     採用された側が発火したら ESP32 へ "SPACE" を送る（ESP32 が BLE キーボードとして
     スペースキーを送出する）。

Mac 版との意図的な違い:

  * 推論を取得スレッドから外し、専用スレッドの 10 Hz タイマーで回す。Mac 版は
    サンプルコールバックの中で 25 サンプルごとに推論していたが、Pi でそれをやると
    RandomForest の推論時間がそのまま取得スレッドを止めてサンプル落ちに直結する。
  * 押下イベントは「そのとき届いた行」ではなく、イベント時刻に最も近い行へ打つ。
    Pi はチャンク単位（20 ms ≒ 5 サンプル）で読むので、届いた順に書くと最大 20 ms
    ずれる。ラベル付けの考え方は `bci_server.merge_labels` と同じ。
  * 判定は `predict_proba` の確率を使う。Mac 版の SVM 経路は 0/1 しか返さないため
    しきい値つまみが実質効かなかった。確率にするとしきい値がそのまま意味を持つ。
  * 不応期・ヒステリシス下側・再発火の規則はモデル pkl の値を使う（v2 は 0.4 s /
    −0.20 / 再上昇 0.2）。押下は 1 秒に 1 回前後、0.5 秒未満の連打もあるため。
  * 採用側の抽選は `random.random()*100 < usage` で行う。Mac 版の
    `randint(0,100) < usage` は 100% にしても 101 分の 1 でセンサー側が選ばれる。

CSV: `data/YYYYmmdd_HHMMSS_neutsuri.csv`
     Timestamp, Ch1..Ch8, PiezoVoltage, Prediction, Command
  * Ch1..Ch8    … brainflow が返す生値 µV（フィルタ無し。学習側と同じ条件）
  * PiezoVoltage… 押下イベント時刻に最も近い1サンプルだけ 1.0（Mac 版と同じスパイク）
  * Prediction  … 直近の推定確率。10 Hz 更新なので値は階段状になる
  * Command     … SPACE を送った行に 1.0（センサー由来）/ 2.0（脳波由来）
"""

import csv
import json
import os

# 1 サンプルだけの推論を OpenMP に並列化させると、スレッドを立てる方が高くつく。
# 96 コアの機械では 1 ティック 4 ms が 6700 ms まで悪化した（Pi でも 4 スレッドで
# 100 ms を超える跳ねが出る）。numpy より先に読み込まれたときのために環境変数でも
# 止めておき、読み込み済みの場合に備えて Predictor.load で threadpoolctl も使う。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random
import threading
import time
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports
from scipy.signal import welch

import neutsuri_features as nf

PIN_THREADS = os.environ.get("NEUTSURI_PIN_THREADS", "1") != "0"
_THREAD_PIN = None          # threadpoolctl のハンドル（保持している間だけ効く）


def pin_threads():
    """推論を 1 スレッドに固定する。1 度だけ効かせ、ハンドルを持ち続ける。"""
    global _THREAD_PIN
    if not PIN_THREADS or _THREAD_PIN is not None:
        return
    try:
        import threadpoolctl
        _THREAD_PIN = threadpoolctl.threadpool_limits(limits=1)
    except Exception:         # noqa: BLE001 - 無くても動く（環境変数側で効く想定）
        _THREAD_PIN = False


CHANNEL_NUM = 8
SAMPLING_RATE = 250
WINDOW_SAMPLES = 500        # 推論リングの長さ 2 s（v1 の窓そのもの。v2 は末尾 window_samples だけ使う）
PREDICT_INTERVAL = 0.1      # Mac 版の「25サンプルに1回」と同じ間隔
THRESH_HIGH_DEFAULT = 0.60
THRESH_LOW_MARGIN = 0.20    # low = high - margin（最小 0.10）
REFRACTORY_S = 0.8          # 発火後この秒数は再発火させない（v1 の既定。v2 は pkl の値）
STALE_AFTER_S = 1.0         # EEG が来なくなったら確率を捨てる

# ESP32 の見え方はボードによって2通りある。どちらも受け付ける。
#   * USB-シリアル変換チップ経由（旧来の DevKit の CP2102 / Silicon Labs）
#     … /dev/ttyUSB* に出る。RTS/DTR が EN/IO0 に配線されていて外からリセットできる。
#     Ne-utsuri/main.py の find_esp32_port と同じ判定。
#   * チップ内蔵USB（ESP32-C6/C3/S3 の USB Serial/JTAG）
#     … /dev/ttyACM* に出る。PID は製品ごとに変わるので VID だけで判定する。
ESP32_VID = 0x10C4              # CP2102
ESP32_PID = 0xEA60
ESP32_NATIVE_VID = 0x303A       # Espressif（チップ内蔵USB）
ESP32_USB_IDS = (
    (ESP32_VID, ESP32_PID),
    (ESP32_NATIVE_VID, None),   # None = PID は問わない
)
ESP32_BAUD = 115200
ESP32_RETRY_S = 3.0
ESP32_BOOT_WAIT_S = 1.5     # リセットを打ったあと BLE の初期化まで待つ

MODEL_FILENAME = "neutsuri_model.pkl"
CSV_SUFFIX = "_neutsuri.csv"
CSV_HEADER = (["Timestamp"] + [f"Ch{i}" for i in range(1, CHANNEL_NUM + 1)]
              + ["PiezoVoltage", "Prediction", "Command"])

CMD_VALUE = {"piezo": 1.0, "eeg": 2.0}


def is_neutsuri_csv(path):
    return os.path.basename(path).endswith(CSV_SUFFIX)


def sidecar_for(csv_path):
    """Ne-utsuri 記録の条件・実測タイミングを書く JSON のパス。

    聴覚BCI側の `_timing.json` と同じ「CSVの隣に置く」流儀に揃える。
    削除・リネームで一緒に動かす必要があるので、判定はここに集約する。
    """
    if not is_neutsuri_csv(csv_path):
        return None
    return os.path.splitext(csv_path)[0] + ".json"


# ==== 特徴量・判定モデル ====
# モデル pkl は 2 形式を受け付ける。
#   v2 … dict {"format": "neutsuri-v2", "model": 推定器, "window_samples", "feature_set",
#              "bands", "normalize", "norm_history_s", "threshold", "refractory_s", ...}
#        neutsuri_train.py が書く。特徴量は neutsuri_features（短窓の帯域対数パワー）、
#        直前 60 s の中央値 / MAD で因果正規化してから推定器に入れる。
#   v1 … sklearn Pipeline そのもの（旧 Ne-utsuri v1 スクリプトの 116 次元・2 秒窓）。
#        互換のために残す。特徴量関数はあちらの写し。
def hjorth_parameters(x):
    activity = np.var(x, axis=0)
    dy = np.diff(x, axis=0)
    mobility = np.sqrt(np.var(dy, axis=0) / activity)
    ddy = np.diff(dy, axis=0)
    complexity = np.sqrt(np.var(ddy, axis=0) / np.var(dy, axis=0)) / mobility
    return activity, mobility, complexity


def extract_features(eeg_window, fs=SAMPLING_RATE):
    """(N, 8) の窓から 116 次元の特徴量を作る（v1 モデル用。v1 script's extract_features と同一）。"""
    means = np.mean(eeg_window, axis=0)
    stds = np.std(eeg_window, axis=0)
    variances = np.var(eeg_window, axis=0)
    maxs = np.max(eeg_window, axis=0)
    mins = np.min(eeg_window, axis=0)

    activity, mobility, complexity = hjorth_parameters(eeg_window)

    corr_matrix = np.corrcoef(eeg_window, rowvar=False)
    corrs = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    band_powers = []
    for ch in range(eeg_window.shape[1]):
        nperseg = fs * 2 if len(eeg_window) >= fs * 2 else len(eeg_window)
        freqs, psd = welch(eeg_window[:, ch], fs, nperseg=nperseg)

        alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
        alpha_power = np.sum(psd[alpha_idx]) if len(alpha_idx) > 0 else 0

        beta_idx = np.where((freqs >= 12) & (freqs <= 30))[0]
        beta_power = np.sum(psd[beta_idx]) if len(beta_idx) > 0 else 0

        theta_idx = np.where((freqs >= 4) & (freqs <= 8))[0]
        theta_power = np.sum(psd[theta_idx]) if len(theta_idx) > 0 else 0

        band_powers.extend([alpha_power, beta_power, theta_power])

    return np.concatenate([means, stds, variances, maxs, mins,
                           activity, mobility, complexity, corrs, band_powers])


class Predictor:
    """`neutsuri_model.pkl` を読んで押下確率を返す（v2 / v1 の両形式）。"""

    def __init__(self, path):
        self.path = path
        self.model = None
        self.error = ""
        self.info = ""
        self.format = None          # "v3" / "v2" / "v1"
        self.window_samples = WINDOW_SAMPLES
        self.refractory_s = REFRACTORY_S
        self.low_margin = THRESH_LOW_MARGIN
        self.rearm_rise = None      # v2: 発火後、確率が最小値からこれだけ上がり直したら再発火
        self.persist = 1            # v3: しきい値超えがこのフレーム続いて初めて発火
        self.threshold = None       # 学習時に決めた推奨しきい値（v2 以降）
        self.feature_set = None
        self.bands = None
        self.normalizer = None
        self.stacker = None         # v3: 直前フレームの連結
        self.prob_smooth = 1        # v3: 確率の移動平均フレーム数
        self._smooth = deque(maxlen=1)
        self._pos = -1          # classes_ のうち「押下」の列
        self.load()

    def load(self):
        self.model = None
        self.info = ""
        self.format = None
        self.normalizer = None
        self.stacker = None
        self.prob_smooth = 1
        self._smooth = deque(maxlen=1)
        self.window_samples = WINDOW_SAMPLES
        self.refractory_s = REFRACTORY_S
        self.low_margin = THRESH_LOW_MARGIN
        self.rearm_rise = None
        self.persist = 1
        self.threshold = None
        if not os.path.exists(self.path):
            self.error = f"{os.path.basename(self.path)} がありません"
            return False
        try:
            import joblib
            obj = joblib.load(self.path)
            pin_threads()
            if isinstance(obj, dict) and str(obj.get("format", "")).startswith("neutsuri-v"):
                self._load_v2(obj)
            else:
                self._load_v1(obj)
            self.error = ""
        except Exception as exc:      # noqa: BLE001 - ブラウザに出す
            self.model = None
            self.error = f"{type(exc).__name__}: {exc}"
        return self.model is not None

    def _set_pos(self, model):
        classes = [float(c) for c in getattr(model, "classes_", [0.0, 1.0])]
        self._pos = classes.index(1.0) if 1.0 in classes else len(classes) - 1

    def _load_v1(self, model):
        self._set_pos(model)
        self.model = model
        self.format = "v1"
        n_in = getattr(model, "n_features_in_", None)
        self.info = f"v1 {type(model).__name__}" + (f" / {n_in}次元" if n_in else "") + " / 2.0 s窓"

    def _load_v2(self, obj):
        model = obj["model"]
        ws = int(obj["window_samples"])
        if ws > WINDOW_SAMPLES:
            raise ValueError(f"window_samples={ws} は推論リング {WINDOW_SAMPLES} より長い")
        if int(obj.get("fs", SAMPLING_RATE)) != SAMPLING_RATE:
            raise ValueError(f"fs={obj.get('fs')} のモデル（{SAMPLING_RATE} Hz 専用）")
        self.feature_set = obj.get("feature_set", "bp")
        if self.feature_set not in nf.FEATURE_SETS:
            raise ValueError(f"未知の feature_set: {self.feature_set}")
        self.bands = tuple(tuple(b) for b in obj.get("bands", nf.BANDS_DEFAULT))
        # v2 は normalize(bool) だけ、v3 は normalizer(名前) を持つ。
        kind = obj.get("normalizer") or ("med_mad" if obj.get("normalize", True) else "none")
        self.normalizer = nf.make_normalizer(
            kind, float(obj.get("norm_history_s", nf.NORM_HISTORY_S_DEFAULT)), PREDICT_INTERVAL)
        n_lags = int(obj.get("n_lags", 1))
        if n_lags > 1:
            self.stacker = nf.LagStacker(n_lags, int(obj.get("lag_step", 1)))
        self.prob_smooth = max(1, int(obj.get("prob_smooth", 1)))
        self._smooth = deque(maxlen=self.prob_smooth)
        self._set_pos(model)
        self.model = model
        self.format = "v2"
        self.window_samples = ws
        self.refractory_s = float(obj.get("refractory_s", REFRACTORY_S))
        self.low_margin = float(obj.get("low_margin", THRESH_LOW_MARGIN))
        rr = obj.get("rearm_rise")
        self.rearm_rise = float(rr) if rr is not None else None
        self.persist = max(1, int(obj.get("persist", 1)))
        thr = obj.get("threshold")
        self.threshold = float(thr) if thr is not None else None
        n_in = getattr(model, "n_features_in_", None)
        tag = str(obj.get("format", "neutsuri-v2")).replace("neutsuri-", "")
        self.info = (f"{tag} {type(model).__name__}" + (f" / {n_in}次元" if n_in else "")
                     + f" / {ws / SAMPLING_RATE:.1f} s窓"
                     + (f" / {kind}" if self.normalizer is not None else "")
                     + (f" / lag {n_lags}" if n_lags > 1 else "")
                     + (f" / 平滑 {self.prob_smooth}" if self.prob_smooth > 1 else "")
                     + (f" / 持続 {self.persist}" if self.persist > 1 else "")
                     + (f" / 推奨しきい値 {self.threshold:.2f}" if self.threshold is not None else ""))

    @property
    def available(self):
        return self.model is not None

    def reset(self):
        """正規化・連結・平滑の履歴を捨てる（モードON、記録開始、モデル再読込のとき）。"""
        if self.normalizer is not None:
            self.normalizer.reset()
        if self.stacker is not None:
            self.stacker.reset()
        self._smooth.clear()

    def probability(self, window):
        """推論リング (WINDOW_SAMPLES, 8) から押下確率を返す。失敗しても 0.0 を返して止まらない。

        v2 は末尾 window_samples だけを使う。毎ティック呼ばれる前提で、正規化の
        履歴もここで進める（呼ぶ間隔が PREDICT_INTERVAL でなくなると履歴長の
        意味が変わるので注意）。
        """
        raw = self._raw_probability(window)
        if self.prob_smooth <= 1:
            return raw
        self._smooth.append(raw)
        return float(np.mean(self._smooth))

    def _raw_probability(self, window):
        """平滑化前の 1 フレーム分の確率。正規化・連結の履歴はここで必ず進める。"""
        if self.model is None:
            return 0.0
        try:
            # 電極が完全に平らだと Hjorth や相関の除算が nan になる。0 埋めで通す
            # （警告はログを埋めるだけなので errstate で黙らせる）。
            with np.errstate(all="ignore"):
                if self.format == "v2":
                    feat = nf.frame_features(window[-self.window_samples:], SAMPLING_RATE,
                                             self.bands, self.feature_set)
                    ready = True
                    if self.normalizer is not None:
                        feat = self.normalizer.update(feat)
                        ready = self.normalizer.ready
                    if self.stacker is not None:
                        # 履歴の並びをティックとそろえるため、未準備でも必ず進める。
                        feat = self.stacker.update(feat)
                    if not ready:
                        return 0.0
                else:
                    feat = np.nan_to_num(extract_features(window),
                                         nan=0.0, posinf=0.0, neginf=0.0)
                feat = feat.reshape(1, -1)
            if hasattr(self.model, "predict_proba"):
                return float(self.model.predict_proba(feat)[0][self._pos])
            return float(self.model.predict(feat)[0])
        except Exception as exc:      # noqa: BLE001
            self.error = f"推論に失敗: {type(exc).__name__}: {exc}"
            return 0.0


def esp32_usb_ids_text():
    """探している VID:PID を1行にする（UI・エラー文用）。"""
    return " / ".join(f"{vid:04X}:{'****' if pid is None else format(pid, '04X')}"
                      for vid, pid in ESP32_USB_IDS)


def esp32_port_kind(port):
    """comports() の1件が ESP32 なら "native" / "bridge"、違えば None を返す。

    native はチップ内蔵USB（C6 等）。リセットの打ち方が変わるので区別する。
    """
    vid = getattr(port, "vid", None)
    pid = getattr(port, "pid", None)
    for want_vid, want_pid in ESP32_USB_IDS:
        if vid == want_vid and (want_pid is None or pid == want_pid):
            return "native" if vid == ESP32_NATIVE_VID else "bridge"
    return None


class Esp32Link:
    """ESP32 との シリアル接続。受信（BTN）と送信（SPACE）を1スレッドで回す。

    ファーム (`Ne-utsuri/piezo_interface/piezo_interface.ino`) の行プロトコル:
        受信  READY  … 起動した（ポートを開くとリセットされるので必ず出る）
              BTN    … ピエゾのスイッチが押された
              SENT   … SPACE を BLE キーボードとして送れた
              NO_BLE … SPACE を受けたが BLE 未接続で送れなかった
        送信  SPACE  … スペースキー送出 + ON_PIN を 200 ms ONにする

    SENT / NO_BLE が返るので、シリアルが繋がっているかとは別に「BLE の相手と
    繋がっているか」まで UI に出せる。切り分けができないと現場で詰むため。
    """

    def __init__(self, exclude_port=None):
        # Cyton ドングルも /dev/ttyUSB* に出るので、そのポートは候補から外す
        self._exclude_port = exclude_port or (lambda: None)

        self.port = None
        self.description = ""
        self.native = False         # チップ内蔵USB（リセットを打たない）
        self.connected = False
        self.error = ""
        self.presses = 0
        self.sent = 0
        self.noise = 0              # 既知トークン以外の受信行（ボーレート違いの目安）
        self.last_line = ""
        self.last_line_at = 0.0
        self.ble = "unknown"        # ok / ng / unknown
        self.opened_at = 0.0

        # plot/CSV 用のイベント列と、判定用のラッチ。Mac 版と同じく用途を分ける
        # （ラッチは実センサー分岐に入るまで保持、イベント列は記録用に消費する）。
        self.press_events = deque(maxlen=4096)
        self._latch = False

        self._outbox = deque()
        self._enabled = False
        self._stop = threading.Event()
        self._redetect = threading.Event()
        self._thread = None

    # --- 制御 ---
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def set_enabled(self, on):
        self._enabled = bool(on)
        if not on:
            self._latch = False
            self.press_events.clear()

    def redetect(self):
        """次のループで閉じて開き直させる（ポート差し替え・ファーム焼き直しのあと）。"""
        self._redetect.set()

    def send_space(self):
        self._outbox.append(b"SPACE\n")

    def take_press(self):
        """判定用ラッチを読み取って落とす（1押下が1回だけ通る）。"""
        if self._latch:
            self._latch = False
            return True
        return False

    def drain_events(self):
        out = []
        while self.press_events:
            out.append(self.press_events.popleft())
        return out

    # --- 検出 ---
    def find_port(self):
        """(デバイス, 説明, 種別) を返す。見つからなければ (None, "", None)。"""
        exclude = self._exclude_port()
        for p in serial.tools.list_ports.comports():
            if p.device == exclude:
                continue
            kind = esp32_port_kind(p)
            if kind:
                default = "USB Serial/JTAG" if kind == "native" else "CP2102"
                return p.device, (p.description or default), kind
        return None, "", None

    # --- スレッド ---
    def _worker(self):
        ser = None
        buf = bytearray()
        next_try = 0.0
        while not self._stop.is_set():
            if self._redetect.is_set():
                self._redetect.clear()
                ser = self._close(ser)
                next_try = 0.0

            if not self._enabled:
                if ser is not None:
                    ser = self._close(ser)
                time.sleep(0.2)
                continue

            if ser is None:
                if time.time() < next_try:
                    time.sleep(0.1)
                    continue
                next_try = time.time() + ESP32_RETRY_S
                ser = self._open()
                buf.clear()
                continue

            try:
                # readline は timeout で切れた断片をそのまま返すので使わない。
                # 生バイトを溜めて自分で \n で切る（"BT" / "N" に割れても壊れない）。
                data = ser.read(max(1, ser.in_waiting))
                if data:
                    buf.extend(data)
                    while b"\n" in buf:
                        line, _, rest = bytes(buf).partition(b"\n")
                        buf = bytearray(rest)
                        self._on_line(line.decode(errors="ignore").strip())

                while self._outbox:
                    ser.write(self._outbox.popleft())
                    self.sent += 1
            except Exception as exc:      # noqa: BLE001 - 抜線はここに来る
                self.error = f"通信が切れました: {type(exc).__name__}: {exc}"
                self.connected = False
                ser = self._close(ser)
                self.ble = "unknown"

    def _open(self):
        port, desc, kind = self.find_port()
        if port is None:
            self.connected = False
            self.port = None
            self.description = ""
            self.native = False
            self.error = f"ESP32 が見つかりません（{esp32_usb_ids_text()} を探しています）"
            return None
        try:
            ser = serial.Serial(port, ESP32_BAUD, timeout=0.02, write_timeout=1.0)
        except Exception as exc:      # noqa: BLE001
            self.connected = False
            self.error = f"{port} を開けません: {type(exc).__name__}: {exc}"
            return None

        if kind == "native":
            # チップ内蔵USBでは RTS/DTR がそのままチップの EN / IO0 に効く。同じ
            # パルスを打つと USB デバイスごと消え、いま開いたポートが無効になる
            # （再列挙まで数秒、ttyACM の番号も変わりうる）。掴んだ端から自分で
            # 切ることになるので打たない。開いた直後の DTR/RTS は pyserial が
            # どちらも True にする＝リセットの掛からない組み合わせなので触らない。
            # 代わりにリセットしないぶん起動時の "READY" は届かない。ファームが
            # 生きているかは BTN と、接続テストの SENT / NO_BLE で見る。
            print(f"[esp32] {port}: 内蔵USBなのでリセットは打ちません（READY は出ません）")
        else:
            # 変換チップ経由は、ポートを開くだけでリセットが掛かるボードとそうで
            # ないボードがあるので、esptool と同じ RTS/DTR シーケンスで明示的に
            # リセットを打つ。前のセッションのまま固まっている ESP32 もこれで復帰し、
            # 起動直後の "READY" が必ず届くので「ファームが動いている」ことを表示できる。
            # 受信バッファは捨てない（READY を消してしまう。ROM のゴミは _on_line が
            # 既知トークン以外を無視して落とす）。
            try:
                ser.dtr = False
                ser.rts = True          # EN を落として保持
                time.sleep(0.12)
                ser.rts = False         # 解放してアプリを起動
            except Exception as exc:      # noqa: BLE001 - 対応していないブリッジもある
                print(f"[esp32] reset pulse skipped: {type(exc).__name__}: {exc}")
            time.sleep(ESP32_BOOT_WAIT_S)   # BLE の初期化まで待つ

        self.port = port
        self.description = desc
        self.native = (kind == "native")
        self.connected = True
        self.error = ""
        self.ble = "unknown"
        self.opened_at = time.time()
        self._outbox.clear()
        print(f"[esp32] connected: {port} ({desc}) kind={kind}")
        return ser

    def _close(self, ser):
        if ser is not None:
            try:
                ser.close()
            except Exception:      # noqa: BLE001
                pass
        self.connected = False
        return None

    def _on_line(self, line):
        # 既知のトークン以外は数えるだけで無視する。ESP32 のブートROMは 115200 で
        # 文字化けを吐くので、それを「最終受信」として出すと接続表示が濁る。
        if line not in ("BTN", "SENT", "NO_BLE", "READY"):
            if line:
                self.noise += 1
            return
        self.last_line = line
        self.last_line_at = time.time()
        if line == "BTN":
            self.presses += 1
            self._latch = True
            self.press_events.append(self.last_line_at)
        elif line == "SENT":
            self.ble = "ok"
        elif line == "NO_BLE":
            self.ble = "ng"
        elif line == "READY":
            self.ble = "unknown"

    def status(self):
        now = time.time()
        return {
            "connected": self.connected,
            "port": self.port,
            "description": self.description,
            "error": self.error,
            "native": self.native,
            "vidpid": esp32_usb_ids_text(),
            "presses": self.presses,
            "sent": self.sent,
            "noise": self.noise,
            "ble": self.ble,
            "last_line": self.last_line,
            "last_line_age": round(now - self.last_line_at, 1) if self.last_line_at else None,
        }


class NeutsuriEngine:
    """Ne-utsuri モード本体。取得スレッドから on_chunk / build_frame を呼ばれる。"""

    def __init__(self, base_dir, data_dir, buffer_size, decim,
                 save_setting=None, exclude_port=None, model_dir=None):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.buffer_size = buffer_size       # 表示リングと同じ長さ（6 s 分）
        self.decim = decim
        self._save_setting = save_setting or (lambda *_: None)

        # モデルは被験者のセット models/<被験者>/neutsuri_model.pkl を優先し、
        # 無ければ従来どおりアプリ直下の neutsuri_model.pkl を使う。
        self.model_dir = model_dir
        self.predictor = Predictor(self.model_path())
        self.link = Esp32Link(exclude_port)

        self.enabled = False
        self.usage = 0                       # 脳波利用率 [%]（判定の優先度）
        self.threshold = THRESH_HIGH_DEFAULT
        self.message = ""

        self.current_prediction = 0.0
        self.pred_at = 0.0
        self.pred_ms = 0.0                   # 推論1回の所要時間（Piの余力の目安）
        self.eeg_at = 0.0
        self.commands = {"eeg": 0, "piezo": 0}
        self.last_command = None

        # 表示用リング（EEG表示バッファと同じ index に書くので波形と時間軸が揃う）
        self.trace = np.zeros((3, buffer_size), dtype=np.float32)
        self.frame_shape = (3, buffer_size // decim)
        self.frame = np.zeros(self.frame_shape, dtype=np.float32).tobytes()

        # 推論用リング（生値 µV、2秒）
        self._win = np.zeros((WINDOW_SAMPLES, CHANNEL_NUM), dtype=np.float64)
        self._win_idx = 0
        self._win_filled = 0
        self._win_lock = threading.Lock()

        # チャンク境界より後ろのイベントは次のチャンクへ回す
        self._pending_press = deque()
        self._pending_cmd = deque()

        self.recording = False
        self.file = None
        self.samples = 0
        self.started_at = 0.0
        self._csv_file = None
        self._csv_writer = None
        self._rec_lock = threading.Lock()
        self._rec = None                     # 記録中の集計

        self._above = False                  # しきい値ヒステリシスの状態
        self._min_since_fire = 1.0           # 発火後の確率の最小値（再発火の判定用）
        self._last_fire = 0.0
        self._run = 0                        # しきい値を連続で超えているティック数
        self._stop = threading.Event()
        self._thread = None

    # --- 起動 / 設定 ---
    def start(self):
        self.link.start()
        if self._thread is None:
            self._thread = threading.Thread(target=self._predict_worker, daemon=True)
            self._thread.start()

    def model_path(self):
        if self.model_dir:
            p = os.path.join(self.model_dir, MODEL_FILENAME)
            if os.path.exists(p):
                return p
        return os.path.join(self.base_dir, MODEL_FILENAME)

    def set_model_dir(self, model_dir):
        """被験者を切り替えたら、その被験者のモデルに読み替える（無ければ共通のもの）。"""
        self.model_dir = model_dir
        path = self.model_path()
        if path != self.predictor.path:
            self.predictor.path = path
            self.predictor.load()
        return self.predictor.available

    def reload_model(self):
        """モデル再読込（被験者フォルダに新しい pkl を置いた後にも効くよう、パスも引き直す）。"""
        self.predictor.path = self.model_path()
        return self.predictor.load()

    def restore(self, settings):
        """settings.json から前回の条件を戻す（提示条件と同じく状態依存にしない）。

        ON/OFF は戻さない。ONのままだと起動しただけで ESP32 のシリアルを掴み、
        使うつもりのないときに Cyton ドングルと取り合いになる。モードは常に
        OFF で立ち上げ、使う人が明示的に ON にする。
        """
        if "neutsuri_usage" in settings:
            self.usage = max(0, min(100, int(settings["neutsuri_usage"])))
        if "neutsuri_threshold" in settings:
            self.threshold = max(0.30, min(0.95, float(settings["neutsuri_threshold"])))

    def resize(self, buffer_size, decim):
        """表示リングの長さ・間引き率をEEG表示バッファに合わせ直す。

        アンプを切り替えるとレートが変わり表示リングの長さも変わる。取得側と
        長さが違うままだと、同じ index に書いているつもりで範囲外に出てしまう。
        """
        if buffer_size == self.buffer_size and decim == self.decim:
            return
        self.buffer_size = buffer_size
        self.decim = decim
        self.trace = np.zeros((3, buffer_size), dtype=np.float32)
        self.frame_shape = (3, buffer_size // decim)
        self.frame = np.zeros(self.frame_shape, dtype=np.float32).tobytes()

    def set_enabled(self, on):
        on = bool(on)
        if not on and self.recording:
            return False, "記録中はOFFにできません"
        self.enabled = on
        self.link.set_enabled(on)
        if not on:
            # OFFの間に溜めたイベントは捨てる。残すと次にONにした最初のチャンクへ
            # 古い押下が載ってしまう。
            self._pending_press.clear()
            self._pending_cmd.clear()
            self.trace[:] = 0.0
            self.current_prediction = 0.0
            self._above = False
            self._run = 0
            with self._win_lock:
                self._win_filled = 0
        # 正規化の履歴は電極を付け直した後の値で作り直す（OFF 中の古い値を混ぜない）
        self.predictor.reset()
        return True, f"Ne-utsuri モード: {'ON' if on else 'OFF'}"

    def set_usage(self, value):
        self.usage = max(0, min(100, int(value)))
        self._save_setting("neutsuri_usage", self.usage)
        return f"判定の優先度: 脳波 {self.usage}% / センサー {100 - self.usage}%"

    def set_threshold(self, value):
        self.threshold = max(0.30, min(0.95, float(value)))
        self._save_setting("neutsuri_threshold", self.threshold)
        return f"予測しきい値: {self.threshold:.2f}（下側 {self.low_threshold:.2f}）"

    @property
    def low_threshold(self):
        return max(0.10, self.threshold - self.predictor.low_margin)

    # --- 取得スレッドから呼ばれる ---
    def on_chunk(self, idx, stamps, eeg):
        """1チャンク分の処理。表示リング更新と同じタイミングで呼ばれる。

        idx    … 表示リング上の書き込み位置（長さ n）
        stamps … brainflow のタイムスタンプ（長さ n、昇順）
        eeg    … 生値 µV (8, n)
        """
        if not self.enabled:
            return
        n = eeg.shape[1]
        self._push_window(eeg)
        self.eeg_at = time.time()

        for ts in self.link.drain_events():
            self._pending_press.append((ts, 1.0))

        piezo = self._marks(self._pending_press, stamps)
        cmd = self._marks(self._pending_cmd, stamps)
        pred = round(self.prediction(), 4)

        self.trace[0, idx] = piezo
        self.trace[1, idx] = pred
        self.trace[2, idx] = cmd

        with self._rec_lock:
            if self.recording and self._csv_writer is not None:
                self._csv_writer.writerows(
                    [stamps[i], *eeg[:, i], piezo[i], pred, cmd[i]] for i in range(n)
                )
                self.samples += n
                self._tally(stamps, piezo, cmd)

    def build_frame(self, write_idx):
        """ブラウザに渡す間引きトレースを作る（リクエスト側は読むだけ）。"""
        if not self.enabled:
            return
        t = np.concatenate((self.trace[:, write_idx:], self.trace[:, :write_idx]), axis=1)
        m = t.shape[1] // self.decim
        blocks = t[:, :m * self.decim].reshape(3, m, self.decim)
        frame = np.empty((3, m), dtype=np.float32)
        # 押下とコマンドは1サンプル幅のスパイクなので、単純な間引きでは大半が消える。
        # 区間の最大を取って必ず残す。確率は連続量なので平均。
        frame[0] = blocks[0].max(axis=1)
        frame[1] = blocks[1].mean(axis=1)
        frame[2] = blocks[2].max(axis=1)
        self.frame = np.ascontiguousarray(frame).tobytes()
        self.frame_shape = (3, m)

    def _push_window(self, eeg):
        n = eeg.shape[1]
        with self._win_lock:
            if n >= WINDOW_SAMPLES:
                self._win[:] = eeg[:, -WINDOW_SAMPLES:].T
                self._win_idx = 0
                self._win_filled = WINDOW_SAMPLES
                return
            pos = (self._win_idx + np.arange(n)) % WINDOW_SAMPLES
            self._win[pos, :] = eeg.T
            self._win_idx = int((self._win_idx + n) % WINDOW_SAMPLES)
            self._win_filled = min(WINDOW_SAMPLES, self._win_filled + n)

    def _window(self):
        with self._win_lock:
            if self._win_filled < WINDOW_SAMPLES:
                return None
            i = self._win_idx
            return np.concatenate((self._win[i:], self._win[:i]), axis=0).copy()

    @staticmethod
    def _marks(pending, stamps):
        """イベント時刻に最も近いサンプルへ値を立てた (n,) 配列を返す。

        チャンクの最後のサンプルより新しいイベントは、まだ収まる行が来ていないので
        次のチャンクへ回す（届いた順に詰めると最大1チャンク分ずれる）。
        """
        out = np.zeros(len(stamps), dtype=np.float32)
        if not pending or len(stamps) == 0:
            return out
        last = stamps[-1]
        while pending:
            ts, value = pending[0]
            if ts > last:
                break
            pending.popleft()
            j = int(np.abs(stamps - ts).argmin())
            out[j] = max(out[j], value)
        return out

    def prediction(self):
        """表示・記録に使う確率。EEG が止まっていたら古い値を出さない。"""
        if self.eeg_at and time.time() - self.eeg_at > STALE_AFTER_S:
            return 0.0
        return self.current_prediction

    # --- 判定ループ ---
    def _predict_worker(self):
        while not self._stop.is_set():
            time.sleep(PREDICT_INTERVAL)
            if not self.enabled:
                continue
            try:
                self._tick()
            except Exception as exc:      # noqa: BLE001 - ループは止めない
                self.message = f"判定でエラー: {type(exc).__name__}: {exc}"

    def _tick(self):
        stale = bool(self.eeg_at) and (time.time() - self.eeg_at > STALE_AFTER_S)
        win = None if stale else self._window()
        if win is not None and self.predictor.available:
            t0 = time.time()
            prob = self.predictor.probability(win)
            self.pred_ms = round((time.time() - t0) * 1000, 1)
        else:
            prob = 0.0
        self.current_prediction = prob
        self.pred_at = time.time()

        fired = self._edge(prob)

        # 判定の優先度。usage% の確率で脳波の判定を、残りで実センサーを採用する。
        if random.random() * 100.0 < self.usage:
            # 脳波側の回。実センサーのラッチはここでは触らない（押下を捨てない）。
            if fired:
                self._fire("eeg")
        elif self.link.take_press():
            self._fire("piezo")

    def _edge(self, prob):
        """確率が high を上抜けた瞬間に1回だけ True（low まで戻るまで再発火しない）。

        v2 モデルでは加えて、発火後に確率が一度下がって（最小値から rearm_rise 以上）
        上がり直したら、low まで戻っていなくても再発火する。0.5 秒未満の連打では
        バーストの確率が high の上に乗ったまま次の押下が来て、low を待つと 2 回目を
        落とすため（neutsuri_train.py の評価で検出率が約 5 ポイント上がる）。

        v3 の persist は「high を連続で何ティック超えたら発火するか」。1 回だけ跳ねた
        確率で発火しなくなるので、誤発火を減らす代わりに最大 (persist-1) × 0.1 s 遅れる。
        """
        now = time.time()
        self._run = self._run + 1 if prob >= self.threshold else 0
        ok = self._run >= self.predictor.persist
        if not self._above:
            if ok and (now - self._last_fire) >= self.predictor.refractory_s:
                self._above = True
                self._last_fire = now
                self._min_since_fire = prob
                return True
        else:
            self._min_since_fire = min(self._min_since_fire, prob)
            if prob <= self.low_threshold:
                self._above = False
            elif (self.predictor.rearm_rise is not None and ok
                  and prob - self._min_since_fire >= self.predictor.rearm_rise
                  and (now - self._last_fire) >= self.predictor.refractory_s):
                self._last_fire = now
                self._min_since_fire = prob
                return True
        return False

    def _fire(self, source):
        ts = time.time()
        self.link.send_space()
        self._pending_cmd.append((ts, CMD_VALUE[source]))
        self.commands[source] += 1
        self.last_command = {"at": ts, "source": source}

    # --- 記録 ---
    def start_recording(self):
        if not self.enabled:
            return False, "Ne-utsuri モードがOFFです"
        with self._rec_lock:
            if self.recording:
                return False, "すでに記録中です"
            path = os.path.join(self.data_dir,
                                time.strftime("%Y%m%d_%H%M%S") + CSV_SUFFIX)
            try:
                self._csv_file = open(path, "w", newline="")
            except OSError as exc:
                return False, f"ファイルを作れません: {exc}"
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(CSV_HEADER)
            self.file = path
            self.samples = 0
            self.started_at = time.time()
            self._rec = {
                "presses": 0, "commands": {"eeg": 0, "piezo": 0},
                "prev_ts": None, "n_int": 0, "dt_sum": 0.0, "dt_sq": 0.0,
                "dt_min": None, "dt_max": None, "first_ts": None, "last_ts": None,
            }
            self.recording = True
        self.message = f"記録開始: {os.path.basename(path)}"
        return True, self.message

    def stop_recording(self):
        with self._rec_lock:
            if not self.recording:
                return False, "記録していません"
            self.recording = False
            path, rec, samples = self.file, self._rec, self.samples
            if self._csv_file is not None:
                self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            self._rec = None
        self._write_sidecar(path, rec, samples)
        self.message = (f"記録停止: {os.path.basename(path)} "
                        f"（{samples} サンプル / 押下 {rec['presses']} 回 / "
                        f"送信 {rec['commands']['eeg'] + rec['commands']['piezo']} 回）")
        return True, self.message

    def _tally(self, stamps, piezo, cmd):
        """記録中の集計をその場で進める（停止後にCSVを読み直さなくて済む）。"""
        r = self._rec
        r["presses"] += int((piezo > 0).sum())
        r["commands"]["piezo"] += int((cmd == CMD_VALUE["piezo"]).sum())
        r["commands"]["eeg"] += int((cmd == CMD_VALUE["eeg"]).sum())
        if r["first_ts"] is None:
            r["first_ts"] = float(stamps[0])
        r["last_ts"] = float(stamps[-1])

        seq = stamps if r["prev_ts"] is None else np.concatenate(([r["prev_ts"]], stamps))
        if len(seq) > 1:
            d = np.diff(seq.astype(np.float64))
            r["n_int"] += d.size
            r["dt_sum"] += float(d.sum())
            r["dt_sq"] += float((d ** 2).sum())
            r["dt_min"] = float(d.min()) if r["dt_min"] is None else min(r["dt_min"], float(d.min()))
            r["dt_max"] = float(d.max()) if r["dt_max"] is None else max(r["dt_max"], float(d.max()))
        r["prev_ts"] = float(stamps[-1])

    def _write_sidecar(self, path, rec, samples):
        """記録の条件と実測タイミングを CSV の隣に残す（CSV は一切書き換えない）。"""
        interval = None
        if rec and rec["n_int"] > 0:
            mean = rec["dt_sum"] / rec["n_int"]
            var = max(0.0, rec["dt_sq"] / rec["n_int"] - mean ** 2)
            interval = {"mean": round(mean * 1000, 4), "sd": round(var ** 0.5 * 1000, 4),
                        "min": round(rec["dt_min"] * 1000, 4),
                        "max": round(rec["dt_max"] * 1000, 4)}
        span = (rec["last_ts"] - rec["first_ts"]) if rec and rec["first_ts"] is not None else 0.0
        report = {
            "file": os.path.basename(path),
            "mode": "neutsuri",
            "samples": samples,
            "duration_s": round(span, 4),
            "effective_fs": round((samples - 1) / span, 4) if samples > 1 and span > 0 else None,
            "sample_interval_ms": interval,
            "presses": rec["presses"] if rec else 0,
            "commands": rec["commands"] if rec else {},
            "usage_percent": self.usage,
            "threshold": {"high": round(self.threshold, 3), "low": round(self.low_threshold, 3),
                          "refractory_s": self.predictor.refractory_s,
                          "rearm_rise": self.predictor.rearm_rise},
            "predict_interval_s": PREDICT_INTERVAL,
            "window_samples": self.predictor.window_samples,
            "model": os.path.relpath(self.predictor.path, self.base_dir) if self.predictor.available else None,
            "model_format": self.predictor.format,
            "model_info": self.predictor.info or None,
            "model_error": self.predictor.error or None,
            "esp32": {"port": self.link.port, "ble": self.link.ble},
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            with open(sidecar_for(path), "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except OSError as exc:      # noqa: BLE001
            print(f"[neutsuri] sidecar failed: {exc}")

    # --- 状態 ---
    def status(self):
        now = time.time()
        return {
            "enabled": self.enabled,
            "usage": self.usage,
            "threshold": round(self.threshold, 2),
            "threshold_low": round(self.low_threshold, 2),
            "prediction": round(self.prediction(), 3),
            "predict_ms": self.pred_ms,
            "eeg_fresh": bool(self.eeg_at) and (now - self.eeg_at <= STALE_AFTER_S),
            "recording": self.recording,
            "file": os.path.basename(self.file) if self.file else None,
            "samples": self.samples,
            "elapsed": round(now - self.started_at, 1) if self.recording else None,
            "commands": dict(self.commands),
            "last_command": self.last_command,
            "message": self.message,
            "model_ok": self.predictor.available,
            "model_path": os.path.relpath(self.predictor.path, self.base_dir),
            "model_info": self.predictor.info,
            "model_error": self.predictor.error,
            "model_threshold": self.predictor.threshold,
            "refractory_s": self.predictor.refractory_s,
            "esp32": self.link.status(),
        }
