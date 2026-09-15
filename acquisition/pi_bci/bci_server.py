#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless auditory-BCI acquisition server for the Raspberry Pi.

Same experiment as main.py (Cyton 8ch @250Hz, four tones at 380/524/708/932 Hz,
0.15 s each with a 0.35 s ISI), but with no Qt GUI: the machine runs headless and
everything is driven from a browser over the direct LAN link.

Differences from the Mac main.py that are deliberate:
  * Acquisition goes through brainflow instead of pyOpenBCI, so the sample
    timestamps come from brainflow's own timestamp channel.
  * Stimuli are scheduled on the sound card's DAC clock through a persistent
    sounddevice output stream, and each onset timestamp is the wall-clock time
    the sample actually reaches the DAC. This removes the sleep-loop jitter of
    the Mac version.
  * LED / tactile ESP32 feedback is not ported.

Run:  python3 bci_server.py            (then open http://192.168.137.20:8080/)
"""

import csv
import glob
import json
import logging
import os
import random
import re
import subprocess
import threading
import time
from collections import deque
from math import gcd

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import serial.tools.list_ports
from flask import (Flask, Response, after_this_request, jsonify, request,
                   send_file, send_from_directory)
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ==== 設定 ====
CHANNEL_NUM = 8
SAMPLING_RATE = 250
WINDOW_SECONDS = 6
FILTER_BAND = (0.5, 40)     # 表示用フィルタ（CSVには生値を保存する）
NUMTAPS = 11

SET_NUM = 10                # 1試行あたりの刺激セット数（main.py の svm 設定と同じ）
SOUND_LENGTH = 0.15
ISI = 0.35
SOA = ISI + SOUND_LENGTH    # 刺激間隔 0.5 s

# ポート80で待てば URL は http://raspberrypi.local/ だけで済む。root なしで80を
# 掴めるよう Pi 側で net.ipv4.ip_unprivileged_port_start=80 を設定してあるが、
# 掴めない環境では 8080 に落とす。
WEB_PORT = int(os.environ.get("BCI_WEB_PORT", "80"))
WEB_PORT_FALLBACK = 8080
DISPLAY_DECIM = 5           # 表示は 250Hz / 5 = 50 Hz
DERIVED_INTERVAL = 0.2      # 表示用データを作り直す間隔 [s]
READ_INTERVAL = 0.02        # brainflow から吸い出す間隔 [s]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DEVICE_MATCH = "USB Audio"   # USBオーディオI/Fを優先して掴む

# 刺激キー → (周波数, パン)。main.py の frequencies と同じ。
STIM_DEF = {
    "1": (524, "left"),
    "2": (708, None),
    "3": (932, "right"),
    "4": (0, None),          # 無音
}
STIM_KEYS = list(STIM_DEF.keys())
TARGET_KEYS = ["1", "2", "3"]      # L / C / R
TARGET_LABEL = {"1": "Left", "2": "Center", "3": "Right"}

# onsets キューに積まれるが試行の記録対象ではない予約の index
PREVIEW_INDEX = -1      # L/C/R の音確認ボタン
ANNOUNCE_INDEX = -2     # ビープ・音声アナウンス

TARGET_ANNOUNCE_WAV = {
    "1": "target_left_jp.wav",
    "2": "target_center_jp.wav",
    "3": "target_right_jp.wav",
}
RESULT_ANNOUNCE_WAV = {
    1: "result_left_jp.wav",
    2: "result_center_jp.wav",
    3: "result_right_jp.wav",
}

os.makedirs(DATA_DIR, exist_ok=True)

# ==== 設定の永続化 ====
# ALSAの音量は再起動で既定値に戻ってしまう。刺激の提示レベルは実験条件そのものなので、
# システム任せにせずアプリ側で持って起動時に復元する。
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")


def load_settings():
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_setting(key, value):
    data = load_settings()
    data[key] = value
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:      # noqa: BLE001
        print(f"[settings] save failed: {exc}")


# ==== 記録済みファイル数 ====
_counts_cache = {"at": 0.0, "value": None}


def recording_counts(ttl=2.0):
    """data/ にターゲットごとに何本記録済みかを数える。

    ファイル名は `YYYYmmdd_HHMMSS_<target>.csv` なので末尾のキーで振り分ける。
    /api/state は毎秒何度も叩かれるので短いTTLでキャッシュする。
    """
    now = time.time()
    if _counts_cache["value"] is not None and now - _counts_cache["at"] < ttl:
        return _counts_cache["value"]

    counts = {key: 0 for key in TARGET_KEYS}
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        key = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
        if key in counts:
            counts[key] += 1
    _counts_cache.update(at=now, value=counts)
    return counts


# ==== 刺激音の生成 ====
def generate_tone(frequency, duration, sample_rate, pan):
    """main.py の generate_tone と同じ波形を作る（常にステレオで返す）。"""
    n = int(sample_rate * duration)
    if frequency == 0:
        return np.zeros((n, 2), dtype=np.float32)

    t = np.linspace(0, duration, n, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    if pan is None:
        # main.py はパン無しのとき振幅をさらに半分にしている
        tone = tone * 0.5
        stereo = np.column_stack((tone, tone))
    elif pan == "left":
        stereo = np.column_stack((tone, np.zeros_like(tone)))
    else:
        stereo = np.column_stack((np.zeros_like(tone), tone))
    return stereo.astype(np.float32)


def find_output_device():
    """USBオーディオI/Fを探す。見つからなければ既定の出力デバイスを使う。

    戻り値の3つ目は「目的のUSBデバイスを掴めたか」。フォールバックした場合は
    内蔵ジャックで鳴ってしまい、ALSAの音量つまみも辿れなくなるので、
    その旨をUIまで持っていけるようにする。
    """
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] >= 2 and AUDIO_DEVICE_MATCH.lower() in dev["name"].lower():
            return idx, dev, True
    idx = sd.default.device[1]
    return idx, sd.query_devices(idx), False


class AlsaMixer:
    """USBオーディオI/Fのハードウェア音量を amixer 経由で操作する。

    デバイス名の "(hw:3,0)" からカード番号を拾い、pvolume を持つ最初の
    simple control（C-Media系なら 'Speaker'）を対象にする。
    """

    def __init__(self, device_name):
        self.card = None
        self.control = None
        self.vmin = 0
        self.vmax = 0
        self.percent = None

        m = re.search(r"hw:(\d+)", device_name or "")
        if not m:
            return
        self.card = int(m.group(1))

        try:
            out = subprocess.run(["amixer", "-c", str(self.card), "scontrols"],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:      # noqa: BLE001
            self.card = None
            return

        for name in re.findall(r"Simple mixer control '([^']+)'", out):
            detail = self._detail(name)
            if detail and "pvolume" in detail:
                limits = re.search(r"Limits: Playback (\d+) - (\d+)", detail)
                if limits:
                    self.control = name
                    self.vmin, self.vmax = int(limits.group(1)), int(limits.group(2))
                    break

        # amixer はサブプロセスなので状態はキャッシュし、変更時にだけ更新する
        self.percent = self.get_percent()

    def _detail(self, control):
        try:
            return subprocess.run(["amixer", "-c", str(self.card), "sget", control],
                                  capture_output=True, text=True, timeout=5).stdout
        except Exception:      # noqa: BLE001
            return ""

    @property
    def available(self):
        return self.control is not None

    def get_percent(self):
        if not self.available:
            return None
        m = re.search(r"\[(\d+)%\]", self._detail(self.control))
        return int(m.group(1)) if m else None

    def set_percent(self, percent):
        if not self.available:
            return False
        percent = max(0, min(100, int(percent)))
        # amixer の % 指定は dB カーブ依存で誤差が出るので、生の値で指定する
        raw = self.vmin + round((self.vmax - self.vmin) * percent / 100)
        try:
            subprocess.run(["amixer", "-c", str(self.card), "sset", self.control, str(raw)],
                           capture_output=True, text=True, timeout=5, check=True)
            self.percent = self.get_percent()
            return True
        except Exception:      # noqa: BLE001
            return False


class AudioEngine:
    """DACクロック上で刺激を並べる常時稼働の出力ストリーム。

    刺激は「出力フレーム番号」で予約する。フレーム番号は等間隔なので提示間隔に
    ドリフトが出ず、実際にDACへ届いた時刻を onset として記録できる。
    """

    BLOCKSIZE = 256

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = []          # 予約中の刺激
        self.onsets = deque()       # (seq_index, onset_wall) を再生順に積む
        self.gain = 1.0             # 刺激・アナウンス共通のソフトウェアゲイン
        self.stream = None
        self._open()

    def _open(self):
        """出力デバイスを選び直してストリームを張る。

        目的のデバイスが掴めなかった場合でもストリーム無しの状態にはしない。
        鳴らせないままだと以降の試行が全て AttributeError で落ちてしまう。
        """
        self.device, info, self.is_usb = find_output_device()
        try:
            self._start_stream(info)
        except Exception as exc:      # noqa: BLE001
            print(f"[audio] cannot open '{info['name']}': {exc} — using system default")
            self.device = sd.default.device[1]
            self.is_usb = False
            self._start_stream(sd.query_devices(self.device))

    def _start_stream(self, info):
        self.samplerate = int(info["default_samplerate"])
        self.device_name = info["name"]

        self._frame_counter = 0     # ストリーム開始からの出力フレーム数
        self._anchor_frame = 0      # 直近ブロック先頭のフレーム番号
        self._anchor_wall = time.time()   # そのブロックがDACに出る実時刻
        self._wav_cache = {}        # サンプルレートが変わるので作り直す

        stream = sd.OutputStream(
            device=self.device, samplerate=self.samplerate, channels=2,
            dtype="float32", blocksize=self.BLOCKSIZE, latency="low",
            callback=self._callback,
        )
        stream.start()
        self.stream = stream

        self.buffers = {
            key: self._resample_tone(freq, pan)
            for key, (freq, pan) in STIM_DEF.items()
        }

    def reopen(self):
        """USBオーディオを挿し直したあと、サービス再起動なしで掴み直す。

        PortAudio はデバイス一覧を初期化時にキャッシュするので、挿抜を反映させるには
        いったん終了して初期化し直す必要がある。開いたままのストリームがあると
        落ちるので、必ず閉じてから行う。
        """
        with self._lock:
            self._pending = []
            self.onsets.clear()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as exc:      # noqa: BLE001
                print(f"[audio] closing old stream: {exc}")
            self.stream = None
        try:
            sd._terminate()
            sd._initialize()
        except Exception as exc:      # noqa: BLE001
            print(f"[audio] PortAudio re-init skipped: {exc}")
        self._open()
        print(f"[audio] reopened: device={self.device_name} sr={self.samplerate} "
              f"usb={self.is_usb}")

    def _resample_tone(self, freq, pan):
        return generate_tone(freq, SOUND_LENGTH, self.samplerate, pan)

    # --- audio thread ---
    def _callback(self, outdata, frames, time_info, status):
        wall_now = time.time()
        # PortAudio のストリーム時計と実時計のずれを取り、このブロックが
        # DACに出る実時刻を求める。
        dac_delta = time_info.outputBufferDacTime - time_info.currentTime
        block_start_frame = self._frame_counter
        block_wall = wall_now + dac_delta

        outdata.fill(0)
        with self._lock:
            self._anchor_frame = block_start_frame
            self._anchor_wall = block_wall

            still_pending = []
            for item in self._pending:
                if item["cursor"] == 0:
                    offset = item["start_frame"] - block_start_frame
                    if offset >= frames:
                        still_pending.append(item)
                        continue
                    offset = max(offset, 0)   # 予約が遅れた場合は即座に鳴らす
                    self.onsets.append((item["index"], block_wall + offset / self.samplerate))
                else:
                    offset = 0

                buf = item["data"]
                n = min(frames - offset, len(buf) - item["cursor"])
                outdata[offset:offset + n] += buf[item["cursor"]:item["cursor"] + n]
                item["cursor"] += n
                if item["cursor"] < len(buf):
                    still_pending.append(item)
            self._pending = still_pending
            gain = self.gain

        if gain != 1.0:
            outdata *= gain
        self._frame_counter += frames

    # --- control thread ---
    def frame_for_wall(self, wall_time):
        with self._lock:
            return int(self._anchor_frame + (wall_time - self._anchor_wall) * self.samplerate)

    def schedule(self, index, stim_key, start_frame):
        with self._lock:
            self._pending.append({
                "index": index,
                "data": self.buffers[stim_key],
                "cursor": 0,
                "start_frame": start_frame,
            })

    def clear_schedule(self):
        with self._lock:
            self._pending = [i for i in self._pending if i["cursor"] > 0]
            self.onsets.clear()

    def _load_wav(self, filename):
        """WAVをデバイスのレートに合わせたステレオfloat32にして返す（結果はキャッシュ）。"""
        if filename in self._wav_cache:
            return self._wav_cache[filename]

        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            print(f"[audio] missing file: {path}")
            self._wav_cache[filename] = None
            return None

        data, fs = sf.read(path, dtype="float32", always_2d=True)
        if fs != self.samplerate:
            g = gcd(int(fs), int(self.samplerate))
            data = resample_poly(data, self.samplerate // g, int(fs) // g, axis=0)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        data = np.ascontiguousarray(data, dtype=np.float32)
        self._wav_cache[filename] = data
        return data

    def play_file(self, filename, block=True):
        """アナウンス音・ビープを刺激と同じストリームに混ぜて再生する。

        ALSAの hw: デバイスは排他なので、別ストリーム(sd.play)は開けない。
        """
        data = self._load_wav(filename)
        if data is None:
            return
        start = self.frame_for_wall(time.time() + 0.05)
        with self._lock:
            self._pending.append({
                "index": ANNOUNCE_INDEX,
                "data": data,
                "cursor": 0,
                "start_frame": start,
            })
        if block:
            time.sleep(0.05 + len(data) / self.samplerate + 0.05)


# ==== EEG取得 ====
def set_latency_timer(value=1, retries=3, delay=0.2):
    """FTDIの latency_timer を下げる（AugmentedArms/ABMI_Utils.set_latency_timer と同じ）。

    既定の16 msだとドライバがその間データを溜めるため、250 Hzのサンプルが
    16 ms周期のバーストで届き、brainflowのタイムスタンプが階段状になる。
    """
    paths = glob.glob("/sys/bus/usb-serial/devices/ttyUSB*/latency_timer")
    if not paths:
        print("[ftdi] usb-serial device not found")
        return False

    all_ok = True
    for path in paths:
        try:
            subprocess.run(["sudo", "tee", path], input=f"{value}\n".encode(),
                           check=True, capture_output=True)
        except Exception as exc:      # noqa: BLE001
            print(f"[ftdi] write failed on {path}: {exc}")
            all_ok = False
            continue

        for _ in range(retries):
            try:
                with open(path) as f:
                    if f.read().strip() == str(value):
                        print(f"[ftdi] {path} = {value}")
                        break
            except Exception:      # noqa: BLE001
                pass
            time.sleep(delay)
        else:
            print(f"[ftdi] could not verify {path}")
            all_ok = False
    return all_ok


def find_cyton_port():
    for port in serial.tools.list_ports.comports():
        if port.manufacturer and "FTDI" in port.manufacturer:
            return port.device
    for port in serial.tools.list_ports.comports():
        if "ttyUSB" in port.device:
            return port.device
    return None


class Acquisition:
    """brainflowでCytonから読み続け、表示バッファとCSVへ流す。"""

    def __init__(self):
        self.board = None
        self.port = None
        self.connected = False
        self.error = ""

        self.buffer_size = WINDOW_SECONDS * SAMPLING_RATE
        self.display = np.zeros((CHANNEL_NUM, self.buffer_size), dtype=np.float32)
        self.raw = np.zeros((CHANNEL_NUM, self.buffer_size), dtype=np.float32)
        self.write_idx = 0
        self.estimated_rate = 0.0
        self._rate_hist = deque()       # (時刻, サンプル数) を5秒分ためて平均する

        # Web側の負荷を下げるため、送信用の間引き波形と生値統計はリーダースレッドで
        # 用意しておき、リクエスト時は完成品を返すだけにする。
        self.eeg_frame = np.zeros((CHANNEL_NUM, self.buffer_size // DISPLAY_DECIM),
                                  dtype=np.float32).tobytes()
        self.eeg_shape = (CHANNEL_NUM, self.buffer_size // DISPLAY_DECIM)
        self.stats = [{"rms": 0.0, "offset": 0.0, "railed": False} for _ in range(CHANNEL_NUM)]
        self._last_derived = 0.0
        self.display_paused = False   # 記録中は意図的に表示を止めていることを示す

        taps = firwin(NUMTAPS,
                      [FILTER_BAND[0] / (SAMPLING_RATE / 2), FILTER_BAND[1] / (SAMPLING_RATE / 2)],
                      pass_zero=False)
        self._taps = taps
        self._zi = [lfilter_zi(taps, 1.0) * 0 for _ in range(CHANNEL_NUM)]

        self._recording = False
        self._csv_file = None
        self._csv_writer = None
        self._rec_pkg = []          # 記録中のパケット番号（タイムベース復元用）
        self.last_pkg = None
        self._rec_lock = threading.Lock()

        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = None
        self._eeg_rows = None
        self._ts_row = None

    def connect(self):
        self.port = find_cyton_port()
        if self.port is None:
            self.error = "Cyton dongle (FTDI) not found"
            self.connected = False
            return False
        set_latency_timer(1)

        params = BrainFlowInputParams()
        params.serial_port = self.port
        self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        try:
            self.board.prepare_session()
            self.board.start_stream()
        except Exception as exc:      # noqa: BLE001 - surfaced to the browser
            self.error = f"{type(exc).__name__}: {exc}"
            self.connected = False
            return False

        board_id = self.board.get_board_id()
        self._eeg_rows = BoardShim.get_eeg_channels(board_id)[:CHANNEL_NUM]
        self._ts_row = BoardShim.get_timestamp_channel(board_id)
        self._pkg_row = BoardShim.get_package_num_channel(board_id)
        self.connected = True
        self.error = ""

        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return True

    def disconnect(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.board is not None:
            try:
                self.board.stop_stream()
                self.board.release_session()
            except Exception:
                pass
        self.connected = False

    def _reader(self):
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            time.sleep(READ_INTERVAL)
            try:
                data = self.board.get_board_data()
            except Exception as exc:      # noqa: BLE001
                self.error = f"read failed: {exc}"
                break
            if data.size == 0:
                continue

            eeg = data[self._eeg_rows, :]                  # µV
            stamps = data[self._ts_row, :]
            pkgs = data[self._pkg_row, :]
            n = eeg.shape[1]

            # 記録中は表示のための計算を一切やらず、CSV書き出しに専念する。
            # （チャンネルごとの lfilter とリングバッファ更新はここでの主な負荷）
            if not self._recording:
                # 表示用にだけフィルタを掛ける（CSVは生値）
                filtered = np.empty_like(eeg)
                for ch in range(CHANNEL_NUM):
                    filtered[ch], self._zi[ch] = lfilter(self._taps, 1.0, eeg[ch], zi=self._zi[ch])

                idx = (self.write_idx + np.arange(n)) % self.buffer_size
                self.display[:, idx] = filtered
                self.raw[:, idx] = eeg
                self.write_idx = int((self.write_idx + n) % self.buffer_size)

            with self._rec_lock:
                if self._recording and self._csv_writer is not None:
                    self._csv_writer.writerows(
                        [stamps[i], *eeg[:, i]] for i in range(n)
                    )
                    self._rec_pkg.extend(int(p) for p in pkgs)

            now = time.time()
            self._rate_hist.append((now, n))
            while len(self._rate_hist) > 2 and now - self._rate_hist[0][0] > 5.0:
                self._rate_hist.popleft()
            span = now - self._rate_hist[0][0]
            if span >= 1.0:
                # 最古チャンクのサンプルは span より前に到着しているので数に入れない。
                # 入れると 1/チャンク数 だけレートが過大に出る（20ms読みで約+0.4%）。
                self.estimated_rate = sum(c for _, c in list(self._rate_hist)[1:]) / span

            if not self._recording and now - self._last_derived >= DERIVED_INTERVAL:
                self._last_derived = now
                self._update_derived()

    def _update_derived(self):
        """表示用の間引き波形と生値統計をまとめて作り直す（リクエスト側は読むだけ）。"""
        i = self.write_idx
        disp = np.concatenate((self.display[:, i:], self.display[:, :i]), axis=1)[:, ::DISPLAY_DECIM]
        disp = disp - disp.mean(axis=1, keepdims=True)
        self.eeg_frame = np.ascontiguousarray(disp, dtype=np.float32).tobytes()
        self.eeg_shape = disp.shape

        raw = np.concatenate((self.raw[:, i:], self.raw[:, :i]), axis=1)
        mean = raw.mean(axis=1)
        centred = raw - mean[:, None]
        rms = np.sqrt((centred ** 2).mean(axis=1))
        peak = np.abs(raw).max(axis=1)
        self.stats = [
            {"rms": round(float(rms[ch]), 2),
             "offset": round(float(mean[ch]), 1),
             "railed": bool(peak[ch] > 150000.0 or rms[ch] < 1e-6)}
            for ch in range(CHANNEL_NUM)
        ]

    def start_recording(self, path):
        with self._rec_lock:
            self._csv_file = open(path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["Timestamp"] + [f"Ch{i}" for i in range(1, CHANNEL_NUM + 1)] + ["Label", "Seq"]
            )
            self._rec_pkg = []
            self._recording = True
            self.display_paused = True

    def stop_recording(self):
        with self._rec_lock:
            self._recording = False
            self.display_paused = False
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
            self.last_pkg = np.array(self._rec_pkg, dtype=np.int32)

    def pause_stream(self):
        """インピーダンス計測のためにストリームだけ止める（セッションは保持）。"""
        self._paused.set()
        time.sleep(0.1)
        self.board.stop_stream()

    def resume_stream(self):
        self.board.get_board_data()      # 溜まった分を捨てる
        self.board.start_stream()
        self._paused.clear()


# ==== 実験制御 ====
class Experiment:
    def __init__(self, acq, audio):
        self.acq = acq
        self.audio = audio
        self.lock = threading.Lock()
        self.running = False
        self.mode = "idle"              # idle / record / predict
        self.target = "1"
        self.progress = 0               # 提示済み刺激数
        self.total = 0
        self.message = ""
        self.last_prediction = None
        self.last_model = None
        self.last_file = None
        self._abort = threading.Event()

    # --- helpers ---
    def build_sequence(self, set_num):
        """main.py の play_stimuli と同じ順序生成（セット境界で同じ音が連続しない）。"""
        sequence, seq_ids, prev = [], [], None
        for set_idx in range(set_num):
            order = random.sample(STIM_KEYS, len(STIM_KEYS))
            while prev is not None and order[0] == prev:
                order = random.sample(STIM_KEYS, len(STIM_KEYS))
            prev = order[-1]
            sequence.extend(order)
            seq_ids.extend([set_idx + 1] * len(order))

        # set_num < 10 のときは末尾の1秒無音の代わりに1セット追加する（main.py と同じ）
        if set_num < 10:
            order = random.sample(STIM_KEYS, len(STIM_KEYS))
            while prev is not None and order[0] == prev:
                order = random.sample(STIM_KEYS, len(STIM_KEYS))
            sequence.extend(order)
            seq_ids.extend([set_num + 1] * len(order))
        return sequence, seq_ids

    def start(self, mode, set_num=SET_NUM):
        with self.lock:
            if self.running:
                return False, "already running"
            if not self.acq.connected:
                return False, "Cyton not connected"
            self.running = True
            self.mode = mode
            self._abort.clear()
        threading.Thread(target=self._run, args=(mode, set_num), daemon=True).start()
        return True, f"{mode} started"

    def abort(self):
        self._abort.set()
        return self.running

    def _run(self, mode, set_num):
        try:
            self.message = "ターゲットを再生中"
            self.last_prediction = None
            self.audio.play_file(TARGET_ANNOUNCE_WAV[self.target])
            time.sleep(1.0)

            sequence, seq_ids = self.build_sequence(set_num)
            self.total = len(sequence)
            self.progress = 0

            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(DATA_DIR, f"{stamp}_{self.target}.csv")
            self.last_file = path

            self.message = "ビープ"
            self.audio.play_file("beep.wav")
            time.sleep(1.0)

            self.acq.start_recording(path)
            self.message = "刺激提示中"

            # 全刺激をDACクロック上に等間隔で予約する
            self.audio.clear_schedule()
            step = int(round(SOA * self.audio.samplerate))
            base = self.audio.frame_for_wall(time.time() + 0.3)
            for i, key in enumerate(sequence):
                self.audio.schedule(i, key, base + i * step)

            onsets = [0.0] * len(sequence)
            got = 0
            deadline = time.time() + len(sequence) * SOA + 10
            while got < len(sequence) and time.time() < deadline:
                if self._abort.is_set():
                    self.audio.clear_schedule()
                    break
                try:
                    idx, wall = self.audio.onsets.popleft()
                except IndexError:
                    time.sleep(0.005)
                    continue
                if idx < 0:
                    continue        # 音確認ボタン・アナウンスの分は試行に含めない
                onsets[idx] = wall
                got += 1
                self.progress = got

            time.sleep(1.0)     # 最終刺激のエポック分を録り切る
            self.acq.stop_recording()

            if self._abort.is_set():
                self.message = "中止しました"
                return

            self.message = "ビープ"
            self.audio.play_file("beep.wav")

            self.message = "ラベル付与中"
            merge_labels(path, onsets[:got], sequence[:got], seq_ids[:got], self.acq.last_pkg)
            _counts_cache["value"] = None       # 本数表示をすぐ更新する

            if mode == "predict":
                name = active_model_name()
                self.message = f"予測中（{name}）"
                try:
                    result = get_predictor(name).predict(path)
                    label = int(np.asarray(result).ravel()[0])
                    self.last_prediction = label
                    self.last_model = name
                    self.message = f"予測: {TARGET_LABEL.get(str(label), label)}（{name}）"
                    # どのモデルが出した予測かを残さないと後から突き合わせられない
                    with open(os.path.splitext(path)[0] + "_predict.json", "w") as f:
                        json.dump({"model": name, "target": self.target,
                                   "prediction": label,
                                   "prediction_label": TARGET_LABEL.get(str(label))},
                                  f, indent=2, ensure_ascii=False)
                    wav = RESULT_ANNOUNCE_WAV.get(label)
                    if wav:
                        self.audio.play_file(wav)
                except Exception as exc:      # noqa: BLE001
                    self.message = f"予測に失敗: {type(exc).__name__}: {exc}"
            else:
                self.message = f"記録完了: {os.path.basename(path)}"
        except Exception as exc:      # noqa: BLE001
            self.message = f"エラー: {type(exc).__name__}: {exc}"
            self.acq.stop_recording()
        finally:
            with self.lock:
                self.running = False
                self.mode = "idle"


def merge_labels(path, onsets, sequence, seq_ids, pkgs=None):
    """刺激時刻に最も近い行にラベルを打つ（main.py の timestamp_merge と同じ）。

    Timestamp は brainflow が付けた測定値のまま触らない。タイミングの良し悪しは
    書き換えずに、隣の `_timing.json` に実測値として書き出す。
    """
    df = pd.read_csv(path)
    df["Label"] = None
    df["Seq"] = None
    for i, onset in enumerate(onsets):
        closest = (df["Timestamp"] - onset).abs().idxmin()
        df.at[closest, "Label"] = sequence[i]
        df.at[closest, "Seq"] = seq_ids[i]
    df.ffill(inplace=True)
    # main.py は infer_objects(copy=False) を呼び捨てにしているが、その引数は
    # pandas 2.0 以降のもので Pi の 1.5.3 では TypeError になる。戻り値を受けて
    # 引数なしで呼ぶ（結果のCSVは同一）。
    df = df.infer_objects()
    df.fillna(0, inplace=True)
    df.to_csv(path, index=False)

    write_timing_report(path, df, onsets, pkgs)


def write_timing_report(csv_path, df, onsets, pkgs):
    """記録のタイミング実測値を CSV の隣に書き出す（CSV自体は一切変更しない）。

    * sample_interval — brainflow のタイムスタンプ間隔。USBのバースト転送が
      残っていればここの sd が大きくなる。
    * audio_soa — サウンドカードのDACクロック上で測った刺激間隔。提示側の精度。
    * label_soa — ラベルが実際に載った行の時刻差。EEG側から見た刺激間隔で、
      audio_soa との差がそのままエポック切り出しの誤差になる。
    """
    def stats(x, unit=1000.0):
        x = np.asarray(x, dtype=float) * unit
        if x.size == 0:
            return None
        return {"mean": round(float(x.mean()), 4), "sd": round(float(x.std()), 4),
                "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}

    t = df["Timestamp"].values.astype(np.float64)
    lab, seq = df["Label"].values, df["Seq"].values
    change = [i for i in range(1, len(df)) if lab[i] != lab[i - 1] or seq[i] != seq[i - 1]]

    report = {
        "file": os.path.basename(csv_path),
        "samples": int(len(df)),
        "duration_s": round(float(t[-1] - t[0]), 4) if len(t) > 1 else 0.0,
        # n サンプルの間にある間隔は n-1 個。n/span にすると 1/n だけ高く出る。
        "effective_fs": round((len(df) - 1) / (t[-1] - t[0]), 4) if len(t) > 1 and t[-1] > t[0] else None,
        "sample_interval_ms": stats(np.diff(t)),
        "audio_soa_ms": stats(np.diff(onsets)) if len(onsets) > 1 else None,
        "label_soa_ms": stats(np.diff(t[change])) if len(change) > 1 else None,
        "labels_placed": len(change),
        "stimuli_played": len(onsets),
    }
    if pkgs is not None and len(pkgs) == len(df) and len(pkgs) > 1:
        delta = np.diff(np.asarray(pkgs, dtype=np.int64)) % 256
        report["dropped_samples"] = int(delta.sum() - (len(pkgs) - 1))

    out = os.path.splitext(csv_path)[0] + "_timing.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ==== インピーダンス ====
class ImpedanceRunner:
    def __init__(self, acq):
        self.acq = acq
        self.values = ["---"] * CHANNEL_NUM
        self.running = False
        self.current_ch = None
        self.lock = threading.Lock()

    def run(self, channels):
        with self.lock:
            if self.running:
                return False, "impedance already running"
            if not self.acq.connected:
                return False, "Cyton not connected"
            self.running = True
        threading.Thread(target=self._worker, args=(channels,), daemon=True).start()
        return True, "impedance started"

    def _worker(self, channels):
        from impedance_check import ImpedanceCheck
        checker = ImpedanceCheck(self.acq.port)
        checker.board = self.acq.board        # 既存セッションを共有する
        try:
            self.acq.pause_stream()
            for ch in channels:
                self.current_ch = ch
                try:
                    self.values[ch - 1] = f"{checker.check_impedance(ch):.1f}"
                except Exception as exc:      # noqa: BLE001
                    self.values[ch - 1] = f"ERR"
                    print(f"[imp] ch{ch}: {exc}")
        finally:
            self.current_ch = None
            try:
                checker.reset_to_defaults(self.acq.board)
            except Exception:
                pass
            self.acq.resume_stream()
            self.running = False


# ==== Web ====
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # モデルのアップロード上限
acq = Acquisition()
audio = None
mixer = None
experiment = None
impedance = None


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/state")
def api_state():
    return jsonify({
        "connected": acq.connected,
        "error": acq.error,
        "port": acq.port,
        "rate": round(acq.estimated_rate, 1),
        "audio_device": audio.device_name if audio else "",
        "audio_is_usb": bool(audio and audio.is_usb),
        "audio_expected": AUDIO_DEVICE_MATCH,
        "target": experiment.target,
        "target_label": TARGET_LABEL[experiment.target],
        "running": experiment.running,
        "mode": experiment.mode,
        "progress": experiment.progress,
        "total": experiment.total,
        "message": experiment.message,
        "prediction": experiment.last_prediction,
        "prediction_label": (TARGET_LABEL.get(str(experiment.last_prediction))
                             if experiment.last_prediction is not None else None),
        "last_file": os.path.basename(experiment.last_file) if experiment.last_file else None,
        "impedance": impedance.values,
        "impedance_running": impedance.running,
        "impedance_channel": impedance.current_ch,
        "signal": acq.stats if acq.connected else [],
        "display_paused": acq.display_paused,
        "counts": recording_counts(),
        "active_model": active_model_name(),
        "last_model": experiment.last_model,
        "stim_volume": round(audio.gain * 100),
        "device_volume": mixer.percent,
        "device_volume_available": mixer.available,
    })


@app.route("/api/volume/<kind>", methods=["POST"])
def api_volume(kind):
    try:
        value = max(0, min(100, int(request.args.get("value", ""))))
    except ValueError:
        return jsonify({"ok": False, "message": "value は 0-100 の整数"}), 400

    if kind == "stim":
        audio.gain = value / 100.0
        save_setting("stim_volume", value)
        return jsonify({"ok": True, "message": f"刺激音量: {value}%"})
    if kind == "device":
        if not mixer.available:
            return jsonify({"ok": False, "message": "ALSAの音量調整が使えません"}), 409
        ok = mixer.set_percent(value)
        if ok:
            save_setting("device_volume", value)
        return jsonify({"ok": ok,
                        "message": f"デバイス音量: {mixer.percent}%" if ok else "amixer 失敗"})
    return jsonify({"ok": False, "message": f"unknown kind {kind}"}), 400


@app.route("/api/eeg")
def api_eeg():
    """リーダースレッドが用意済みの間引き波形をそのまま返すだけ。"""
    return Response(acq.eeg_frame, mimetype="application/octet-stream",
                    headers={"X-Channels": str(acq.eeg_shape[0]),
                             "X-Samples": str(acq.eeg_shape[1])})


@app.route("/api/play/<key>", methods=["POST"])
def api_play(key):
    if key not in TARGET_KEYS:
        return jsonify({"ok": False, "message": f"unknown key {key}"}), 400
    if experiment.running:
        return jsonify({"ok": False, "message": "実験中は変更できません"}), 409
    experiment.target = key
    frame = audio.frame_for_wall(time.time() + 0.1)
    audio.schedule(PREVIEW_INDEX, key, frame)
    return jsonify({"ok": True, "message": f"ターゲット: {TARGET_LABEL[key]}"})


@app.route("/api/start/<mode>", methods=["POST"])
def api_start(mode):
    if mode not in ("record", "predict"):
        return jsonify({"ok": False, "message": f"unknown mode {mode}"}), 400
    try:
        set_num = int(request.args.get("set_num", SET_NUM))
    except ValueError:
        return jsonify({"ok": False, "message": "set_num は整数で指定してください"}), 400
    set_num = max(1, min(SET_NUM, set_num))
    ok, msg = experiment.start(mode, set_num)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/abort", methods=["POST"])
def api_abort():
    experiment.abort()
    return jsonify({"ok": True, "message": "中止を要求しました"})


@app.route("/api/impedance", methods=["POST"])
def api_impedance():
    if experiment.running:
        # 計測はストリームを一時停止するので、試行中に走らせると記録が途切れる
        return jsonify({"ok": False, "message": "実験中は計測できません"}), 409
    ch = request.args.get("ch")
    channels = list(range(1, CHANNEL_NUM + 1)) if ch in (None, "all") else [int(ch)]
    ok, msg = impedance.run(channels)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


# ==== 予測モデルの管理 ====
# models/<name>/ を1つのモデルセットとして扱う。
#   svm_model.pkl … 必須。学習済みバンドル。
#   model.py      … 任意。前処理ごと差し替えたいときだけ置く。無ければ同梱のものを使う。
#   note.txt      … 任意。UIに出す覚え書き。
# 実験ごとに使うモデルを変えたいので、選択は settings.json に持って再起動後も維持する。
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL = "default"
SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")

os.makedirs(MODELS_DIR, exist_ok=True)

_loaded_models = {}     # name -> モジュール（読み込み済み）
_bundle_info = {}       # pkl のパス -> (mtime, メタ情報)


def model_dir(name):
    if not name or not SAFE_MODEL_NAME.match(name):
        return None
    path = os.path.realpath(os.path.join(MODELS_DIR, name))
    if os.path.dirname(path) != os.path.realpath(MODELS_DIR):
        return None
    return path


def ensure_default_model():
    """同梱の svm_model.pkl を models/default/ に移して初期セットを作る。"""
    dst = os.path.join(MODELS_DIR, DEFAULT_MODEL)
    if os.path.exists(os.path.join(dst, "svm_model.pkl")):
        return
    legacy = os.path.join(BASE_DIR, "svm_model.pkl")
    if not os.path.exists(legacy):
        return
    os.makedirs(dst, exist_ok=True)
    import shutil
    shutil.copy2(legacy, os.path.join(dst, "svm_model.pkl"))
    with open(os.path.join(dst, "note.txt"), "w") as f:
        f.write("deploy.sh が配る同梱モデル\n")
    print(f"[model] created {DEFAULT_MODEL} from bundled svm_model.pkl")


def load_module_from(path, module_name):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_bundled_model = None


def bundled_model_module():
    """同梱の model.py を一度だけ読み込む（重いので遅延）。"""
    global _bundled_model
    if _bundled_model is None:
        _bundled_model = load_module_from(os.path.join(BASE_DIR, "model.py"), "bci_model_bundled")
    return _bundled_model


def alias_model_classes(mod):
    """モデルモジュールの自作クラスを `__main__` にも生やす。

    `python model.py` で学習した pkl は TrialCalibratedSVC などを
    `__main__.TrialCalibratedSVC` として参照する。サーバの `__main__` は
    bci_server なので、そのままでは AttributeError で復元できない。
    セットを切り替えるたびに呼び直して、そのセットの定義を指すようにする。
    """
    import sys
    main = sys.modules["__main__"]
    for attr, obj in vars(mod).items():
        if isinstance(obj, type) and getattr(obj, "__module__", "") == mod.__name__:
            setattr(main, attr, obj)


def bundle_info(pkl_path):
    """pkl から中身の要約を読む（毎回の一覧表示で読み直さないようキャッシュ）。"""
    try:
        mtime = os.path.getmtime(pkl_path)
    except OSError:
        return {}
    cached = _bundle_info.get(pkl_path)
    if cached and cached[0] == mtime:
        return cached[1]

    info = {}
    try:
        import joblib
        alias_model_classes(bundled_model_module())
        b = joblib.load(pkl_path)
        if isinstance(b, dict):
            acc = b.get("loocv_acc")
            info = {
                "preset": b.get("preset"),
                "labels": b.get("label_candidates"),
                "trained_on": os.path.basename(str(b.get("data_folder", "")).rstrip("/")),
                "loocv_acc": round(acc, 4) if isinstance(acc, (int, float)) else None,
            }
    except Exception as exc:      # noqa: BLE001
        info = {"error": f"{type(exc).__name__}: {exc}"}
    _bundle_info[pkl_path] = (mtime, info)
    return info


def get_predictor(name):
    """モデルセット name のモジュールを返す（predict(csv_path) を持つ）。"""
    if name in _loaded_models:
        mod = _loaded_models[name]
        alias_model_classes(mod)      # 別セットに奪われている可能性があるので毎回張り直す
        return mod

    d = model_dir(name)
    if d is None or not os.path.isdir(d):
        raise FileNotFoundError(f"モデルが見つかりません: {name}")
    pkl = os.path.join(d, "svm_model.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"svm_model.pkl がありません: {name}")

    # セット内に model.py があれば前処理ごと差し替える。無ければ同梱のものを使う。
    src = os.path.join(d, "model.py")
    own = os.path.exists(src)
    if not own:
        src = os.path.join(BASE_DIR, "model.py")

    mod = load_module_from(src, f"bci_model_{name}")
    # model.py はモジュール変数 model_path を見に行くので、セットの pkl を指させる
    mod.model_path = pkl
    alias_model_classes(mod)
    print(f"[model] loaded '{name}' (model.py: {'own' if own else 'bundled'})")

    _loaded_models[name] = mod
    return mod


def active_model_name():
    name = load_settings().get("active_model")
    if name and model_dir(name) and os.path.exists(
            os.path.join(MODELS_DIR, name, "svm_model.pkl")):
        return name
    return DEFAULT_MODEL


def list_models():
    rows = []
    for d in sorted(glob.glob(os.path.join(MODELS_DIR, "*"))):
        if not os.path.isdir(d):
            continue
        pkl = os.path.join(d, "svm_model.pkl")
        note = os.path.join(d, "note.txt")
        row = {
            "name": os.path.basename(d),
            "has_pkl": os.path.exists(pkl),
            "own_model_py": os.path.exists(os.path.join(d, "model.py")),
            "note": open(note).read().strip()[:200] if os.path.exists(note) else "",
            "loaded": os.path.basename(d) in _loaded_models,
        }
        if row["has_pkl"]:
            st = os.stat(pkl)
            row["size"] = st.st_size
            row["mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
            row.update(bundle_info(pkl))
        rows.append(row)
    return rows


# ==== 保存データの管理 ====
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*\.csv$")


def resolve_data_file(name):
    """data/ 直下の CSV だけを指すことを保証してフルパスを返す。

    ブラウザから来た名前をそのまま結合するとディレクトリ外へ抜けられるので、
    書式を絞ったうえで realpath が DATA_DIR 直下に収まることも確認する。
    """
    if not name or not SAFE_NAME.match(name):
        return None
    path = os.path.realpath(os.path.join(DATA_DIR, name))
    if os.path.dirname(path) != os.path.realpath(DATA_DIR):
        return None
    return path


def sidecar_paths(csv_path):
    """CSVに付随するファイル（タイミング記録・解析キャッシュ）を集める。"""
    stem = os.path.splitext(csv_path)[0]
    paths = [stem + "_timing.json"]
    # model.py は ./eeg_cache/<folder>_<stem>_lc..._hc....npy を作る。消し忘れると
    # 同じ名前を付け直したときに古い波形が再利用されてしまう。
    paths += glob.glob(os.path.join(BASE_DIR, "eeg_cache",
                                    f"*{os.path.basename(stem)}_*.npy"))
    return paths


@app.route("/api/files")
def api_files():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), reverse=True):
        stem = os.path.splitext(path)[0]
        key = os.path.basename(stem).rsplit("_", 1)[-1]
        st = os.stat(path)
        rows.append({
            "name": os.path.basename(path),
            "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "target": TARGET_LABEL.get(key, "—"),
            "timing": os.path.exists(stem + "_timing.json"),
        })
    return jsonify({"files": rows})


@app.route("/api/files/download")
def api_files_download():
    """data/ の中身をまとめてZIPで返す。

    CSV本体だけでなく `_timing.json` / `_predict.json` も入れる。CSVだけ持ち帰ると
    タイミングの実測値や使用モデルが分からなくなるため。
    記録中のファイルは書きかけなので除外する。
    """
    import tempfile
    import zipfile

    skip = set()
    if experiment.running and experiment.last_file:
        stem = os.path.splitext(os.path.basename(experiment.last_file))[0]
        skip = {stem + suffix for suffix in (".csv", "_timing.json", "_predict.json")}

    members = [p for p in sorted(glob.glob(os.path.join(DATA_DIR, "*")))
               if os.path.isfile(p) and os.path.basename(p) not in skip]
    if not members:
        return jsonify({"ok": False, "message": "ダウンロードできるファイルがありません"}), 404

    # メモリに載せると記録数が増えたときに苦しいので一時ファイルに書く
    tmp = tempfile.NamedTemporaryFile(prefix="bci_data_", suffix=".zip", delete=False)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in members:
                z.write(path, arcname=os.path.join("data", os.path.basename(path)))
    except Exception as exc:      # noqa: BLE001
        os.unlink(tmp.name)
        return jsonify({"ok": False, "message": f"ZIP作成に失敗: {exc}"}), 500

    @after_this_request
    def cleanup(response):
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return response

    name = f"bci_data_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    print(f"[files] zip {len(members)} files -> {name} "
          f"({os.path.getsize(tmp.name) / 1048576:.1f} MB)")
    return send_file(tmp.name, mimetype="application/zip",
                     as_attachment=True, download_name=name)


@app.route("/api/files/delete", methods=["POST"])
def api_files_delete():
    if experiment.running:
        return jsonify({"ok": False, "message": "実験中は削除できません"}), 409

    names = (request.get_json(silent=True) or {}).get("names") or []
    deleted, failed = [], []
    for name in names:
        path = resolve_data_file(name)
        if path is None or not os.path.exists(path):
            failed.append(name)
            continue
        try:
            for extra in sidecar_paths(path):
                if os.path.exists(extra):
                    os.remove(extra)
            os.remove(path)
            deleted.append(name)
        except OSError as exc:      # noqa: BLE001
            print(f"[files] delete {name}: {exc}")
            failed.append(name)

    _counts_cache["value"] = None
    msg = f"{len(deleted)}件を削除しました"
    if failed:
        msg += f"（失敗 {len(failed)}件: {', '.join(failed[:3])}）"
    return jsonify({"ok": not failed, "message": msg, "deleted": deleted})


@app.route("/api/files/rename", methods=["POST"])
def api_files_rename():
    if experiment.running:
        return jsonify({"ok": False, "message": "実験中はリネームできません"}), 409

    body = request.get_json(silent=True) or {}
    old = body.get("old", "")
    new = (body.get("new") or "").strip()
    if not new.lower().endswith(".csv"):
        new += ".csv"

    src, dst = resolve_data_file(old), resolve_data_file(new)
    if src is None or not os.path.exists(src):
        return jsonify({"ok": False, "message": f"見つかりません: {old}"}), 400
    if dst is None:
        return jsonify({"ok": False, "message": "使えない名前です（英数字と _ - . のみ）"}), 400
    if os.path.exists(dst):
        return jsonify({"ok": False, "message": f"すでに存在します: {new}"}), 409

    try:
        # 付随ファイルも同じ幹の名前に付け替える。キャッシュは名前が変わると
        # 対応が取れなくなるので消す（次回アクセス時に作り直される）。
        old_stem, new_stem = os.path.splitext(src)[0], os.path.splitext(dst)[0]
        os.rename(src, dst)
        if os.path.exists(old_stem + "_timing.json"):
            os.rename(old_stem + "_timing.json", new_stem + "_timing.json")
        for cache in glob.glob(os.path.join(BASE_DIR, "eeg_cache",
                                            f"*{os.path.basename(old_stem)}_*.npy")):
            os.remove(cache)
    except OSError as exc:      # noqa: BLE001
        return jsonify({"ok": False, "message": f"リネーム失敗: {exc}"}), 500

    _counts_cache["value"] = None
    if experiment.last_file and os.path.basename(experiment.last_file) == old:
        experiment.last_file = dst
    return jsonify({"ok": True, "message": f"{old} → {os.path.basename(dst)}"})


@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models(), "active": active_model_name()})


@app.route("/api/models/select", methods=["POST"])
def api_models_select():
    if experiment.running:
        return jsonify({"ok": False, "message": "実験中は切り替えられません"}), 409
    name = request.args.get("name", "")
    d = model_dir(name)
    if d is None or not os.path.exists(os.path.join(d, "svm_model.pkl")):
        return jsonify({"ok": False, "message": f"使えないモデルです: {name}"}), 400
    try:
        get_predictor(name)      # 選んだ時点で読み込んで、壊れていればここで判る
    except Exception as exc:      # noqa: BLE001
        _loaded_models.pop(name, None)
        return jsonify({"ok": False, "message": f"読み込み失敗: {type(exc).__name__}: {exc}"}), 400
    save_setting("active_model", name)
    return jsonify({"ok": True, "message": f"モデル: {name}"})


@app.route("/api/models/upload", methods=["POST"])
def api_models_upload():
    if experiment.running:
        return jsonify({"ok": False, "message": "実験中は変更できません"}), 409

    name = (request.form.get("name") or "").strip()
    d = model_dir(name)
    if d is None:
        return jsonify({"ok": False, "message": "使えない名前です（英数字と _ - . のみ）"}), 400

    saved = []
    os.makedirs(d, exist_ok=True)
    for field, filename in (("pkl", "svm_model.pkl"), ("py", "model.py")):
        f = request.files.get(field)
        if f is None or not f.filename:
            continue
        f.save(os.path.join(d, filename))
        saved.append(filename)

    note = (request.form.get("note") or "").strip()
    if note:
        with open(os.path.join(d, "note.txt"), "w") as fh:
            fh.write(note + "\n")

    if not saved and not note:
        return jsonify({"ok": False, "message": "ファイルが選ばれていません"}), 400

    # 差し替えたら読み込み済みモジュールとメタ情報を捨てて、次回に読み直させる
    _loaded_models.pop(name, None)
    _bundle_info.pop(os.path.join(d, "svm_model.pkl"), None)
    return jsonify({"ok": True, "message": f"{name}: {', '.join(saved) or 'メモのみ'} を保存"})


@app.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    if experiment.running:
        return jsonify({"ok": False, "message": "実験中は削除できません"}), 409
    name = request.args.get("name", "")
    d = model_dir(name)
    if d is None or not os.path.isdir(d):
        return jsonify({"ok": False, "message": f"見つかりません: {name}"}), 400
    if name == active_model_name():
        return jsonify({"ok": False, "message": "使用中のモデルは削除できません"}), 409
    import shutil
    try:
        shutil.rmtree(d)
    except OSError as exc:      # noqa: BLE001
        return jsonify({"ok": False, "message": f"削除失敗: {exc}"}), 500
    _loaded_models.pop(name, None)
    return jsonify({"ok": True, "message": f"{name} を削除しました"})


@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    """USB機器（オーディオI/F・Cytonドングル）を検出し直す。

    オーディオは起動時にしか見ていなかったので、挿し直してもサービスを再起動
    するまで戻らなかった。ここで両方まとめて拾い直す。
    """
    global mixer
    if experiment.running or impedance.running:
        return jsonify({"ok": False, "message": "実行中です"}), 409

    parts = []
    try:
        audio.reopen()
        # 内蔵ジャックへフォールバックしている間はALSAつまみを出さない。
        # 名前に hw:N があると拾えてしまうが、それは「USB機器の音量」ではない。
        mixer = AlsaMixer(audio.device_name if audio.is_usb else "")
        settings = load_settings()
        if "stim_volume" in settings:
            audio.gain = settings["stim_volume"] / 100.0
        if "device_volume" in settings and mixer.available:
            mixer.set_percent(settings["device_volume"])
        parts.append("音声: " + (audio.device_name if audio.is_usb
                                else f"{audio.device_name}（USB未検出）"))
    except Exception as exc:      # noqa: BLE001
        parts.append(f"音声の再取得に失敗: {type(exc).__name__}: {exc}")

    acq.disconnect()
    eeg_ok = acq.connect()
    parts.append("EEG: " + (acq.port if eeg_ok else acq.error))

    return jsonify({"ok": eeg_ok and audio.is_usb, "message": " / ".join(parts)})


def main():
    global audio, mixer, experiment, impedance
    ensure_default_model()
    audio = AudioEngine()
    print(f"[audio] device={audio.device_name} sr={audio.samplerate}")
    if not audio.is_usb:
        print(f"[audio] WARNING: '{AUDIO_DEVICE_MATCH}' not found — falling back to the "
              f"built-in output. Stimuli will not come out of the USB interface.")

    mixer = AlsaMixer(audio.device_name if audio.is_usb else "")
    if mixer.available:
        print(f"[audio] mixer card={mixer.card} control='{mixer.control}' "
              f"range={mixer.vmin}-{mixer.vmax} now={mixer.percent}%")
    else:
        print("[audio] ALSA volume control not available")

    # 前回の音量を復元する（ALSA側は再起動で既定値へ戻るため）
    settings = load_settings()
    if "stim_volume" in settings:
        audio.gain = settings["stim_volume"] / 100.0
        print(f"[audio] restored stim volume = {settings['stim_volume']}%")
    if "device_volume" in settings and mixer.available:
        if mixer.set_percent(settings["device_volume"]):
            print(f"[audio] restored device volume = {mixer.percent}%")

    experiment = Experiment(acq, audio)
    impedance = ImpedanceRunner(acq)

    BoardShim.disable_board_logger()
    if acq.connect():
        print(f"[eeg] connected on {acq.port}")
    else:
        print(f"[eeg] not connected: {acq.error}")

    # werkzeug は既定でリクエストを1件ずつログに書く。ブラウザが毎秒何度も
    # ポーリングするのでそのぶん取得スレッドのCPUを奪う。警告以上だけ残す。
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    for port in (WEB_PORT, WEB_PORT_FALLBACK):
        try:
            suffix = "" if port == 80 else f":{port}"
            print(f"[web] http://raspberrypi.local{suffix}/")
            app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)
            return
        except PermissionError:
            print(f"[web] port {port} not permitted, falling back")
        except OSError as exc:
            print(f"[web] port {port} unavailable ({exc}), falling back")


if __name__ == "__main__":
    main()
