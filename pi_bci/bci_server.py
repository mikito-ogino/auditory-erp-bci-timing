#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless auditory-BCI acquisition server for the Raspberry Pi / Windows.

Same experiment as main.py (8ch EEG, four tones at 380/524/708/932 Hz, 0.15 s each
with a 0.35 s ISI), but with no Qt GUI: the machine runs headless and everything is
driven from a browser over the direct LAN link.

Two amplifiers are supported and switchable from the browser (see `devices.py`):
  * OpenBCI Cyton — 250 Hz, 8ch, auto-detected FTDI dongle.
  * Polymate Pocket MP208 — 1000 Hz, 8ch + EXT 2ch (the stimulus audio itself is
    recorded alongside the EEG so onsets can be read off the acoustic waveform).
    Its API is a Windows-only extension module, so on the Pi selecting it just
    reports why it cannot be loaded.

Recordings and models are per subject: `data/<subject>/` and `models/<subject>/`.
The model used for prediction is always the current subject's.

Differences from the Mac main.py that are deliberate:
  * Acquisition goes through brainflow / the Polymate API instead of pyOpenBCI, so
    the sample timestamps come from the driver rather than from a sleep loop.
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
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from math import gcd

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
from flask import (Flask, Response, after_this_request, jsonify, request,
                   send_file, send_from_directory)
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly

# brainflow は Cyton だけが使う。Polymate 専用の Windows 機には入っていないことが
# あるので、無ければ無いまま起動する（Cyton を選ぶと devices.py が理由を返す）。
try:
    from brainflow.board_shim import BoardShim
except Exception as _exc:      # noqa: BLE001
    BoardShim = None
    print(f"[eeg] brainflow unavailable: {_exc}")

import devices
import erp_view
import neutsuri

# ==== 設定 ====
CHANNEL_NUM = 8
# サンプリングレートはアンプで決まる（Cyton 250 / Polymate 1000）。ここの値は
# 未接続のときの既定で、接続すると Acquisition が実機の値に張り替える。
DEFAULT_DEVICE = devices.CYTON
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
# 既定は IPv4 のみ。ブラウザが WiFi 側の IPv6 を引いても IPv4 に落ちてくれるので、
# 直結ケーブル運用ではこれが都合が良い。ただし mDNS が IPv6 しか広告しない機体では
# ホスト名で開けなくなるので、`BCI_WEB_HOST=::` で両対応にできるようにしておく。
WEB_HOST = os.environ.get("BCI_WEB_HOST", "0.0.0.0")
DISPLAY_RATE = 50           # 表示に送るレート [Hz]（間引き率はレートから決める）
DERIVED_INTERVAL = 0.2      # 表示用データを作り直す間隔 [s]
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


# ==== 被験者 ====
# 記録データ・予測モデルは被験者に紐づける。
#   data/<被験者>/   … その被験者の記録（CSV と付随ファイル）
#   models/<被験者>/ … その被験者のモデル（svm_model.pkl ほか）
# 「使用中のモデル」は常に「選択中の被験者のモデル」。別々に選べるようにすると
# 食い違ったまま実験してしまうので、1対1に固定する。
SAFE_SUBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


def known_subjects():
    subs = load_settings().get("subjects") or []
    return [x for x in subs if SAFE_SUBJECT_NAME.match(x)]


def current_subject():
    name = load_settings().get("current_subject")
    return name if name and SAFE_SUBJECT_NAME.match(name) else None


def subject_data_dir(name=None):
    """data/<被験者>/ を返す（無ければ作る）。被験者未選択なら None。"""
    name = name or current_subject()
    if not name or not SAFE_SUBJECT_NAME.match(name):
        return None
    path = os.path.realpath(os.path.join(DATA_DIR, name))
    if os.path.dirname(path) != os.path.realpath(DATA_DIR):
        return None
    os.makedirs(path, exist_ok=True)
    return path


def set_current_subject(name):
    """被験者を切り替える。音量もその被験者の設定に戻す。"""
    if not SAFE_SUBJECT_NAME.match(name or ""):
        raise ValueError("使えない被験者名です（英数字・_-. のみ、64文字以内）")
    subs = known_subjects()
    if name not in subs:
        subs.append(name)
        save_setting("subjects", subs)
    save_setting("current_subject", name)
    d = subject_data_dir(name)
    os.makedirs(os.path.join(MODELS_DIR, name), exist_ok=True)
    # Ne-utsuri の記録も被験者フォルダへ書き、押下判定モデルも被験者のものに読み替える
    if neu is not None and d:
        neu.data_dir = d
        neu.set_model_dir(os.path.join(MODELS_DIR, name))
    apply_subject_volumes(name)
    _counts_cache["value"] = None
    return name


# ==== 音量（被験者ごとに保持する） ====
# 提示レベルは実験条件そのものなので、被験者ごとに最後の設定を覚えて、
# 被験者を選び直したときと再起動後に必ず復元する。ALSAの音量は再起動で
# 既定値に戻ってしまうため、システム任せにせずここで持つ。
def subject_volumes(name=None):
    """その被験者の {"stim": %, "device": %}（未設定のキーは入らない）。"""
    name = name or current_subject()
    settings = load_settings()
    vols = (settings.get("subject_volumes") or {}).get(name) or {}
    out = {}
    for key in ("stim", "device"):
        if isinstance(vols.get(key), (int, float)):
            out[key] = int(vols[key])
    # 被験者ごとの記録が無ければ、以前の全体設定を初期値として引き継ぐ
    out.setdefault("stim", settings.get("stim_volume", 100))
    if "device" not in out and isinstance(settings.get("device_volume"), (int, float)):
        out["device"] = int(settings["device_volume"])
    return out


def save_subject_volume(kind, value, name=None):
    name = name or current_subject()
    if not name:
        return
    data = load_settings()
    vols = data.get("subject_volumes") or {}
    entry = dict(vols.get(name) or {})
    entry[kind] = int(value)
    vols[name] = entry
    save_setting("subject_volumes", vols)


def apply_subject_volumes(name=None):
    """被験者の音量設定を実際のオーディオへ反映する。"""
    vols = subject_volumes(name)
    if audio is not None and "stim" in vols:
        audio.gain = max(0, min(100, vols["stim"])) / 100.0
    if mixer is not None and mixer.available and "device" in vols:
        mixer.set_percent(vols["device"])
    return vols


# ==== 頭部計測値（被験者ごとに保持する） ====
# 10-20法の電極位置は頭の実測値で決まる。頭のかたちは被験者ごとに違うので、
# 画面に入力された値をその被験者に紐づけて覚え、選び直したときと再起動後に復元する。
HEAD_KEYS = ("ni", "lr")


def subject_head(name=None):
    """その被験者の頭部計測値 {"ni": cm, "lr": cm}（未設定のキーは入らない）。"""
    name = name or current_subject()
    settings = load_settings()
    saved = (settings.get("subject_head") or {}).get(name) or {}
    fallback = settings.get("head") or {}      # 被験者ごとの記録が無いときの初期値
    out = {}
    for key in HEAD_KEYS:
        for src in (saved, fallback):
            v = src.get(key)
            if isinstance(v, (int, float)) and 0 < v <= 100:
                out[key] = float(v)
                break
    return out


def save_subject_head(values, name=None):
    """入力された頭部計測値を被験者に紐づけて保存する。"""
    clean = {}
    for key in HEAD_KEYS:
        v = values.get(key)
        if isinstance(v, (int, float)) and 0 < v <= 100:
            clean[key] = round(float(v), 1)
    if not clean:
        return {}
    # 全体設定にも残しておく（被験者未選択・新規被験者のときの初期値になる）
    save_setting("head", {**(load_settings().get("head") or {}), **clean})
    name = name or current_subject()
    if name:
        heads = load_settings().get("subject_head") or {}
        heads[name] = {**(heads.get(name) or {}), **clean}
        save_setting("subject_head", heads)
    return subject_head(name)


# ==== 記録済みファイル数 ====
_counts_cache = {"at": 0.0, "value": None}


def recording_counts(ttl=2.0):
    """選択中の被験者のフォルダにターゲットごとに何本記録済みかを数える。

    ファイル名は `YYYYmmdd_HHMMSS_<target>.csv` なので末尾のキーで振り分ける。
    /api/state は毎秒何度も叩かれるので短いTTLでキャッシュする。
    """
    now = time.time()
    if _counts_cache["value"] is not None and now - _counts_cache["at"] < ttl:
        return _counts_cache["value"]

    counts = {key: 0 for key in TARGET_KEYS}
    d = subject_data_dir()
    if d:
        for path in glob.glob(os.path.join(d, "*.csv")):
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


def alsa_cards():
    """ALSAが見ているサウンドカードを {番号: {"id","name","usb"}} で返す。

    USBかどうかは製品名ではなく `/proc/asound/cardN/usbid` の有無で決める。
    USBアダプタの名前は製品ごとにばらばら（"USB Audio Device" とは限らない）で、
    名前一致だけだと本物のUSB出力を取りこぼす。
    """
    cards = {}
    try:
        with open("/proc/asound/cards") as f:
            text = f.read()
    except OSError:
        return cards
    for num, cid, name in re.findall(r"^\s*(\d+)\s+\[([^\]]+)\]:\s*\S+\s+-\s+(.*)$",
                                     text, re.M):
        idx = int(num)
        cards[idx] = {"id": cid.strip(), "name": name.strip(),
                      "usb": os.path.exists(f"/proc/asound/card{idx}/usbid")}
    return cards


def default_sink_card():
    """PulseAudio / PipeWire の既定シンクが実際に鳴らしているカード番号。

    PipeWire がカードを掴んでいるPiでは PortAudio から `hw:` を開けず、
    出力チャンネル数も 0 と報告される。そこでは `default`（→ pulse）経由でしか
    鳴らせないので、その先がどのカードなのかをここで解決する。
    """
    try:
        info = subprocess.run(["pactl", "info"], capture_output=True,
                              text=True, timeout=5).stdout
        sinks = subprocess.run(["pactl", "list", "sinks"], capture_output=True,
                               text=True, timeout=5).stdout
    except Exception:      # noqa: BLE001 - pactl が無い環境では単に諦める
        return None

    m = re.search(r"Default Sink:\s*(\S+)", info)
    if not m:
        return None
    for block in sinks.split("Sink #"):
        if re.search(r"^\s*Name:\s*" + re.escape(m.group(1)) + r"\s*$", block, re.M):
            card = re.search(r'alsa\.card\s*=\s*"(\d+)"', block)
            return int(card.group(1)) if card else None
    return None


def find_output_device():
    """刺激を出すデバイスを選ぶ。戻り値は (index, info, is_usb, alsa_card)。

    1. PortAudio から直接開けるUSBカード（`hw:`）を最優先で掴む。排他で開けるので
       いちばん素直にDACクロックへ乗る。
    2. PipeWire/PulseAudio がカードを掴んでいると `hw:` は出力0chと報告され、
       選びようがない。その場合は既定デバイス（`default` → pulse）で鳴らすが、
       その先がUSBカードなら「USBオーディオで鳴っている」のは事実なので
       is_usb は True にする。ここを名前一致だけで判定していたころは、
       音は出ているのにUIが「USB未接続」と言い続けていた。
    3. どちらでもなければ内蔵出力へのフォールバック。提示レベルも遅延も普段と
       変わってしまうので、その旨をUIまで持っていく。
    """
    cards = alsa_cards()
    usb = {i for i, c in cards.items() if c["usb"]}

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] < 2:
            continue
        m = re.search(r"hw:(\d+)", dev["name"])
        card = int(m.group(1)) if m else None
        if card in usb or AUDIO_DEVICE_MATCH.lower() in dev["name"].lower():
            return idx, dev, True, card

    idx = sd.default.device[1]
    card = default_sink_card()
    if card is None and len(cards) == 1:
        # pactl が無くてもカードが1枚しかなければ、既定の行き先はそれしかない
        card = next(iter(cards))
    return idx, sd.query_devices(idx), card in usb, card


class AlsaMixer:
    """USBオーディオI/Fのハードウェア音量を amixer 経由で操作する。

    対象はカード番号で受け取る。pulse 経由で鳴らしている場合はデバイス名に
    "hw:N" が出てこないが、カードのミキサ自体はそのまま効くので、
    音量つまみを諦める理由にはならない。pvolume を持つ最初の simple control
    （C-Media系なら 'Speaker'）を対象にする。
    """

    def __init__(self, card):
        self.card = None
        self.control = None
        self.vmin = 0
        self.vmax = 0
        self.percent = None

        if isinstance(card, str):        # 旧来の呼び出し（デバイス名）にも耐える
            m = re.search(r"hw:(\d+)", card)
            card = int(m.group(1)) if m else None
        if card is None:
            return
        self.card = int(card)

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
        self.device, info, self.is_usb, self.card = find_output_device()
        try:
            self._start_stream(info)
        except Exception as exc:      # noqa: BLE001
            print(f"[audio] cannot open '{info['name']}': {exc} — using system default")
            self.device = sd.default.device[1]
            self.is_usb = False
            self.card = None
            self._start_stream(sd.query_devices(self.device))

        cards = alsa_cards()
        self.card_name = cards[self.card]["name"] if self.card in cards else None

    @property
    def route(self):
        """UIに出す出力経路。default 経由のときは行き先のカード名も添える。"""
        if self.card_name and self.card_name.lower() not in self.device_name.lower():
            return f"{self.card_name}（{self.device_name} 経由）"
        return self.device_name

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
class Acquisition:
    """選ばれたアンプ（devices.py）から読み続け、表示バッファとCSVへ流す。

    サンプリングレートはアンプごとに違う（Cyton 250 / Polymate 1000）。バッファ長・
    表示の間引き率・表示フィルタはすべてレートから決めるので、機種を切り替えたら
    `set_device()` で作り直す。
    """

    def __init__(self, device_name=DEFAULT_DEVICE):
        self.backends = devices.build_backends(
            CHANNEL_NUM, BASE_DIR,
            exclude_vid_pid=neutsuri.ESP32_USB_IDS)
        self.device_name = device_name if device_name in self.backends else DEFAULT_DEVICE
        self.backend = self.backends[self.device_name]

        self.port = None
        self.connected = False
        self.error = ""
        self.estimated_rate = 0.0

        self._recording = False
        self._csv_file = None
        self._csv_writer = None
        self._rec_pkg = []          # 記録中のパケット番号（取りこぼし検出用）
        self.last_pkg = None
        self._rec_lock = threading.Lock()

        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = None
        self.display_paused = False   # 記録中は意図的に表示を止めていることを示す

        # Ne-utsuri モード（別モードなので main() で差し込む）。聴覚BCIの試行中は
        # タイミングを守るため一切呼ばない。250 Hz 前提のモデルなので Cyton 専用。
        self.neutsuri = None
        self._neutsuri_engine = None

        self._configure()

    # --- レートに合わせたバッファ作り ---
    def _configure(self):
        """アンプのレートに合わせて表示バッファ・フィルタ・間引き率を作り直す。"""
        self.fs = int(self.backend.sampling_rate)
        self.n_ext = int(self.backend.n_ext)
        self.ext_names = list(self.backend.ext_names)
        # 表示は機種にかかわらず DISPLAY_RATE で送る。1000 Hz なら 20 間引き。
        self.decim = max(1, int(round(self.fs / DISPLAY_RATE)))
        self.buffer_size = WINDOW_SECONDS * self.fs

        self.display = np.zeros((CHANNEL_NUM, self.buffer_size), dtype=np.float32)
        self.raw = np.zeros((CHANNEL_NUM, self.buffer_size), dtype=np.float32)
        self.write_idx = 0
        self._rate_hist = deque()       # (時刻, サンプル数) を5秒分ためて平均する

        # Web側の負荷を下げるため、送信用の間引き波形と生値統計はリーダースレッドで
        # 用意しておき、リクエスト時は完成品を返すだけにする。
        disp_len = self.buffer_size // self.decim
        self.eeg_frame = np.zeros((CHANNEL_NUM, disp_len), dtype=np.float32).tobytes()
        self.eeg_shape = (CHANNEL_NUM, disp_len)
        self.stats = [{"rms": 0.0, "offset": 0.0, "railed": False}
                      for _ in range(CHANNEL_NUM)]
        self._last_derived = 0.0

        taps = firwin(NUMTAPS,
                      [FILTER_BAND[0] / (self.fs / 2), FILTER_BAND[1] / (self.fs / 2)],
                      pass_zero=False)
        self._taps = taps
        self._zi = [lfilter_zi(taps, 1.0) * 0 for _ in range(CHANNEL_NUM)]

        self._attach_neutsuri()

    def _attach_neutsuri(self):
        """Ne-utsuri を今のアンプに合わせて繋ぎ直す（対応外なら外す）。

        押下判定モデルは 250 Hz・2秒窓で学習されているので、レートが違うアンプでは
        使えない。表示リングの長さも変わるため、繋いだままにすると書き込みが範囲外に
        出てしまう。対応外の機種では必ず外し、UIには「この機種では使えない」と出す。
        """
        engine = self._neutsuri_engine
        if engine is None:
            return
        if self.backend.supports_neutsuri and self.fs == neutsuri.SAMPLING_RATE:
            engine.resize(self.buffer_size, self.decim)
            self.neutsuri = engine
        else:
            if engine.enabled:
                engine.set_enabled(False)
            self.neutsuri = None

    def set_neutsuri_engine(self, engine):
        self._neutsuri_engine = engine
        self._attach_neutsuri()

    @property
    def neutsuri_supported(self):
        return bool(self.backend.supports_neutsuri
                    and self.fs == neutsuri.SAMPLING_RATE)

    # --- 機種の切り替え ---
    def device_options(self):
        """UIに出す機種一覧（使えない機種は理由つきで並べる）。"""
        rows = []
        for name, backend in self.backends.items():
            ok, reason = backend.availability()
            rows.append({
                "name": name,
                "label": backend.label,
                "sampling_rate": backend.sampling_rate,
                "n_ext": backend.n_ext,
                "needs_port": backend.needs_port,
                "available": ok,
                "reason": reason,
            })
        return rows

    def set_device(self, name):
        if name not in self.backends:
            return False, f"未対応の機種です: {name}"
        if self.connected:
            return False, "接続中は機種を変えられません（先に切断してください）"
        self.device_name = name
        self.backend = self.backends[name]
        self.port = None
        self.error = ""
        self.estimated_rate = 0.0
        self._configure()
        save_setting("device", name)
        return True, f"{self.backend.label}（{self.fs} Hz）に切り替えました"

    # --- 接続 ---
    def connect(self, port=None):
        """アンプを開いて読み出しスレッドを起こす。port 未指定なら自動で探す。"""
        ok, reason = self.backend.availability()
        if not ok:
            self.error = reason
            self.connected = False
            return False

        try:
            self.port = self.backend.open(port or None)
        except devices.DeviceError as exc:
            self.error = str(exc)
            self.connected = False
            return False
        except Exception as exc:      # noqa: BLE001 - そのままUIに出す
            self.error = f"{type(exc).__name__}: {exc}"
            self.connected = False
            return False

        # 実機のレートが宣言と違うことは無いが、バッファは接続後の値で揃える
        if int(self.backend.sampling_rate) != self.fs:
            self._configure()

        self.connected = True
        self.error = ""
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return True

    def disconnect(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self.backend.close()
        self.connected = False

    # --- 読み出し ---
    def _reader(self):
        interval = self.backend.read_interval
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            time.sleep(interval)
            try:
                chunk = self.backend.read()
            except Exception as exc:      # noqa: BLE001
                self.error = f"read failed: {exc}"
                break
            if chunk is None:
                continue

            eeg, stamps, pkgs, ext = chunk        # eeg: (ch, n) µV
            n = eeg.shape[1]

            # 記録中は表示のための計算を一切やらず、CSV書き出しに専念する。
            # （チャンネルごとの lfilter とリングバッファ更新はここでの主な負荷）
            if not self._recording:
                # 表示用にだけフィルタを掛ける（CSVは生値）
                filtered = np.empty_like(eeg)
                for ch in range(CHANNEL_NUM):
                    filtered[ch], self._zi[ch] = lfilter(self._taps, 1.0, eeg[ch],
                                                         zi=self._zi[ch])

                idx = (self.write_idx + np.arange(n)) % self.buffer_size
                self.display[:, idx] = filtered
                self.raw[:, idx] = eeg
                # Ne-utsuri は表示リングと同じ位置に押下・確率を書くので、ブラウザ上で
                # EEG波形とそのまま重ねて読める。CSV書き出しもここから行う。
                if self.neutsuri is not None:
                    self.neutsuri.on_chunk(idx, stamps, eeg)
                self.write_idx = int((self.write_idx + n) % self.buffer_size)

            with self._rec_lock:
                if self._recording and self._csv_writer is not None:
                    if ext is None:
                        self._csv_writer.writerows(
                            [stamps[i], *eeg[:, i]] for i in range(n))
                    else:
                        # EXT には刺激音そのものが入る。オンセットを音響波形から
                        # 取れるようにするため EEG と同じ行に並べて残す。
                        self._csv_writer.writerows(
                            [stamps[i], *eeg[:, i], *ext[i]] for i in range(n))
                    if pkgs is not None:
                        self._rec_pkg.extend(int(v) for v in pkgs)

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
        disp = np.concatenate((self.display[:, i:], self.display[:, :i]),
                              axis=1)[:, ::self.decim]
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

        if self.neutsuri is not None:
            self.neutsuri.build_frame(self.write_idx)

    # --- 記録 ---
    def csv_header(self):
        return (["Timestamp"] + [f"Ch{i}" for i in range(1, CHANNEL_NUM + 1)]
                + self.ext_names + ["Label", "Seq"])

    def start_recording(self, path):
        with self._rec_lock:
            self._csv_file = open(path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self.csv_header())
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
            # パケット番号を出さないアンプ（Polymate）では取りこぼし数を出せない
            self.last_pkg = (np.array(self._rec_pkg, dtype=np.int32)
                             if self._rec_pkg else None)

    # --- インピーダンス計測のための一時停止 ---
    def pause_stream(self):
        """インピーダンス計測のためにストリームだけ止める（セッションは保持）。"""
        self._paused.set()
        time.sleep(0.1)
        self.backend.pause()

    def resume_stream(self):
        self.backend.resume()
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
                return False, f"{self.acq.backend.label} が未接続です"
            subject = current_subject()
            if subject is None:
                return False, "被験者が選択されていません"
            # 2つのモードで同じアンプを取り合わないよう、記録は必ず片方だけにする
            if neu is not None and neu.recording:
                return False, "Ne-utsuri の記録中です（先に停止してください）"
            self.running = True
            self.mode = mode
            self._abort.clear()
        threading.Thread(target=self._run, args=(mode, set_num, subject),
                         daemon=True).start()
        return True, f"{mode} started"

    def abort(self):
        self._abort.set()
        return self.running

    def _run(self, mode, set_num, subject):
        try:
            self.message = "ターゲットを再生中"
            self.last_prediction = None
            self.audio.play_file(TARGET_ANNOUNCE_WAV[self.target])
            time.sleep(1.0)

            sequence, seq_ids = self.build_sequence(set_num)
            self.total = len(sequence)
            self.progress = 0

            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(subject_data_dir(subject), f"{stamp}_{self.target}.csv")
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
                name = subject
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
    """計測の手順はアンプごとに違うので devices.py 側の実装に投げる。

    Cyton は1chずつ（進捗を出せる）、Polymate は全ch一括で返る。
    """

    def __init__(self, acq):
        self.acq = acq
        self.values = ["---"] * CHANNEL_NUM
        self.running = False
        self.current_ch = None
        self.error = ""
        self.lock = threading.Lock()

    def run(self, channels):
        with self.lock:
            if self.running:
                return False, "impedance already running"
            if not self.acq.connected:
                return False, f"{self.acq.backend.label} が未接続です"
            self.running = True
            self.error = ""
        threading.Thread(target=self._worker, args=(channels,), daemon=True).start()
        return True, "impedance started"

    def _worker(self, channels):
        done = set()

        def store(ch, value):
            if 1 <= ch <= CHANNEL_NUM:
                self.values[ch - 1] = value

        def on_channel(ch):
            self.current_ch = ch
            store(ch, "…")            # 計測中のchはそれと分かる表示にする

        def on_value(ch, value):
            # 全ch終わるのを待たず、測れたそばから画面に出す
            store(ch, value)
            done.add(ch)

        try:
            self.acq.pause_stream()
            values = self.acq.backend.measure_impedance(channels, on_channel, on_value)
            # 測ったchだけを上書きする。1ch計測で他chの表示を消さないため、
            # self.values 全体の差し替えはしない。
            for ch, value in (values or {}).items():
                store(ch, value)
                done.add(ch)
        except Exception as exc:      # noqa: BLE001 - 理由をそのままUIに出す
            for ch in channels:       # 測れなかったchだけERRにする
                if ch not in done:
                    store(ch, "ERR")
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[imp] {self.error}")
        finally:
            self.current_ch = None
            try:
                self.acq.resume_stream()
            except Exception as exc:      # noqa: BLE001
                print(f"[imp] resume failed: {exc}")
            self.running = False


# ==== Web ====
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # モデルのアップロード上限
acq = None          # main() で機種を決めてから作る
audio = None
mixer = None
experiment = None
impedance = None
neu = None          # Ne-utsuri モード（neutsuri.NeutsuriEngine）


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
        "device": acq.device_name,
        "device_label": acq.backend.label,
        "device_needs_port": acq.backend.needs_port,
        "sampling_rate": acq.fs,
        "ext_channels": acq.ext_names,
        "audio_device": audio.route if audio else "",
        "audio_is_usb": bool(audio and audio.is_usb),
        "audio_expected": AUDIO_DEVICE_MATCH,
        "audio_card": (audio.card_name if audio else None),
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
        "impedance_error": impedance.error,
        "signal": acq.stats if acq.connected else [],
        "display_paused": acq.display_paused,
        "counts": recording_counts(),
        "current_subject": current_subject(),
        "active_model": active_model_name(),
        "model_ready": bool(active_model_name() and os.path.exists(
            os.path.join(MODELS_DIR, active_model_name() or "", "svm_model.pkl"))),
        "last_model": experiment.last_model,
        "stim_volume": round(audio.gain * 100),
        "device_volume": mixer.percent,
        "device_volume_available": mixer.available,
        "training_running": _training["running"],
        "training_message": _training["message"],
        "clock": {"epoch": time.time(),
                  "synced_at": _clock_sync["at"],
                  "synced_delta": _clock_sync["delta"]},
        "neutsuri_supported": acq.neutsuri_supported,
        "neutsuri": neu.status(),
    })


@app.route("/api/volume/<kind>", methods=["POST"])
def api_volume(kind):
    try:
        value = max(0, min(100, int(request.args.get("value", ""))))
    except ValueError:
        return jsonify({"ok": False, "message": "value は 0-100 の整数"}), 400

    # 全体設定にも残しておく（被験者未選択のときの初期値になる）
    if kind == "stim":
        audio.gain = value / 100.0
        save_setting("stim_volume", value)
        save_subject_volume("stim", value)
        return jsonify({"ok": True, "message": f"刺激音量: {value}%"})
    if kind == "device":
        if not mixer.available:
            return jsonify({"ok": False, "message": "ALSAの音量調整が使えません"}), 409
        ok = mixer.set_percent(value)
        if ok:
            save_setting("device_volume", value)
            save_subject_volume("device", value)
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


@app.route("/api/devices")
def api_devices():
    return jsonify({"devices": acq.device_options(), "current": acq.device_name,
                    "ports": acq.backend.list_ports()})


@app.route("/api/device/select", methods=["POST"])
def api_device_select():
    if experiment.running or impedance.running or (neu is not None and neu.recording):
        return jsonify({"ok": False, "message": "実行中は変更できません"}), 409
    ok, msg = acq.set_device(request.args.get("name", ""))
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/connect", methods=["POST"])
def api_connect():
    """アンプに接続する。ポート未指定なら候補を総当たりして探す。"""
    if experiment.running or impedance.running:
        return jsonify({"ok": False, "message": "実行中です"}), 409
    port = (request.args.get("port") or "").strip()
    if acq.connected:
        acq.disconnect()
    ok = acq.connect(port or None)
    if ok:
        save_setting(f"port_{acq.device_name}", acq.port)
    return jsonify({"ok": ok,
                    "message": (f"{acq.backend.label} に接続しました（{acq.port} / "
                                f"{acq.fs} Hz）" if ok else acq.error)}), (200 if ok else 409)


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    if experiment.running or impedance.running:
        return jsonify({"ok": False, "message": "実行中は切断できません"}), 409
    acq.disconnect()
    return jsonify({"ok": True, "message": "切断しました"})


# ==== 被験者 ====
@app.route("/api/subjects")
def api_subjects():
    return jsonify({"subjects": known_subjects(), "current": current_subject()})


@app.route("/api/subjects/select", methods=["POST"])
def api_subjects_select():
    if experiment.running or (neu is not None and neu.recording):
        return jsonify({"ok": False, "message": "記録中は切り替えられません"}), 409
    name = (request.args.get("name") or "").strip()
    try:
        set_current_subject(name)
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    vols = subject_volumes(name)
    return jsonify({"ok": True,
                    "message": f"被験者: {name}（刺激音量 {vols.get('stim')}%"
                               + (f" / デバイス音量 {vols['device']}%" if 'device' in vols else "")
                               + "）"})


@app.route("/api/subjects/head", methods=["GET", "POST"])
def api_subjects_head():
    """電極配置に使う頭部計測値の読み書き（選択中の被験者のもの）。"""
    if request.method == "POST":
        values = {}
        for key in HEAD_KEYS:
            try:
                values[key] = float(request.args.get(key, ""))
            except ValueError:
                pass
        head = save_subject_head(values)
        if not head:
            return jsonify({"ok": False, "message": "ni / lr は 0-100 cm の数値"}), 400
        return jsonify({"ok": True, "subject": current_subject(), "head": head})
    return jsonify({"ok": True, "subject": current_subject(), "head": subject_head()})


@app.route("/api/impedance", methods=["POST"])
def api_impedance():
    if experiment.running:
        # 計測はストリームを一時停止するので、試行中に走らせると記録が途切れる
        return jsonify({"ok": False, "message": "実験中は計測できません"}), 409
    if neu.recording:
        return jsonify({"ok": False, "message": "Ne-utsuri 記録中は計測できません"}), 409
    ch = request.args.get("ch")
    channels = list(range(1, CHANNEL_NUM + 1)) if ch in (None, "all") else [int(ch)]
    ok, msg = impedance.run(channels)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


# ==== Ne-utsuri モード ====
# 聴覚BCIとは独立したモード。刺激は出さず、ESP32のピエゾ押下と EEG を同時に記録し、
# 「脳波の予測」と「実センサー」のどちらを採用するかを 0–100% の優先度で混ぜる。
def _neutsuri_guard():
    """Ne-utsuri を使える状態か。使えない理由があれば応答を返す。

    押下判定モデルは Cyton の 250 Hz・2秒窓で学習されている。他のレートのアンプで
    ONにしても意味のある確率が出ないので、モードそのものを塞ぐ。
    """
    if acq.neutsuri_supported:
        return None
    return jsonify({"ok": False,
                    "message": f"{acq.backend.label}（{acq.fs} Hz）では Ne-utsuri を"
                               f"使えません（押下判定モデルは 250 Hz 学習）"}), 409


@app.route("/api/neutsuri/mode", methods=["POST"])
def api_neutsuri_mode():
    on = request.args.get("on", "1") not in ("0", "false", "off")
    if on and experiment.running:
        return jsonify({"ok": False, "message": "聴覚BCIの実験中は開始できません"}), 409
    if on:
        blocked = _neutsuri_guard()
        if blocked:
            return blocked
    ok, msg = neu.set_enabled(on)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/neutsuri/usage", methods=["POST"])
def api_neutsuri_usage():
    try:
        value = int(request.args.get("value", ""))
    except ValueError:
        return jsonify({"ok": False, "message": "value は 0-100 の整数"}), 400
    return jsonify({"ok": True, "message": neu.set_usage(value)})


@app.route("/api/neutsuri/threshold", methods=["POST"])
def api_neutsuri_threshold():
    try:
        value = float(request.args.get("value", ""))
    except ValueError:
        return jsonify({"ok": False, "message": "value は 0.30-0.95 の実数"}), 400
    return jsonify({"ok": True, "message": neu.set_threshold(value)})


@app.route("/api/neutsuri/start", methods=["POST"])
def api_neutsuri_start():
    if experiment.running:
        return jsonify({"ok": False, "message": "聴覚BCIの実験中は記録できません"}), 409
    if impedance.running:
        return jsonify({"ok": False, "message": "インピーダンス計測中は記録できません"}), 409
    if not acq.connected:
        return jsonify({"ok": False, "message": f"{acq.backend.label} が未接続です"}), 409
    blocked = _neutsuri_guard()
    if blocked:
        return blocked
    ok, msg = neu.start_recording()
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/neutsuri/stop", methods=["POST"])
def api_neutsuri_stop():
    ok, msg = neu.stop_recording()
    if ok:
        _counts_cache["value"] = None
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/neutsuri/esp32", methods=["POST"])
def api_neutsuri_esp32():
    """ESP32を探し直す（挿し直し・ファーム焼き直しのあと）。"""
    if not neu.enabled:
        return jsonify({"ok": False, "message": "Ne-utsuri モードがOFFです"}), 409
    neu.link.redetect()
    return jsonify({"ok": True, "message": "ESP32を再検出します（数秒かかります）"})


@app.route("/api/neutsuri/test", methods=["POST"])
def api_neutsuri_test():
    """SPACE を手動で1回送る。ファームが SENT / NO_BLE を返すので BLE 側まで確認できる。"""
    if not neu.link.connected:
        return jsonify({"ok": False, "message": "ESP32が未接続です"}), 409
    neu.link.send_space()
    return jsonify({"ok": True, "message": "SPACE を送信しました（応答は ESP32 欄に出ます）"})


@app.route("/api/neutsuri/trace")
def api_neutsuri_trace():
    """押下・確率・送信コマンドの間引きトレース（EEG波形と同じ時間軸）。"""
    return Response(neu.frame, mimetype="application/octet-stream",
                    headers={"X-Rows": str(neu.frame_shape[0]),
                             "X-Samples": str(neu.frame_shape[1])})


@app.route("/api/neutsuri/model/reload", methods=["POST"])
def api_neutsuri_model_reload():
    ok = neu.reload_model()
    return jsonify({"ok": ok,
                    "message": (f"モデルを読み込みました（{neu.predictor.info}）" if ok
                                else f"読み込み失敗: {neu.predictor.error}")})


# ==== 予測モデルの管理 ====
# models/<被験者>/ を1つのモデルセットとして扱う。
#   svm_model.pkl … 必須。学習済みバンドル。
#   model.py      … 任意。前処理ごと差し替えたいときだけ置く。無ければ同梱のものを使う。
#   note.txt      … 任意。UIに出す覚え書き。
# 使用中のモデルは常に「選択中の被験者のモデル」。別々に選べるようにすると
# 食い違ったまま実験してしまうので、被験者名と1対1に固定する。
MODELS_DIR = os.path.join(BASE_DIR, "models")
# 学習でも予測でも model.py はこの名前で読む。pkl 内の自作クラスの参照名に
# なるので、学習した pkl を後から復元できるよう固定しておく必要がある。
SHARED_MODEL_MODULE = "bci_model_shared"
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


def load_module_from(path, module_name):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def bundled_path(filename):
    """共有モジュール（model.py / hyperparam_search.py）の場所。

    Pi では deploy.sh が bci_server.py と同じ階層に置く。リポジトリのまま動かす
    ときは pi_bci/ の親（リポジトリ直下）にあるので、そちらも見る。
    """
    here = os.path.join(BASE_DIR, filename)
    if os.path.exists(here):
        return here
    parent = os.path.join(os.path.dirname(BASE_DIR), filename)
    return parent if os.path.exists(parent) else here


_bundled_model = None


def register_bundled_as_model(mod, overwrite=True):
    """`import model` をこのモジュールに解決させる。

    hyperparam_search.py は `import model` で同梱 model.py を読む。また
    `python model.py` や train_model_set.py が作った pkl は自作クラスを
    `model.TrialCalibratedSVC` として参照する。Pi では deploy.sh が model.py を
    bci_server.py と同階層に置くので通るが、リポジトリのまま Mac で動かすと
    model.py はひとつ上（リポジトリ直下）にあって sys.path に無いため
    ModuleNotFoundError になる。読み込み済みのものを `model` としても登録して塞ぐ。
    コピーを二重に持たせないよう、新しく import させるのではなく使い回す。
    """
    if overwrite or "model" not in sys.modules:
        sys.modules["model"] = mod
    return mod


def bundled_model_module():
    """同梱の model.py を一度だけ読み込む（重いので遅延）。"""
    global _bundled_model
    if _bundled_model is None:
        _bundled_model = load_module_from(bundled_path("model.py"), "bci_model_bundled")
        # 学習で読み直したものが既にあればそちらを優先する（そちらが最新）。
        register_bundled_as_model(_bundled_model, overwrite=False)
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
    # ブラウザの「学習」で作った pkl は自作クラスを SHARED_MODEL_MODULE の名前で
    # 参照する。予測時にその名前で引けるよう、まだ誰も登録していなければ登録する。
    #
    # 上書きしてはいけない。pickle はクラスを名前で照合するので、学習中の
    # モジュールから名前を奪うと最後の保存が
    # 「Can't pickle <class 'bci_model_shared.TrialCalibratedSVC'>:
    #   it's not the same object as ...」で失敗する。学習の裏でモデル一覧
    # (`/api/models` → bundle_info) を開くだけで踏む。
    # 復元だけならどのコピーでもクラスは引けるので、先に登録されている方を尊重してよい。
    sys.modules.setdefault(SHARED_MODEL_MODULE, mod)


class _StaleTrainModuleFinder:
    """使い捨てモジュール名で保存された古い pkl を読めるようにする import フック。

    2026-08 以前の「学習」ボタンは model.py を bci_train_model_<ミリ秒> という
    一度きりの名前で読んでいたため、その pkl は TrialCalibratedSVC をその名前で
    参照している。名前は学習スレッドが終われば消え、予測は
    ModuleNotFoundError で失敗する。中身は同梱 model.py と同じなので横流しする。
    """
    PREFIX = "bci_train_model_"

    def find_spec(self, fullname, path=None, target=None):
        import importlib.machinery
        if not fullname.startswith(self.PREFIX):
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):
        import types
        shim = types.ModuleType(spec.name)
        shim.__dict__.update(vars(bundled_model_module()))
        shim.__name__ = spec.name
        return shim

    def exec_module(self, module):
        pass


sys.meta_path.append(_StaleTrainModuleFinder())


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
            acc = b.get("cv_accuracy", b.get("loocv_acc"))
            info = {
                "preset": b.get("preset"),
                "labels": b.get("label_candidates") or b.get("labels"),
                "trained_on": os.path.basename(
                    str(b.get("data_folder", "")).rstrip("/\\")),
                "loocv_acc": round(acc, 4) if isinstance(acc, (int, float)) else None,
                # 1000Hz学習のモデルに250Hzの記録を食わせても例外は出ないので、
                # どのレートで学習したかは一覧に必ず出す。
                "sampling_rate": b.get("sampling_rate"),
                "use_sequence": b.get("use_sequence"),
                "n_files": b.get("n_files"),
                "trained_at": b.get("trained_at"),
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
        src = bundled_path("model.py")

    mod = load_module_from(src, f"bci_model_{name}")
    # model.py はモジュール変数 model_path を見に行くので、セットの pkl を指させる
    mod.model_path = pkl
    if not own:
        # 同梱 model.py の BACKEND は既定が "legacy"。そのままだと clfs.pkl /
        # model.pkl（Pi には配っていない）を探しに行って予測が -1 になる。
        # セットの pkl は必ず train_svm() のバンドルなので svm に固定する。
        mod.BACKEND = "svm"
    alias_model_classes(mod)
    print(f"[model] loaded '{name}' (model.py: {'own' if own else 'bundled'})")

    _loaded_models[name] = mod
    return mod


def active_model_name():
    """使用中のモデル名 = 選択中の被験者名（1対1で対応）。"""
    return current_subject()


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
    """選択中の被験者フォルダ直下の CSV だけを指すことを保証してフルパスを返す。

    ブラウザから来た名前をそのまま結合するとディレクトリ外へ抜けられるので、
    書式を絞ったうえで realpath が被験者フォルダ直下に収まることも確認する。
    """
    d = subject_data_dir()
    if not d or not name or not SAFE_NAME.match(name):
        return None
    path = os.path.realpath(os.path.join(d, name))
    if os.path.dirname(path) != os.path.realpath(d):
        return None
    return path


def sidecar_paths(csv_path):
    """CSVに付随するファイル（タイミング記録・予測結果・解析キャッシュ）を集める。"""
    stem = os.path.splitext(csv_path)[0]
    # _predict.json はどのモデルがどう予測したかの記録。CSVだけ消して残すと
    # 実体のない予測結果が居座るので、必ず一緒に扱う。
    paths = [stem + "_timing.json", stem + "_predict.json", stem + "_impedance.csv"]
    side = neutsuri.sidecar_for(csv_path)      # Ne-utsuri 記録の条件・実測タイミング
    if side:
        paths.append(side)
    # model.py は ./eeg_cache/<folder>_<stem>_lc..._hc....npy を作る。消し忘れると
    # 同じ名前を付け直したときに古い波形が再利用されてしまう。
    paths += glob.glob(os.path.join(BASE_DIR, "eeg_cache",
                                    f"*{os.path.basename(stem)}_*.npy"))
    return paths


@app.route("/api/files")
def api_files():
    rows = []
    d = subject_data_dir()
    if not d:
        return jsonify({"files": rows, "subject": None})
    for path in sorted(glob.glob(os.path.join(d, "*.csv")), reverse=True):
        stem = os.path.splitext(path)[0]
        key = os.path.basename(stem).rsplit("_", 1)[-1]
        st = os.stat(path)
        side = neutsuri.sidecar_for(path)
        rows.append({
            "name": os.path.basename(path),
            "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            # Ne-utsuri 記録はターゲットの概念がないので、モード名を出して区別する
            "target": "Ne-utsuri" if side else TARGET_LABEL.get(key, "—"),
            "timing": os.path.exists(side or (stem + "_timing.json")),
            # 平均ERPを出せる記録か（ラベル付きの聴覚BCI記録だけ）
            "erp": bool(not side and erp_view.target_of(path)),
        })
    return jsonify({"files": rows, "subject": current_subject()})


# 平均ERPの計算結果。同じファイルを繰り返し開いても読み直さないよう、
# ファイルの更新時刻ごとに数件だけ持っておく（Piでは1本 0.2 秒ほどかかる）。
_erp_cache = OrderedDict()
ERP_CACHE_MAX = 6


def latest_erp_file():
    """平均ERPを出せる記録のうち最新のもの（無ければ None）。"""
    d = subject_data_dir()
    if not d:
        return None
    for path in sorted(glob.glob(os.path.join(d, "*.csv")), reverse=True):
        if erp_view.target_of(path) and not neutsuri.sidecar_for(path):
            return os.path.basename(path)
    return None


@app.route("/api/erp")
def api_erp():
    """記録1本の条件別平均ERP。数値だけ返し、描画はブラウザ側で行う。

    リアルタイム表示と同じく、記録中は取得スレッドを邪魔しないよう計算しない。
    """
    if experiment.running or (neu is not None and neu.recording):
        return jsonify({"ok": False, "message": "記録中は計算しません（タイミング優先）"}), 409

    name = request.args.get("file") or latest_erp_file()
    if not name:
        return jsonify({"ok": False, "message": "ラベル付きの記録がまだありません"}), 404
    path = resolve_data_file(name)
    if path is None or not os.path.exists(path):
        return jsonify({"ok": False, "message": f"{name} が見つかりません"}), 404

    key = os.path.basename(path)
    mtime = os.stat(path).st_mtime
    hit = _erp_cache.get(key)
    if hit and hit[0] == mtime:
        _erp_cache.move_to_end(key)
        return jsonify(hit[1])

    try:
        result = erp_view.compute(path)
    except erp_view.ErpError as exc:
        return jsonify({"ok": False, "file": os.path.basename(path),
                        "message": str(exc)}), 422
    except Exception as exc:      # noqa: BLE001 - surfaced to the browser
        return jsonify({"ok": False, "file": os.path.basename(path),
                        "message": f"{type(exc).__name__}: {exc}"}), 500

    result["target_label"] = TARGET_LABEL.get(result["target"], "—")
    _erp_cache[key] = (mtime, result)
    while len(_erp_cache) > ERP_CACHE_MAX:
        _erp_cache.popitem(last=False)
    return jsonify(result)


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
    if neu.recording and neu.file:
        # Ne-utsuri は停止まで書き続けるので、走っている分は必ず外す
        skip.add(os.path.basename(neu.file))

    d = subject_data_dir()
    if not d:
        return jsonify({"ok": False, "message": "被験者が選択されていません"}), 400
    members = [p for p in sorted(glob.glob(os.path.join(d, "*")))
               if os.path.isfile(p) and os.path.basename(p) not in skip]
    if not members:
        return jsonify({"ok": False, "message": "ダウンロードできるファイルがありません"}), 404

    # メモリに載せると記録数が増えたときに苦しいので一時ファイルに書く
    tmp = tempfile.NamedTemporaryFile(prefix="bci_data_", suffix=".zip", delete=False)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in members:
                z.write(path, arcname=os.path.join("data", current_subject(),
                                                   os.path.basename(path)))
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
    if experiment.running or neu.recording:
        return jsonify({"ok": False, "message": "記録中は削除できません"}), 409

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
    _erp_cache.clear()          # 名前と中身の対応が変わったので作り直させる
    msg = f"{len(deleted)}件を削除しました"
    if failed:
        msg += f"（失敗 {len(failed)}件: {', '.join(failed[:3])}）"
    return jsonify({"ok": not failed, "message": msg, "deleted": deleted})


@app.route("/api/files/rename", methods=["POST"])
def api_files_rename():
    if experiment.running or neu.recording:
        return jsonify({"ok": False, "message": "記録中はリネームできません"}), 409

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
        old_side = neutsuri.sidecar_for(src)
        os.rename(src, dst)
        if os.path.exists(old_stem + "_timing.json"):
            os.rename(old_stem + "_timing.json", new_stem + "_timing.json")
        # Ne-utsuri 記録の付随JSONは `<幹>.json`。新しい名前が _neutsuri.csv でなくなっても
        # 幹に合わせて連れて行く（置いていくと元の記録の条件が読めなくなる）。
        if old_side and os.path.exists(old_side):
            os.rename(old_side, neutsuri.sidecar_for(dst) or (new_stem + ".json"))
        for cache in glob.glob(os.path.join(BASE_DIR, "eeg_cache",
                                            f"*{os.path.basename(old_stem)}_*.npy")):
            os.remove(cache)
    except OSError as exc:      # noqa: BLE001
        return jsonify({"ok": False, "message": f"リネーム失敗: {exc}"}), 500

    _counts_cache["value"] = None
    _erp_cache.clear()          # リネームは mtime が変わらないので必ず捨てる
    if experiment.last_file and os.path.basename(experiment.last_file) == old:
        experiment.last_file = dst
    return jsonify({"ok": True, "message": f"{old} → {os.path.basename(dst)}"})


@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models(), "active": active_model_name()})


@app.route("/api/models/select", methods=["POST"])
def api_models_select():
    """モデルの選択は被験者の選択と同じこと。被験者切り替えに読み替える。

    旧UI（モデル一覧から選ぶ）から叩かれても壊れないように残してある。
    """
    return api_subjects_select()


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
        return jsonify({"ok": False,
                        "message": "選択中の被験者のモデルは削除できません"}), 409
    import shutil
    try:
        shutil.rmtree(d)
    except OSError as exc:      # noqa: BLE001
        return jsonify({"ok": False, "message": f"削除失敗: {exc}"}), 500
    _loaded_models.pop(name, None)
    return jsonify({"ok": True, "message": f"{name} を削除しました"})


# ==== 被験者データでの学習 ====
# 学習はブラウザから起動し、進行状況は /api/state の training_* で拾う。
# 同梱 model.py / hyperparam_search.py は毎回ディスクから読み直す。サーバーを
# 起動したまま学習側を直すことがあるので、その修正を反映させるため。
_training = {"running": False, "message": "", "subject": None}


def _fresh_module(filename):
    """同梱モジュールをディスクから読み直す。

    model.py だけは固定名（SHARED_MODEL_MODULE）で読む。学習した pkl は
    TrialCalibratedSVC などを「学習時のモジュール名」で参照するので、
    使い捨ての名前で読むとその名前が消えた後の予測で
    ModuleNotFoundError になり、pkl を復元できなくなる。
    """
    if filename == "model.py":
        mod = load_module_from(bundled_path(filename), SHARED_MODEL_MODULE)
        register_bundled_as_model(mod)
        return mod
    return load_module_from(bundled_path(filename),
                           f"bci_train_{os.path.splitext(filename)[0]}_"
                           f"{int(time.time() * 1000)}")


def _requested_use_sequence(body):
    """リクエストの刺激セット数（UIの選択値）を検証して返す。未指定なら None。"""
    raw = body.get("use_sequence")
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("刺激セット数は整数で指定してください") from exc
    if not 1 <= n <= SET_NUM + 1:
        raise ValueError("刺激セット数が不正です")
    return n


def _acc_text(value):
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _after_training(subject, out):
    """学習済み pkl を差し替えたので、読み込み済みのモデルを捨てる。"""
    _loaded_models.pop(subject, None)
    _bundle_info.pop(out, None)


def _run_training(subject, use_sequence=None):
    _training.update(running=True, message=f"{subject}: 学習中…", subject=subject)
    try:
        mod = _fresh_module("model.py")
        d = subject_data_dir(subject)
        out = os.path.join(MODELS_DIR, subject, "svm_model.pkl")
        stats = mod.train(data_folder_override=d, model_path_override=out,
                          use_sequence_override=use_sequence)
        _after_training(subject, out)
        _training["message"] = (
            f"{subject}: 学習完了（LOOCV {_acc_text(stats.get('cv_accuracy'))}, "
            f"{stats.get('n_files', '?')}ファイル, "
            f"{stats.get('use_sequence', '?')}セット/トライアル, "
            f"{stats.get('sampling_rate', '?')} Hz）")
    except Exception as exc:      # noqa: BLE001 - 失敗理由はUIに出す
        import traceback
        traceback.print_exc()
        _training["message"] = f"{subject}: 学習失敗 — {type(exc).__name__}: {exc}"
    finally:
        _training["running"] = False


def _training_preconditions():
    """学習を始められるか。(被験者名, エラー応答) を返す。"""
    if experiment.running or (neu is not None and neu.recording):
        return None, (jsonify({"ok": False, "message": "記録中は学習できません"}), 409)
    if _training["running"]:
        return None, (jsonify({"ok": False, "message": "学習中です"}), 409)
    subject = current_subject()
    if not subject:
        return None, (jsonify({"ok": False, "message": "被験者が選択されていません"}), 400)
    d = subject_data_dir(subject)
    if not d or not glob.glob(os.path.join(d, "*.csv")):
        return None, (jsonify({"ok": False,
                               "message": f"{subject} の記録データがありません"}), 400)
    return subject, None


@app.route("/api/train", methods=["POST"])
def api_train():
    subject, err = _training_preconditions()
    if err:
        return err
    try:
        use_sequence = _requested_use_sequence(request.get_json(silent=True) or {})
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    threading.Thread(target=_run_training, args=(subject, use_sequence),
                     daemon=True).start()
    return jsonify({"ok": True, "message": f"{subject} のデータで学習を開始しました"})


# ---- ハイパーパラメータ探索付きの学習（時間予算を指定する） ----
SEARCH_DURATION_CHOICES = {
    "10m": 10 * 60, "30m": 30 * 60, "1h": 3600, "2h": 2 * 3600,
    "3h": 3 * 3600, "6h": 6 * 3600, "12h": 12 * 3600, "24h": 24 * 3600,
}


def _run_training_search(subject, duration_seconds, use_sequence=None):
    _training.update(running=True, subject=subject,
                     message=f"{subject}: ハイパーパラメータ探索を開始します"
                             f"（最大 {duration_seconds/60:.0f}分）…")
    try:
        # model.py が先。hyperparam_search.py は読み込み時に `import model` を
        # 実行するので、その前に同梱 model.py を `model` として登録しておく。
        mod = _fresh_module("model.py")
        hs = _fresh_module("hyperparam_search.py")
        d = subject_data_dir(subject)
        out = os.path.join(MODELS_DIR, subject, "svm_model.pkl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ckpt = os.path.join(MODELS_DIR, subject, "hyperparam_search_checkpoint.json")

        def log(msg):
            print(msg)
            _training["message"] = f"{subject}: {msg}"

        # 短い時間予算では最終確認評価に取り過ぎないよう、予算に応じて短縮する。
        reserve = min(300.0, max(20.0, duration_seconds * 0.05))
        result = hs.run_search(d, duration_seconds, final_reserve_seconds=reserve,
                               checkpoint_path=ckpt, log=log,
                               use_sequence_override=use_sequence)

        with open(os.path.join(MODELS_DIR, subject, "best_params.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log("探索完了。見つかったパラメータで最終モデルを学習しています…")
        stats = mod.train(data_folder_override=d, model_path_override=out,
                          params_=result["params"],
                          use_sequence_override=result.get("use_sequence"))
        _after_training(subject, out)
        _training["message"] = (
            f"{subject}: 探索→学習完了（探索{result['iterations']}件, "
            f"探索時 CV平均 {_acc_text(result['cv_accuracy_search'])}, "
            f"確認評価 {_acc_text(result['cv_accuracy_final'])}"
            f"[{' '.join(f'{k}={v:.2f}' for k, v in (result.get('cv_per_scheme_final') or {}).items())}], "
            f"最終学習 {_acc_text(stats.get('cv_accuracy'))}, "
            f"{stats.get('use_sequence', '?')}セット/トライアル）")
    except Exception as exc:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        _training["message"] = f"{subject}: 探索/学習失敗 — {type(exc).__name__}: {exc}"
    finally:
        _training["running"] = False


@app.route("/api/train_search", methods=["POST"])
def api_train_search():
    subject, err = _training_preconditions()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    key = body.get("duration")
    duration = SEARCH_DURATION_CHOICES.get(key)
    if duration is None:
        return jsonify({"ok": False, "message": "探索時間の指定が不正です"}), 400
    try:
        use_sequence = _requested_use_sequence(body)
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    threading.Thread(target=_run_training_search,
                     args=(subject, duration, use_sequence), daemon=True).start()
    return jsonify({"ok": True,
                    "message": f"{subject} のデータでハイパーパラメータ探索"
                               f"（最大{key}）を開始しました"})


# ==== 時刻合わせ（画面を開いている PC の時計から）====
# Pi には RTC が無く、電源を切ると時計が止まる。直結LANのときはネットに出られず
# NTP も当てにできないので、記録ファイル名も CSV の Timestamp も過去の日時に
# なってしまう。ブラウザが定期的に自分の時計を送ってきて、ずれていたら合わせる。
CLOCK_SYNC_THRESHOLD = 5.0        # これ以上ずれていたら合わせる（秒）
CLOCK_MIN_EPOCH = 1767225600.0    # 2026-01-01。これより前を送ってくる時計は壊れている
CLOCK_MAX_EPOCH = 4102444800.0    # 2100-01-01
_clock_sync = {"at": None, "delta": None, "source": None}


def _format_clock_delta(seconds):
    """ずれを読める単位にする（+42秒 / +12分 / +6.8日）。"""
    a = abs(seconds)
    if a < 90:
        return f"{seconds:+.0f}秒"
    if a < 5400:
        return f"{seconds / 60:+.0f}分"
    if a < 2 * 86400:
        return f"{seconds / 3600:+.1f}時間"
    return f"{seconds / 86400:+.1f}日"


def set_system_clock(epoch):
    """システム時刻を epoch（UNIX秒）に合わせる。

    NTP が有効なままだと `timedatectl set-time` は断られるので date を使う。
    RTC が無いぶん、次の起動時にここから始まるよう fake-hwclock にも書く。
    """
    subprocess.run(["sudo", "-n", "date", "-u", "-s", f"@{epoch:.3f}"],
                   check=True, capture_output=True, text=True)
    save = subprocess.run(["sudo", "-n", "/usr/sbin/fake-hwclock", "save"],
                          capture_output=True, text=True)
    if save.returncode != 0:      # 無くても実害はないので続ける
        print(f"[clock] fake-hwclock save skipped: {save.stderr.strip()}")


@app.route("/api/time", methods=["POST"])
def api_time():
    """ブラウザの PC 時計で Pi の時計を合わせる。"""
    body = request.get_json(silent=True) or {}
    try:
        epoch = float(body.get("epoch"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "epoch（UNIX秒）が必要です"}), 400
    if not CLOCK_MIN_EPOCH <= epoch <= CLOCK_MAX_EPOCH:
        return jsonify({"ok": False, "message": "PC側の時刻が異常です"}), 400

    delta = epoch - time.time()
    if abs(delta) <= CLOCK_SYNC_THRESHOLD:
        return jsonify({"ok": True, "changed": False, "delta": delta,
                        "message": "時刻はPCと一致しています"})

    # 時計を飛ばすと brainflow の Timestamp・刺激の予定時刻・探索の時間予算が
    # そろって食い違う。動いている間は触らず、終わってから合わせる。
    busy = (experiment.running or impedance.running or _training["running"]
            or (neu is not None and neu.recording))
    if busy:
        return jsonify({"ok": False, "changed": False, "delta": delta,
                        "message": "実行中なので時刻合わせを見送りました"}), 409

    try:
        set_system_clock(epoch)
    except Exception as exc:      # noqa: BLE001 - 失敗理由はUIに出す
        detail = getattr(exc, "stderr", "") or str(exc)
        return jsonify({"ok": False, "changed": False, "delta": delta,
                        "message": f"時刻合わせに失敗: {detail}"}), 500

    _clock_sync.update(at=time.time(), delta=delta, source=request.remote_addr)
    print(f"[clock] {_format_clock_delta(delta)} 補正 → "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} (from {request.remote_addr})")
    return jsonify({"ok": True, "changed": True, "delta": delta,
                    "message": f"PCの時計に合わせました（{_format_clock_delta(delta)}）"})


@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    """USB機器（オーディオI/F・EEGアンプ）を検出し直す。

    オーディオは起動時にしか見ていなかったので、挿し直してもサービスを再起動
    するまで戻らなかった。ここで両方まとめて拾い直す。
    """
    global mixer
    if experiment.running or impedance.running or neu.recording:
        return jsonify({"ok": False, "message": "実行中です"}), 409

    parts = []
    try:
        audio.reopen()
        # 内蔵ジャックへフォールバックしている間はALSAつまみを出さない。
        # そこで動かせるのは「USB機器の音量」ではないため。
        mixer = AlsaMixer(audio.card if audio.is_usb else None)
        apply_subject_volumes()      # 選択中の被験者の音量に戻す
        parts.append("音声: " + (audio.route if audio.is_usb
                                else f"{audio.route}（USB未検出）"))
    except Exception as exc:      # noqa: BLE001
        parts.append(f"音声の再取得に失敗: {type(exc).__name__}: {exc}")

    acq.disconnect()
    eeg_ok = acq.connect(load_settings().get(f"port_{acq.device_name}"))
    parts.append(f"{acq.backend.label}: " + (str(acq.port) if eeg_ok else acq.error))

    # Ne-utsuri モード中は ESP32 も同じボタンで拾い直す（USB機器の再検出なので）
    if neu.enabled:
        neu.link.redetect()
        parts.append("ESP32: 再検出中")

    return jsonify({"ok": eeg_ok and audio.is_usb, "message": " / ".join(parts)})


def main():
    global acq, audio, mixer, experiment, impedance, neu

    # 前回使っていたアンプで立ち上げる（ブラウザから切り替えられる）
    settings = load_settings()
    acq = Acquisition(settings.get("device", DEFAULT_DEVICE))
    print(f"[eeg] device={acq.backend.label} ({acq.fs} Hz, EXT {acq.n_ext}ch)")
    for row in acq.device_options():
        if not row["available"]:
            print(f"[eeg] {row['label']}: 使用不可 — {row['reason']}")

    audio = AudioEngine()
    print(f"[audio] device={audio.device_name} sr={audio.samplerate} "
          f"card={audio.card} ({audio.card_name}) usb={audio.is_usb}")
    if not audio.is_usb:
        print("[audio] WARNING: no USB sound card in the output path — falling back to the "
              "built-in output. Stimuli will not come out of the USB interface.")

    mixer = AlsaMixer(audio.card if audio.is_usb else None)
    if mixer.available:
        print(f"[audio] mixer card={mixer.card} control='{mixer.control}' "
              f"range={mixer.vmin}-{mixer.vmax} now={mixer.percent}%")
    else:
        print("[audio] ALSA volume control not available")

    # 選択中の被験者の音量を復元する（ALSA側は再起動で既定値へ戻るため）
    subject = current_subject()
    vols = apply_subject_volumes(subject)
    print(f"[audio] restored volumes for {subject or '(被験者未選択)'}: "
          f"stim={round(audio.gain * 100)}% device={mixer.percent}%")

    experiment = Experiment(acq, audio)
    impedance = ImpedanceRunner(acq)

    # Ne-utsuri モード。Cyton ドングルと同じ /dev/ttyUSB* に ESP32 が出るので、
    # 掴んでいるポートを候補から外せるよう acq を参照させる。
    neu = neutsuri.NeutsuriEngine(BASE_DIR, subject_data_dir() or DATA_DIR,
                                  acq.buffer_size, acq.decim,
                                  save_setting=save_setting,
                                  exclude_port=lambda: acq.port,
                                  model_dir=(os.path.join(MODELS_DIR, current_subject())
                                             if current_subject() else None))
    acq.set_neutsuri_engine(neu)
    neu.start()
    neu.restore(settings)
    print(f"[neutsuri] model: {os.path.relpath(neu.predictor.path, BASE_DIR)}: "
          f"{neu.predictor.info or neu.predictor.error} / "
          f"mode={'ON' if neu.enabled else 'OFF'} usage={neu.usage}% "
          f"threshold={neu.threshold:.2f}")

    if BoardShim is not None:
        try:
            BoardShim.disable_board_logger()
        except Exception as exc:      # noqa: BLE001
            print(f"[eeg] board logger: {exc}")
    # 旧レイアウト（data/ 直下）の記録は被験者フォルダに入っていないので一覧に出ない。
    # 黙って見えなくなると「消えた」と誤解されるため、起動時に必ず知らせる。
    legacy = [os.path.basename(x) for x in glob.glob(os.path.join(DATA_DIR, "*.csv"))]
    if legacy:
        print(f"[data] data/ 直下に旧レイアウトの記録が {len(legacy)} 本あります "
              f"（例 {legacy[0]}）。被験者ごとの一覧には出ません。使うには "
              f"data/<被験者名>/ へ移してください:")
        print(f"       mkdir -p {DATA_DIR}/<被験者名> && "
              f"mv {DATA_DIR}/*.csv {DATA_DIR}/*.json {DATA_DIR}/<被験者名>/")

    if acq.connect(settings.get(f"port_{acq.device_name}")):
        print(f"[eeg] connected on {acq.port} @ {acq.fs} Hz")
    else:
        print(f"[eeg] not connected: {acq.error}")

    # werkzeug は既定でリクエストを1件ずつログに書く。ブラウザが毎秒何度も
    # ポーリングするのでそのぶん取得スレッドのCPUを奪う。警告以上だけ残す。
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # SIGTERM で終わると brainflow のセッションが返らず、次の起動が
    # BOARD_NOT_READY_ERROR になる（systemctl restart / kill の両方で起きる）。
    # 記録中に落とされた場合は CSV も閉じないままサイドカーも書かれない。
    # どちらも後始末してから終わる。
    def _shutdown(signum, _frame):
        print(f"[web] signal {signum} received — shutting down")
        try:
            if neu is not None and neu.recording:
                ok, msg = neu.stop_recording()
                print(f"[neutsuri] {msg}")
        except Exception as exc:      # noqa: BLE001
            print(f"[neutsuri] stop on shutdown failed: {exc}")
        try:
            acq.disconnect()
            print("[eeg] session released")
        except Exception as exc:      # noqa: BLE001
            print(f"[eeg] disconnect on shutdown failed: {exc}")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    for port in (WEB_PORT, WEB_PORT_FALLBACK):
        try:
            suffix = "" if port == 80 else f":{port}"
            print(f"[web] http://raspberrypi.local{suffix}/  (bind {WEB_HOST})")
            app.run(host=WEB_HOST, port=port, threaded=True, debug=False, use_reloader=False)
            return
        except PermissionError:
            print(f"[web] port {port} not permitted, falling back")
        except OSError as exc:
            print(f"[web] port {port} unavailable ({exc}), falling back")


if __name__ == "__main__":
    main()
