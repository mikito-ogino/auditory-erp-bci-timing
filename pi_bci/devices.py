#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEG アンプのバックエンド（OpenBCI Cyton / Miyuki Giken Polymate Pocket）。

`bci_server.py` の Acquisition はここで定義したバックエンドだけを触る。アンプごとに
違うのは「開き方・読み方・インピーダンスの測り方・サンプリングレート・補助チャンネル」
だけなので、その4点をこのファイルに閉じ込めて Acquisition 側を共通化する。

バックエンドは import 時にハードウェア依存のモジュールを読まない。Polymate の API は
Windows 専用の拡張モジュール（`polymate/pocket.pyd`）なので、Pi や Mac ではそもそも
import できない。選ばれたときに初めて読み、失敗した理由をそのままUIに出す。
"""

import glob
import os
import re
import subprocess
import sys
import time

import numpy as np
import serial.tools.list_ports

CYTON = "cyton"
POLYMATE = "polymate"


class DeviceError(RuntimeError):
    """アンプを開けなかった・読めなかったときに投げる（メッセージはUIに出る）。"""


class DeviceBackend:
    """アンプ1機種ぶんのインターフェース。

    read() は「今バッファに溜まっている分」を返す。戻り値は
    (eeg[ch, n] µV, stamps[n] 実時刻, pkgs[n] or None, ext[n, n_ext] or None)。
    データが無ければ None を返す（呼び出し側が待つ）。
    """

    name = ""
    label = ""
    sampling_rate = 250
    n_ext = 0                   # 補助チャンネル数（音響トリガーなど）
    ext_names = ()
    needs_port = False          # UIにポート指定欄を出すか
    read_interval = 0.02        # read() を呼ぶ間隔 [s]
    supports_impedance = True
    supports_neutsuri = False   # Ne-utsuri モード（250Hz・ESP32 前提）を出すか

    def __init__(self):
        self.port = None

    # --- 可用性 ---
    def availability(self):
        """(使えるか, 理由) を返す。理由は使えないときだけ入る。"""
        return True, ""

    def list_ports(self):
        """接続候補のポート一覧（UI表示用）。"""
        return [{"port": p.device, "description": (p.description or "")}
                for p in serial.tools.list_ports.comports()]

    # --- 接続 ---
    def open(self, port=None):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def pause(self):
        raise NotImplementedError

    def resume(self):
        raise NotImplementedError

    def measure_impedance(self, channels, on_channel=None, on_value=None):
        """実際に測ったchだけを {ch: kΩ文字列} で返す。

        on_channel(ch) は計測開始、on_value(ch, 値) は1chぶんの結果が出た時点の
        通知。まとめて返すだけだと全ch終わるまで画面が更新されないので、1chずつ
        測れる機種は必ず on_value を呼ぶ。測っていないchを戻り値に入れないのは、
        呼び出し側がそのchの既存の表示を消さずに済むようにするため。
        """
        raise NotImplementedError


# =====================================================================
# OpenBCI Cyton（brainflow・250 Hz・補助チャンネルなし）
# =====================================================================
class CytonBackend(DeviceBackend):
    name = CYTON
    label = "OpenBCI Cyton"
    sampling_rate = 250
    n_ext = 0
    needs_port = False          # FTDI ドングルを自動で見つける
    read_interval = 0.02
    supports_impedance = True
    supports_neutsuri = True

    # Ne-utsuri の ESP32 も CP2102 として /dev/ttyUSB* に出る。フォールバックで
    # そちらを掴むと Cyton も ESP32 も動かなくなるので必ず外す。PID が None の
    # 行は「その VID なら PID を問わず外す」（C6 等の内蔵USBは製品ごとに PID が違う）。
    EXCLUDE_VID_PID = ()

    def __init__(self, channel_num=8, exclude_vid_pid=()):
        super().__init__()
        self.channel_num = channel_num
        self.EXCLUDE_VID_PID = tuple(exclude_vid_pid)
        self.board = None
        self._eeg_rows = None
        self._ts_row = None
        self._pkg_row = None

    def availability(self):
        try:
            from brainflow.board_shim import BoardShim      # noqa: F401
        except Exception as exc:      # noqa: BLE001
            return False, f"brainflow が読み込めません: {exc}"
        return True, ""

    def _excluded(self, port):
        vid = getattr(port, "vid", None)
        pid = getattr(port, "pid", None)
        return any(vid == want_vid and (want_pid is None or pid == want_pid)
                   for want_vid, want_pid in self.EXCLUDE_VID_PID)

    def find_port(self):
        for port in serial.tools.list_ports.comports():
            if port.manufacturer and "FTDI" in port.manufacturer:
                return port.device
        for port in serial.tools.list_ports.comports():
            if self._excluded(port):
                continue
            if "ttyUSB" in port.device:
                return port.device
        return None

    def open(self, port=None):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

        self.port = port or self.find_port()
        if self.port is None:
            raise DeviceError("Cyton ドングル（FTDI）が見つかりません")
        set_latency_timer(1)

        params = BrainFlowInputParams()
        params.serial_port = self.port
        self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        try:
            self.board.prepare_session()
            self.board.start_stream()
        except Exception as exc:      # noqa: BLE001 - そのままUIに出す
            self.board = None
            raise DeviceError(f"{type(exc).__name__}: {exc}") from exc

        board_id = self.board.get_board_id()
        self._eeg_rows = BoardShim.get_eeg_channels(board_id)[:self.channel_num]
        self._ts_row = BoardShim.get_timestamp_channel(board_id)
        self._pkg_row = BoardShim.get_package_num_channel(board_id)
        return self.port

    def close(self):
        if self.board is not None:
            try:
                self.board.stop_stream()
                self.board.release_session()
            except Exception:      # noqa: BLE001
                pass
            self.board = None

    def read(self):
        data = self.board.get_board_data()
        if data.size == 0:
            return None
        eeg = data[self._eeg_rows, :].astype(np.float32)     # µV
        stamps = data[self._ts_row, :]
        pkgs = data[self._pkg_row, :]
        return eeg, stamps, pkgs, None

    def pause(self):
        self.board.stop_stream()

    def resume(self):
        self.board.get_board_data()      # 溜まった分を捨てる
        self.board.start_stream()

    def measure_impedance(self, channels, on_channel=None, on_value=None):
        from impedance_check import ImpedanceCheck

        checker = ImpedanceCheck(self.port)
        checker.board = self.board        # 既存セッションを共有する
        values = {}
        try:
            for ch in channels:
                if on_channel:
                    on_channel(ch)
                try:
                    value = f"{checker.check_impedance(ch):.1f}"
                except Exception as exc:      # noqa: BLE001
                    value = "ERR"
                    print(f"[imp] ch{ch}: {exc}")
                values[ch] = value
                if on_value:
                    on_value(ch, value)       # 1ch終わるたびに画面へ出す
        finally:
            try:
                checker.reset_to_defaults(self.board)
            except Exception:      # noqa: BLE001
                pass
        return values


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


# =====================================================================
# Polymate Pocket MP208（1000 Hz・EXT 2ch で刺激音そのものを同時記録）
# =====================================================================
class PolymateBackend(DeviceBackend):
    name = POLYMATE
    label = "Polymate Pocket (MP208)"
    sampling_rate = 1000
    n_ext = 2
    ext_names = ("EXT1", "EXT2")
    needs_port = True           # COM番号を指定できる（空欄なら総当たりで探す）
    read_interval = 0.001       # pp.nums() を細かく見る
    supports_impedance = True
    supports_neutsuri = False   # 押下判定モデルが 250 Hz 学習なので出さない

    # API 本体（pocket.pyd）の探索先。実行中の Python のバージョンに合うものを使う。
    API_SUBDIRS = ("reference/PythonAPI_PolymatePocket/PolymatePocket/py{v}",
                   "PythonAPI_PolymatePocket/PolymatePocket/py{v}",
                   "polymate_api/py{v}")

    def __init__(self, channel_num=8, base_dir=None):
        super().__init__()
        self.channel_num = channel_num
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.n_total = channel_num + self.n_ext
        self._pp = None
        self._h = None

    # --- API の読み込み ---
    def _api_paths(self):
        v = f"{sys.version_info.major}{sys.version_info.minor}"
        for pattern in self.API_SUBDIRS:
            for root in (self.base_dir, os.path.dirname(self.base_dir)):
                yield os.path.join(root, pattern.format(v=v))

    def _load_api(self):
        """polymate.pocket を読み込む（成功したら self._pp に入れて返す）。"""
        if self._pp is not None:
            return self._pp
        tried = []
        for path in self._api_paths():
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
            tried.append(path)
        try:
            import polymate.pocket as pp
        except Exception as exc:      # noqa: BLE001
            raise DeviceError(
                f"Polymate API を読み込めません（{type(exc).__name__}: {exc}）。"
                f"pocket.pyd は Windows 専用です。探した場所: "
                + ", ".join(os.path.relpath(p, self.base_dir) for p in tried)
            ) from exc
        self._pp = pp
        return pp

    def availability(self):
        try:
            self._load_api()
        except DeviceError as exc:
            return False, str(exc)
        return True, ""

    # --- 接続 ---
    def list_ports(self):
        return [{"port": self._port_number(p.device) or p.device,
                 "description": (p.description or "")}
                for p in serial.tools.list_ports.comports()]

    @staticmethod
    def _port_number(name):
        """"COM4" / "4" / "/dev/ttyUSB0" → API に渡すポート番号の文字列。"""
        m = re.search(r"(\d+)\s*$", str(name).strip())
        return m.group(1) if m else None

    def _candidate_ports(self):
        """総当たり接続の候補。OSが見えているCOM番号を若い順に返す。"""
        nums = []
        for p in serial.tools.list_ports.comports():
            n = self._port_number(p.device)
            if n and n not in nums:
                nums.append(n)
        return sorted(nums, key=int)

    def _create_handle(self, pp):
        h = pp.create(self.sampling_rate, self.n_total, self.sampling_rate * 10)
        for ch in range(1, self.channel_num + 1):
            pp.setup_ele(h, ch, "ELE", ch, 1000, 1000)
            pp.setup_unit(h, ch, "uV")
        # EXT1/EXT2 にステレオの刺激音そのものを入れておき、オンセットを
        # 音響波形から取れるようにする（Label 列に頼らない）。
        pp.setup_ele(h, self.channel_num + 1, "EXT", 1, 1, 0, "ON")
        pp.setup_ele(h, self.channel_num + 2, "EXT", 2, 1, 0, "OF")
        return h

    def _try_port(self, pp, port, probe_seconds=1.0):
        """1つのCOM番号を開いて、実際にサンプルが流れてくるかまで確かめる。

        開けただけでは足りない。他機種のUSBシリアルでも open が通ってしまうことが
        あるので、start してデータが来ることを確認できたポートだけ採用する。
        """
        h = self._create_handle(pp)
        if not pp.open(h, "COM", str(port)):
            pp.release(h)
            return None
        try:
            pp.start(h)
        except Exception:      # noqa: BLE001
            pp.close(h)
            pp.release(h)
            return None

        deadline = time.time() + probe_seconds
        while time.time() < deadline:
            try:
                if pp.nums(h) > 0:
                    return h
            except Exception:      # noqa: BLE001
                break
            time.sleep(0.02)

        try:
            pp.stop(h)
            pp.close(h)
        except Exception:      # noqa: BLE001
            pass
        pp.release(h)
        return None

    def open(self, port=None):
        pp = self._load_api()

        num = self._port_number(port) if port else None
        candidates = [num] if num else self._candidate_ports()
        if not candidates:
            raise DeviceError("COMポートが1つも見つかりません")

        tried = []
        for cand in candidates:
            h = self._try_port(pp, cand)
            if h is not None:
                self._h = h
                self.port = cand
                return cand
            tried.append(f"COM{cand}")

        if num:
            raise DeviceError(f"COM{num} に接続できませんでした")
        raise DeviceError("Polymate が見つかりません（試したポート: "
                          + ", ".join(tried) + "）")

    def close(self):
        if self._h is not None:
            pp = self._pp
            try:
                pp.stop(self._h)
                pp.close(self._h)
                pp.release(self._h)
            except Exception:      # noqa: BLE001
                pass
            self._h = None

    def read(self):
        pp = self._pp
        n = pp.nums(self._h)
        if n < 10:
            return None
        data, actual = pp.data(self._h, n)
        if actual <= 0:
            return None
        # data: (actual, n_total) — 先頭 channel_num 列が EEG(µV)、残りが EXT
        eeg = data[:actual, :self.channel_num].T.astype(np.float32)
        ext = data[:actual, self.channel_num:self.n_total].astype(np.float32)
        # API は到着時刻を持たないので、末尾を「今」として等間隔に振る。
        dt = 1.0 / self.sampling_rate
        t_now = time.time()
        stamps = t_now - (actual - 1 - np.arange(actual)) * dt
        return eeg, stamps, None, ext

    def pause(self):
        self._pp.stop(self._h)

    def resume(self):
        self._pp.start(self._h)

    def measure_impedance(self, channels, on_channel=None, on_value=None):
        """Polymate は全チャンネルを一括で測る（channels は無視する）。"""
        pp = self._pp
        pp.start_impd(self._h)
        try:
            impd, result = pp.data_impd(self._h, 5000)
        finally:
            try:
                pp.stop_impd(self._h)
            except Exception:      # noqa: BLE001
                pass
        if result != 1:
            raise DeviceError(f"インピーダンス計測に失敗しました (result={result})")
        print(f"[imp] raw impd (kohm) = {impd}")
        values = {ch + 1: f"{float(impd[ch]):.1f}" for ch in range(self.channel_num)}
        if on_value:
            for ch, value in values.items():
                on_value(ch, value)
        return values


# =====================================================================
# レジストリ
# =====================================================================
def build_backends(channel_num=8, base_dir=None, exclude_vid_pid=()):
    """機種名 → バックエンドの辞書を作る（起動時に1回だけ）。"""
    return {
        CYTON: CytonBackend(channel_num, exclude_vid_pid),
        POLYMATE: PolymateBackend(channel_num, base_dir),
    }
