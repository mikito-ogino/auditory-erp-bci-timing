# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.signal import iirfilter, filtfilt, iirnotch

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, sosfilt
from scipy.signal import iirfilter, filtfilt

class ImpedanceCheck:

    # --- Constants (GUI-based settings) ---
    SERIES_R = 2200.0          # Keep series resistor at 2.2 kΩ
    I_DRIVE = 6.0e-9           # Use the same value as GUI's leadOffDrive_amps (default: 6 nA)
    MEAS_SEC = 6.0             # Measurement duration: 6s, but only the last 1s will be used
    BAND = (5, 50)             # Bandpass filter (Hz), matches OpenBCI_GUI default
    MAINS_HZ = 50.0            # Mains notch frequency (East Japan = 50 Hz; use 60 for 60 Hz regions)
    CABLE_COLORS = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
    board = None
    fs = 250                    # Default sampling rate for Cyton

    def __init__(self, port):
        self.impedance_data = []
        self.port = port

        # --- Cyton channel config presets (to mimic OpenBCI_GUI impedance behavior) ---
        # Cyton SDK channel settings: x(CH, POWER_DOWN, GAIN, INPUT, BIAS, SRB2, SRB1)X
        # gain: 0..6 (1,2,4,6,8,12,24), input_type: 0=normal, bias: 0/1, srb2: 0/1, srb1: 0/1
        self.STREAM_CFG_DEFAULT = dict(gain=6, input_type=0, bias=1, srb2=1, srb1=0)  # normal preset (gain 24x)
        # OpenBCI_GUI does NOT change channel settings for impedance — it keeps the
        # normal config (gain 24x, srb2=1) and only toggles the lead-off (z) bits.
        # (The old preset gain=1x/srb2=0 railed the amplifier and inflated readings ~7x.)
        self.IMP_CFG = dict(gain=6, input_type=0, bias=1, srb2=1, srb1=0)


        # Track what this app last set per channel so we can revert on toggle-off
        self._ch_cfg = [self.STREAM_CFG_DEFAULT.copy() for _ in range(8)]
        self._ch_last_cfg = [None for _ in range(8)]

    def build_channel_settings_cmd(self, ch: int, gain: int, input_type: int, bias: int, srb2: int, srb1: int, power_down: int = 0) -> str:
        """Build Cyton channel settings command: x(CH, POWER_DOWN, GAIN, INPUT, BIAS, SRB2, SRB1)X"""
        return f"x{ch}{power_down}{gain}{input_type}{bias}{srb2}{srb1}X"

    def build_impedance_cmd(self, ch: int, active: bool, is_n: bool) -> str:
        """Build Cyton impedance/lead-off command: z(CH, PCHAN, NCHAN)Z"""
        p = "0"
        n = "0"
        if active:
            if is_n:
                n = "1"
            else:
                p = "1"
        return f"z{ch}{p}{n}Z"

    def reset_to_defaults(self, board: BoardShim):
        try:
            board.config_board("d")
            time.sleep(0.1)
            return True, "OK"
        except Exception as e:
            return False, f"ERR: {e}"

    def calc_impedance_from_vrms(self, vrms_uV):
        Vrms_V = float(vrms_uV) * 1e-6
        Z = (np.sqrt(2.0) * Vrms_V) / self.I_DRIVE
        Z -= self.SERIES_R
        return max(Z, 0.0)

    def take_recent_1s(self, x_uV, fs):
        n = int(fs * 1.0)
        seg = x_uV[-n:]
        return float(np.sqrt(np.mean(seg ** 2)))

    def bandpass_apply(self, x, fs):
        b, a = iirfilter(4, [self.BAND[0]/(fs/2.0), self.BAND[1]/(fs/2.0)],
                        btype='band', ftype='butter')
        return filtfilt(b, a, x)

    def gui_std_vrms(self, x_uV, fs):
        """OpenBCI_GUI-faithful Vrms = std-dev of the last 1 s after bandpass +
        mains notch (this is the GUI's data_std_uV). No lock-in."""
        x = np.asarray(x_uV, dtype=float)
        if x.size < int(fs):
            return float('nan')
        y = self.bandpass_apply(x, fs)
        w0 = self.MAINS_HZ / (fs / 2.0)
        if 0.0 < w0 < 1.0:
            bn, an = iirnotch(w0, 30.0)
            y = filtfilt(bn, an, y)
        return float(np.std(y[-int(fs):]))

    def change_leadoff(self, ch, is_on):
        ch_idx = ch - 1  # zero-based index
        is_n = True

        if is_on:
            self._ch_last_cfg[ch_idx] = self._ch_cfg[ch_idx].copy()
            self._ch_cfg[ch_idx] = self.IMP_CFG.copy()
        else:
            if self._ch_last_cfg[ch_idx] is not None:
                self._ch_cfg[ch_idx] = self._ch_last_cfg[ch_idx].copy()
                self._ch_last_cfg[ch_idx] = None
            else:
                self._ch_cfg[ch_idx] = self.STREAM_CFG_DEFAULT.copy()

        cfg = self._ch_cfg[ch_idx]
        x_cmd = self.build_channel_settings_cmd(
            ch=ch,
            gain=cfg["gain"],
            input_type=cfg["input_type"],
            bias=cfg["bias"],
            srb2=cfg["srb2"],
            srb1=cfg["srb1"],
            power_down=0
        )
        z_cmd = self.build_impedance_cmd(ch=ch, active=is_on, is_n=is_n)

        cmd = x_cmd + z_cmd

        try:
            resp = self.board.config_board(cmd)
            print(f"Ch{ch} Cmd: {cmd} | Resp: {resp}")
        except UnicodeDecodeError:
            print(f"Ch{ch} Cmd: {cmd} | (Success but response decode error)")

        time.sleep(0.1)

    def prepare_board(self):
        global board
        port = self.port
        if not port:
            raise RuntimeError("Cyton serial port not found.")

        params = BrainFlowInputParams()
        params.serial_port = port
        board_id = BoardIds.CYTON_BOARD.value
        self.fs = BoardShim.get_sampling_rate(board_id)

        self.board = BoardShim(board_id, params)

        self.board.prepare_session()

    def stop_board(self):
        self.reset_to_defaults(self.board)
        try:
            self.board.release_session()
        except Exception:
            pass

    def check_impedance(self, ch):
        self.reset_to_defaults(self.board)
        self.change_leadoff(ch, True)
        self.board.start_stream()
        self.board.get_board_data()
        time.sleep(self.MEAS_SEC+0.2)
        data = self.board.get_board_data()
        eeg_idxs = BoardShim.get_eeg_channels(self.board.get_board_id())
        row = eeg_idxs[ch - 1]
        x_uV = data[row, :]
        gui_rms = self.gui_std_vrms(x_uV, self.fs)  # GUI data_std_uV: std(bandpass+notch), last 1 s
        z_ohm = self.calc_impedance_from_vrms(gui_rms) if np.isfinite(gui_rms) else float('nan')
        z_kohm = z_ohm / 1000.0
        # --- DIAG ---
        _fs = int(self.fs)
        _raw = x_uV[-_fs:] if x_uV.size else np.array([0.0])
        raw_rms = float(np.sqrt(np.mean(_raw ** 2)))
        _xmax = float(np.max(x_uV)) if x_uV.size else 0.0
        _n_rail = int(np.sum(_raw >= 0.99 * _xmax)) if _xmax > 0 else 0
        print(f"[DIAG] CH{ch} fs={self.fs} n={data.shape[1]} raw_uVrms={raw_rms:.1f} "
              f"rail%={100.0 * _n_rail / len(_raw):.1f} std_uV={gui_rms:.2f} -> {z_kohm:.2f} kΩ")
        # --- end DIAG ---
        self.board.stop_stream()
        self.change_leadoff(ch, False)

        return z_kohm
