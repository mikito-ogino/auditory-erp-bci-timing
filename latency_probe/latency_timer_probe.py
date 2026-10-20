#!/usr/bin/env python
"""
FTDI latency-timer probe for the OpenBCI Cyton dongle (FT231X).

Purpose
-------
macOS (Apple's built-in AppleUSBFTDI driver) exposes no API to read the FTDI
latency timer.  The only way to confirm its effective value is to observe how
serial data *arrives* at user space while the Cyton streams.  This script
streams for a fixed time, timestamps every read that returns data, and infers
the timer setting from the chunk-size / inter-arrival pattern.

Serial handshake (follows pyOpenBCI 0.13, cyton.py)
----------------------------------------------------
    Serial(port, baudrate=115200, timeout=...)   # cyton.py L53
    sleep(2.0)                                   # L58  let the dongle settle
    write(b'v')                                  # L59  soft reset
    read until b'$$$' (board banner)             # find_port L101-104
    write(b'b')                                  # start_stream L252
    ...
    write(b's')                                  # stop_stream L117
    close()

Chunk capture
-------------
The spec called for read(in_waiting or 1) with timeout 0.001 s.  With pyserial
3.5 on POSIX, read(1) returns as soon as *one* byte is readable, so every USB
transfer would be split into a 1-byte chunk followed by an (n-1)-byte chunk,
corrupting the size distribution.  To avoid that, each iteration waits with
select() (<= 1 ms) until the descriptor is readable, then reads *everything*
reported by in_waiting in one read().  One such read == one chunk, stamped
with time.perf_counter_ns() immediately after select() returns.

Inference logic
---------------
The FT231X pushes a USB IN transfer when EITHER (a) the latency timer expires
with data pending, OR (b) its receive FIFO holds 62 payload bytes (64-byte
bulk packet minus 2 modem-status bytes) - whichever comes first.  The Cyton
emits one 33-byte packet every 4 ms (250 Hz, ~8.25 kB/s on average).  On the
115200-baud UART link a packet takes 33*10/115200 = 2.9 ms to arrive, then the
line idles ~1.1 ms until the next packet.  62 bytes therefore accumulate in
about 7.5 ms at the average rate (never faster than 5.4 ms at pure line rate).

Expected arrival patterns:

  * Latency timer = 1 ms
        The timer fires long before 62 B accumulate.  Chunks are <= 33 B
        (typically ~11-33 B), and arrivals come at ~1 ms steps while a packet
        is on the wire, or once per 4-ms packet if the driver merges them.
        Signature: chunk mode <= 33 B, most intervals in 0.5-5.5 ms.

  * Latency timer = 16 ms (FTDI default; "not configured")
        62 B fill (~7.5 ms) always beats the 16-ms timer, so transfers are
        ~62 B at ~7.5-ms spacing.  A pure 16-ms period is NOT expected.
        Signature: chunk mode in 56-64 B, most intervals in 5.5-9.5 ms.

  * Anything else (e.g. chunks of hundreds of bytes, periods of tens of ms)
        indicates that host-side buffering (AppleUSBFTDI / IOKit read
        coalescing, scheduler latency) or pyserial's read granularity, not the
        FTDI timer, dominates what user space sees.  The script then reports
        "UNDETERMINED" with the reasons, instead of guessing.

The verdict is one of: "1 ms-equivalent", "16 ms-equivalent", "UNDETERMINED",
always printed together with the numbers it was based on.

Outputs
-------
  stdout                      summary statistics + verdict
  latency_probe/probe_YYYYMMDD_HHMMSS.csv
                              one row per chunk: index, t_rel_ms, interval_ms,
                              n_bytes, in_analysis (0 during warm-up)

Usage
-----
  ~/.pyenv/versions/pyopenbci-env/bin/python latency_probe/latency_timer_probe.py
  ... --duration 20 --port /dev/cu.usbserial-DP04VZU5
"""

import argparse
import csv
import datetime as _dt
import fcntl
import os
import struct
import select
import subprocess
import sys
import time

import numpy as np
import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ftdi_latency import FtdiLatency      # libusb/ctypes helper (same folder)
except Exception:                              # libusb missing -> option unavailable
    FtdiLatency = None

DEFAULT_PORT = "/dev/cu.usbserial-DP04VZU5"
BAUD = 115200                 # cyton.py: baud=115200
PACKET_LEN = 33               # Cyton packet: 0xA0, id, 24 B EEG, 6 B aux, 0xCx
START_BYTE = 0xA0
END_BYTE_MIN, END_BYTE_MAX = 0xC0, 0xCF
NOMINAL_FS = 250.0
UART_BYTES_PER_MS = BAUD / 10.0 / 1000.0      # 11.52 B/ms (8N1)
FTDI_FIFO_PAYLOAD = 62

# macOS IOSerialFamily ioctl: IOSSDATALAT = _IOW('T', 0, unsigned long)
# (IOKit/serial/ioss.h: "Sets the receive latency (in microseconds)").
# _IOW = IOC_IN | (sizeof(t) << 16) | (g << 8) | n  with IOC_IN = 0x80000000
IOSSDATALAT = 0x80000000 | (8 << 16) | (ord("T") << 8) | 0

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# pre-flight
# --------------------------------------------------------------------------- #
def check_port_free(port: str) -> None:
    """Abort if another process (main.py, OpenBCI_GUI, ...) holds the port."""
    variants = {port}
    if port.startswith("/dev/cu."):
        variants.add(port.replace("/dev/cu.", "/dev/tty.", 1))
    elif port.startswith("/dev/tty."):
        variants.add(port.replace("/dev/tty.", "/dev/cu.", 1))
    try:
        res = subprocess.run(["lsof", "-t", *sorted(variants)],
                             capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        print("[pre-flight] lsof not found; skipping port-holder check")
        return
    except subprocess.TimeoutExpired:
        print("[pre-flight] lsof timed out; skipping port-holder check")
        return
    pids = res.stdout.split()
    if pids:
        print(f"[pre-flight] ERROR: port {port} is held by PID(s) {', '.join(pids)}")
        ps = subprocess.run(["ps", "-o", "pid=,comm=,args=", "-p", ",".join(pids)],
                            capture_output=True, text=True)
        print(ps.stdout.rstrip())
        print("Quit main.py / OpenBCI_GUI (or any other reader) and retry.")
        sys.exit(1)
    # informational: likely culprits running even if they do not hold the port yet
    pg = subprocess.run(["pgrep", "-fl", "main.py|OpenBCI_GUI|offline_test|impedance_check"],
                        capture_output=True, text=True)
    hits = [l for l in pg.stdout.splitlines() if "latency_timer_probe" not in l]
    if hits:
        print("[pre-flight] WARNING: possibly related processes are running:")
        for l in hits:
            print("   ", l)
    print(f"[pre-flight] port {port} is free")


# --------------------------------------------------------------------------- #
# serial helpers (pyOpenBCI handshake)
# --------------------------------------------------------------------------- #
def read_until(ser: serial.Serial, marker: bytes, timeout_s: float) -> bytes:
    buf = bytearray()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        n = ser.in_waiting
        data = ser.read(n if n else 1)
        if data:
            buf += data
            if marker in buf:
                break
    return bytes(buf)


def set_data_latency(ser: serial.Serial, latency_us: int) -> bool:
    """Ask the serial driver for a receive latency of latency_us microseconds
    via IOSSDATALAT.  Returns True if the ioctl was accepted.  Whether the
    driver actually honours it must be verified from the arrival pattern."""
    try:
        fcntl.ioctl(ser.fileno(), IOSSDATALAT, struct.pack("L", latency_us))
    except OSError as e:
        print(f"[serial] IOSSDATALAT({latency_us} us) rejected: {e}")
        return False
    print(f"[serial] IOSSDATALAT({latency_us} us) accepted by driver")
    return True


def open_and_reset(port: str, latency_us=None) -> serial.Serial:
    ser = serial.Serial(port=port, baudrate=BAUD, timeout=0.1)
    print(f"[serial] opened {port} @ {BAUD} baud")
    if latency_us is not None:
        set_data_latency(ser, latency_us)
    time.sleep(2.0)                    # cyton.py L58
    ser.reset_input_buffer()
    ser.write(b"v")                    # cyton.py L59 soft reset
    banner = read_until(ser, b"$$$", timeout_s=6.0)
    text = banner.decode("utf-8", errors="replace")
    print("[serial] reset banner:")
    for line in text.replace("$$$", "").splitlines():
        if line.strip():
            print("   |", line.rstrip())
    if b"$$$" not in banner:
        print("[serial] ERROR: no '$$$' after 'v' (board off / out of range / wrong port?)")
        ser.close()
        sys.exit(2)
    if b"Failure" in banner:
        print("[serial] ERROR: dongle reports failure talking to the board")
        ser.close()
        sys.exit(2)
    return ser


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
def capture(ser: serial.Serial, duration_s: float):
    """Stream for duration_s; return (t_ns list, n_bytes list, raw bytes, t0_ns)."""
    ser.timeout = 0.001                # spec: tiny timeout so read() never blocks
    fd = ser.fileno()
    t_ns, n_bytes = [], []
    raw = bytearray()

    ser.reset_input_buffer()
    ser.write(b"b")                    # cyton.py L252 start stream
    t0 = time.perf_counter_ns()
    deadline = t0 + int(duration_s * 1e9)
    try:
        while True:
            now = time.perf_counter_ns()
            if now >= deadline:
                break
            ready, _, _ = select.select([fd], [], [], 0.001)
            if not ready:
                continue
            t = time.perf_counter_ns()
            n = ser.in_waiting
            if n <= 0:
                n = 1
            data = ser.read(n)         # returns immediately: n bytes are queued
            if not data:
                continue
            t_ns.append(t)
            n_bytes.append(len(data))
            raw += data
    finally:
        ser.write(b"s")                # cyton.py L117 stop stream (always)
        time.sleep(0.3)
        try:
            ser.reset_input_buffer()
        finally:
            ser.close()
        print("[serial] stream stopped ('s') and port closed")
    return t_ns, n_bytes, bytes(raw), t0


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def count_packets(raw: bytes):
    """Count 33-byte Cyton packets (0xA0 ... 0xC0-0xCF). Return (n, ids, n_skipped_bytes)."""
    n, i, skipped = 0, 0, 0
    ids = []
    L = len(raw)
    while i + PACKET_LEN <= L:
        if raw[i] == START_BYTE and END_BYTE_MIN <= raw[i + PACKET_LEN - 1] <= END_BYTE_MAX:
            ids.append(raw[i + 1])
            n += 1
            i += PACKET_LEN
        else:
            i += 1
            skipped += 1
    return n, ids, skipped


def dropped_from_ids(ids):
    if len(ids) < 2:
        return 0
    a = np.asarray(ids, dtype=int)
    d = np.diff(a) % 256
    return int(np.sum(d - 1))        # d == 1 means consecutive


def text_hist(values, edges, labels, width=50):
    counts, _ = np.histogram(values, bins=edges)
    peak = counts.max() if counts.size and counts.max() > 0 else 1
    lines = []
    for lab, c in zip(labels, counts):
        bar = "#" * int(round(width * c / peak))
        lines.append(f"  {lab:>10s} | {c:6d} | {bar}")
    return "\n".join(lines), counts


def analyse(t_ns, n_bytes, raw, t0, warmup_s: float):
    t_ns = np.asarray(t_ns, dtype=np.int64)
    n_bytes = np.asarray(n_bytes, dtype=int)
    t_rel_ms = (t_ns - t0) / 1e6
    in_analysis = t_rel_ms >= warmup_s * 1e3

    out = {}
    out["n_chunks_total"] = int(len(t_ns))
    out["n_bytes_total"] = int(n_bytes.sum())
    out["span_s"] = float((t_ns[-1] - t_ns[0]) / 1e9) if len(t_ns) > 1 else 0.0

    # packets / sample rate on the full capture
    n_pkt, ids, skipped = count_packets(raw)
    out["n_packets"] = n_pkt
    out["bytes_skipped"] = skipped
    out["dropped_ids"] = dropped_from_ids(ids)
    out["fs_eff"] = n_pkt / out["span_s"] if out["span_s"] > 0 else float("nan")
    out["bytes_per_s"] = out["n_bytes_total"] / out["span_s"] if out["span_s"] > 0 else float("nan")

    sel = in_analysis
    nb = n_bytes[sel]
    tt = t_rel_ms[sel]
    iv = np.diff(tt) if len(tt) > 1 else np.array([])
    out["n_chunks_analysed"] = int(len(nb))
    out["n_intervals"] = int(len(iv))

    if len(nb):
        vals, cnts = np.unique(nb, return_counts=True)
        order = np.argsort(-cnts)
        out["size_mode"] = int(vals[order[0]])
        out["size_mode_frac"] = float(cnts[order[0]] / len(nb))
        out["size_top"] = [(int(vals[k]), int(cnts[k])) for k in order[:10]]
        out["frac_le33"] = float(np.mean(nb <= 33))
        out["frac_56_64"] = float(np.mean((nb >= 56) & (nb <= 64)))
        out["frac_gt100"] = float(np.mean(nb > 100))
        out["size_median"] = float(np.median(nb))
    if len(iv):
        q = np.percentile(iv, [5, 25, 50, 75, 95])
        out["iv_p5"], out["iv_p25"], out["iv_med"], out["iv_p75"], out["iv_p95"] = map(float, q)
        out["iv_max"] = float(iv.max())
        out["frac_iv_0p5_5p5"] = float(np.mean((iv >= 0.5) & (iv < 5.5)))
        out["frac_iv_5p5_9p5"] = float(np.mean((iv >= 5.5) & (iv < 9.5)))
        out["frac_iv_ge12"] = float(np.mean(iv >= 12.0))
        out["frac_iv_lt0p5"] = float(np.mean(iv < 0.5))
    return out, t_rel_ms, in_analysis, nb, iv


def verdict(o):
    """Return (label, reasons list). See module docstring for the logic."""
    reasons = []
    if o.get("n_intervals", 0) < 100:
        reasons.append(f"too few chunks for inference ({o.get('n_intervals', 0)} intervals)")
        return "UNDETERMINED", reasons

    fs_ok = 240.0 <= o["fs_eff"] <= 260.0
    if not fs_ok:
        reasons.append(f"effective sample rate {o['fs_eff']:.1f} Hz is outside 240-260 Hz; "
                       f"stream not nominal, inference unreliable")

    mode = o["size_mode"]
    c1 = (mode <= 33, f"chunk mode {mode} B <= 33 B")
    c2 = (o["frac_le33"] >= 0.70, f"{o['frac_le33']*100:.0f}% of chunks <= 33 B (need >= 70%)")
    c3 = (o["frac_iv_0p5_5p5"] >= 0.70,
          f"{o['frac_iv_0p5_5p5']*100:.0f}% of intervals in 0.5-5.5 ms (need >= 70%)")
    one_ms = all(c for c, _ in (c1, c2, c3))

    d1 = (56 <= mode <= 64, f"chunk mode {mode} B in 56-64 B")
    d2 = (o["frac_56_64"] >= 0.50, f"{o['frac_56_64']*100:.0f}% of chunks in 56-64 B (need >= 50%)")
    d3 = (o["frac_iv_5p5_9p5"] >= 0.50,
          f"{o['frac_iv_5p5_9p5']*100:.0f}% of intervals in 5.5-9.5 ms (need >= 50%)")
    sixteen_ms = all(c for c, _ in (d1, d2, d3))

    if one_ms and fs_ok and not sixteen_ms:
        reasons += [m for _, m in (c1, c2, c3)]
        reasons.append(f"median interval {o['iv_med']:.2f} ms; 62-B FIFO fill (~7.5 ms) never reached "
                       f"-> timer fires first -> latency timer ~1 ms")
        return "1 ms-equivalent", reasons
    if sixteen_ms and fs_ok and not one_ms:
        reasons += [m for _, m in (d1, d2, d3)]
        reasons.append(f"median interval {o['iv_med']:.2f} ms; transfers are governed by the 62-B FIFO "
                       f"fill, i.e. the timer is longer than ~7.5 ms -> consistent with the 16 ms default")
        return "16 ms-equivalent", reasons

    # undetermined: explain what was seen
    reasons.append(f"1-ms checks: " + "; ".join(("OK " if c else "FAIL ") + m for c, m in (c1, c2, c3)))
    reasons.append(f"16-ms checks: " + "; ".join(("OK " if c else "FAIL ") + m for c, m in (d1, d2, d3)))
    if o["frac_gt100"] >= 0.30 or mode > 100:
        reasons.append(f"{o['frac_gt100']*100:.0f}% of chunks exceed 100 B (mode {mode} B): larger than any "
                       f"single FT231X transfer (<= 62 B) -> host-side coalescing (AppleUSBFTDI/IOKit "
                       f"buffering or process scheduling) hides the FTDI timer")
    if o["frac_iv_ge12"] >= 0.30:
        reasons.append(f"{o['frac_iv_ge12']*100:.0f}% of intervals >= 12 ms: longer than both the 62-B fill "
                       f"(~7.5 ms) and the 1-ms timer -> arrival cadence set by the host, not the FTDI timer")
    if o["frac_iv_lt0p5"] >= 0.30:
        reasons.append(f"{o['frac_iv_lt0p5']*100:.0f}% of intervals < 0.5 ms: reads are splitting single "
                       f"transfers (pyserial/driver read granularity), size statistics unreliable")
    if one_ms and sixteen_ms:
        reasons.append("both signatures matched simultaneously - thresholds too loose for this capture")
    return "UNDETERMINED", reasons


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--duration", type=float, default=10.0, help="stream capture time [s] (default 10)")
    ap.add_argument("--warmup", type=float, default=0.5,
                    help="seconds at stream start excluded from statistics (default 0.5)")
    ap.add_argument("--skip-port-check", action="store_true")
    ap.add_argument("--outdir", default=HERE, help="directory for the CSV (default: this folder)")
    ap.add_argument("--ftdi-latency-ms", type=int, default=None, metavar="MS",
                    help="write this value (1-255) into the FT231X latency timer over USB "
                         "(libusb, bypassing the Apple driver) after opening the port, then probe")
    ap.add_argument("--set-latency-us", type=int, default=None, metavar="US",
                    help="after opening, request this receive latency [us] via the macOS "
                         "IOSSDATALAT ioctl (e.g. 1000); the verdict then shows whether the "
                         "driver honoured it")
    args = ap.parse_args()

    print("=" * 72)
    print("FTDI latency-timer probe  (OpenBCI Cyton / FT231X)")
    print(f"  {_dt.datetime.now().isoformat(timespec='seconds')}  port={args.port}  "
          f"duration={args.duration:g}s  warmup={args.warmup:g}s  "
          f"set_latency_us={args.set_latency_us}  ftdi_latency_ms={args.ftdi_latency_ms}")
    print("=" * 72)

    if not args.skip_port_check:
        check_port_free(args.port)

    ser = open_and_reset(args.port, args.set_latency_us)

    chip_latency = None
    if FtdiLatency is not None:
        try:
            dev = FtdiLatency()
            before = dev.get()
            if args.ftdi_latency_ms is not None:
                dev.set(args.ftdi_latency_ms)
            chip_latency = dev.get()
            dev.close()
            print(f"[ftdi] chip latency timer read back over USB: {before} ms"
                  + (f" -> {chip_latency} ms" if args.ftdi_latency_ms is not None else ""))
        except SystemExit as e:
            print(f"[ftdi] USB readback unavailable: {e}")
        except OSError as e:
            print(f"[ftdi] USB readback failed: {e}")
    elif args.ftdi_latency_ms is not None:
        print("[ftdi] --ftdi-latency-ms ignored: libusb helper not importable")
    print(f"[capture] streaming for {args.duration:g} s ...")
    t_ns, n_bytes, raw, t0 = capture(ser, args.duration)

    if len(t_ns) == 0:
        print("[capture] ERROR: no data received after 'b'")
        sys.exit(3)

    o, t_rel_ms, in_analysis, nb, iv = analyse(t_ns, n_bytes, raw, t0, args.warmup)

    # ---- CSV --------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"probe_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_idx", "t_rel_ms", "interval_ms", "n_bytes", "in_analysis"])
        prev = None
        for i, (t, n, a) in enumerate(zip(t_rel_ms, n_bytes, in_analysis)):
            ivl = "" if prev is None else f"{t - prev:.4f}"
            w.writerow([i, f"{t:.4f}", ivl, int(n), int(a)])
            prev = t

    # ---- report -----------------------------------------------------------
    print()
    print("---- stream sanity ------------------------------------------------")
    print(f"  chunks (all)            : {o['n_chunks_total']}")
    print(f"  bytes  (all)            : {o['n_bytes_total']}  ({o['bytes_per_s']:.0f} B/s; "
          f"nominal {PACKET_LEN*NOMINAL_FS:.0f} B/s)")
    print(f"  capture span            : {o['span_s']:.3f} s")
    print(f"  33-B packets (0xA0..0xC?): {o['n_packets']}  -> effective fs = {o['fs_eff']:.2f} Hz "
          f"(nominal {NOMINAL_FS:.0f})")
    print(f"  dropped packets (by id) : {o['dropped_ids']}   bytes skipped in parse: {o['bytes_skipped']}")
    print(f"  chunks analysed (t >= {args.warmup:g}s): {o['n_chunks_analysed']}  "
          f"intervals: {o['n_intervals']}")

    print()
    print("---- chunk size (bytes per read) ----------------------------------")
    if len(nb):
        print(f"  mode   : {o['size_mode']} B  ({o['size_mode_frac']*100:.1f}% of chunks)")
        print(f"  median : {o['size_median']:.0f} B")
        print(f"  <=33 B : {o['frac_le33']*100:.1f}%    56-64 B : {o['frac_56_64']*100:.1f}%    "
              f">100 B : {o['frac_gt100']*100:.1f}%")
        print("  top sizes (B: count): " + ", ".join(f"{v}:{c}" for v, c in o["size_top"]))
        edges = [0, 12, 23, 34, 45, 56, 65, 100, 200, 400, 10**9]
        labels = ["1-11", "12-22", "23-33", "34-44", "45-55", "56-64", "65-99", "100-199", "200-399", ">=400"]
        h, _ = text_hist(nb, edges, labels)
        print(h)

    print()
    print("---- inter-arrival interval (ms) ----------------------------------")
    if len(iv):
        print(f"  median : {o['iv_med']:.3f} ms   p5 {o['iv_p5']:.3f}  p25 {o['iv_p25']:.3f}  "
              f"p75 {o['iv_p75']:.3f}  p95 {o['iv_p95']:.3f}  max {o['iv_max']:.1f}")
        print(f"  <0.5 ms : {o['frac_iv_lt0p5']*100:.1f}%   0.5-5.5 ms : {o['frac_iv_0p5_5p5']*100:.1f}%   "
              f"5.5-9.5 ms : {o['frac_iv_5p5_9p5']*100:.1f}%   >=12 ms : {o['frac_iv_ge12']*100:.1f}%")
        edges = list(range(0, 41)) + [10**9]
        labels = [f"{k}-{k+1}" for k in range(40)] + [">40"]
        h, counts = text_hist(iv, edges, labels)
        # print only bins up to the last non-empty one (plus the >40 bin)
        lines = h.split("\n")
        nz = [k for k, c in enumerate(counts[:-1]) if c > 0]
        last = (max(nz) + 1) if nz else 1
        print("\n".join(lines[:last] + [lines[-1]]))

    print()
    print("---- model ----------------------------------------------------------")
    print(f"  UART line rate {UART_BYTES_PER_MS:.2f} B/ms; one 33-B packet occupies "
          f"{PACKET_LEN/UART_BYTES_PER_MS:.2f} ms of every 4.00 ms")
    print(f"  {FTDI_FIFO_PAYLOAD}-B FIFO fills in ~{FTDI_FIFO_PAYLOAD/(PACKET_LEN*NOMINAL_FS/1000):.1f} ms at "
          f"the average rate (>= {FTDI_FIFO_PAYLOAD/UART_BYTES_PER_MS:.1f} ms at line rate)")
    print("  1 ms timer  -> chunks <= 33 B, intervals ~1-5 ms")
    print("  16 ms timer -> chunks ~62 B,  intervals ~7.5 ms (FIFO fill wins; no 16-ms period)")

    label, reasons = verdict(o)
    print()
    print("==== VERDICT ========================================================")
    print(f"  latency timer (from arrival pattern): {label}")
    if chip_latency is not None:
        print(f"  latency timer (chip register via USB): {chip_latency} ms")
    for r in reasons:
        print(f"   - {r}")
    print("=====================================================================")
    print(f"[csv] {csv_path}")


if __name__ == "__main__":
    main()
