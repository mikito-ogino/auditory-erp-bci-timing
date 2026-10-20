#!/usr/bin/env python
"""
Read / set the FTDI latency timer of the OpenBCI Cyton dongle (FT231X) on
macOS by talking to the chip directly over USB (libusb via ctypes).

Why this exists
---------------
Apple's DriverKit AppleUSBFTDI exposes no latency-timer setting: the IOKit
IOSSDATALAT ioctl is accepted but ignored (verified with
latency_timer_probe.py).  The chip itself, however, answers the standard FTDI
vendor requests on endpoint 0:

    SIO_SET_LATENCY_TIMER  bmRequestType 0x40, bRequest 0x09, wValue = ms (1..255)
    SIO_GET_LATENCY_TIMER  bmRequestType 0xC0, bRequest 0x0A, 1 data byte

The timer is a volatile register: it resets to 16 ms when the dongle is
re-plugged / re-enumerated, and a driver may rewrite it when the port is
opened.  Always confirm with `get` (or with latency_timer_probe.py) *after*
the serial port has been opened in the state you care about.

Dependencies: only Homebrew's libusb (libusb-1.0.dylib); no Python package.

Usage
-----
  python latency_probe/ftdi_latency.py get
  python latency_probe/ftdi_latency.py set 1
  python latency_probe/ftdi_latency.py set 16     # restore default
"""

import ctypes
import ctypes.util
import sys

FTDI_VID = 0x0403
FT231X_PID = 0x6015           # 24597, as shown by ioreg for "FT231X USB UART"

SIO_SET_LATENCY_TIMER = 0x09
SIO_GET_LATENCY_TIMER = 0x0A
REQTYPE_OUT = 0x40            # host->device, vendor, device
REQTYPE_IN = 0xC0             # device->host, vendor, device
TIMEOUT_MS = 1000

LIBUSB_CANDIDATES = [
    "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
    "/opt/homebrew/lib/libusb-1.0.dylib",
    "/usr/local/lib/libusb-1.0.dylib",
]


def load_libusb():
    for p in LIBUSB_CANDIDATES:
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    found = ctypes.util.find_library("usb-1.0")
    if found:
        return ctypes.CDLL(found)
    sys.exit("libusb-1.0 not found (brew install libusb)")


class FtdiLatency:
    def __init__(self, vid=FTDI_VID, pid=FT231X_PID):
        self.lib = load_libusb()
        lib = self.lib
        lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
        lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
        lib.libusb_control_transfer.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8,
                                                ctypes.c_uint16, ctypes.c_uint16,
                                                ctypes.c_char_p, ctypes.c_uint16, ctypes.c_uint]
        lib.libusb_control_transfer.restype = ctypes.c_int
        lib.libusb_close.argtypes = [ctypes.c_void_p]
        lib.libusb_exit.argtypes = [ctypes.c_void_p]
        lib.libusb_strerror.argtypes = [ctypes.c_int]
        lib.libusb_strerror.restype = ctypes.c_char_p

        self.ctx = ctypes.c_void_p()
        r = lib.libusb_init(ctypes.byref(self.ctx))
        if r != 0:
            sys.exit(f"libusb_init failed: {self._err(r)}")
        self.h = lib.libusb_open_device_with_vid_pid(self.ctx, vid, pid)
        if not self.h:
            lib.libusb_exit(self.ctx)
            sys.exit(f"FTDI device {vid:04x}:{pid:04x} not found or could not be opened")

    def _err(self, r):
        return self.lib.libusb_strerror(r).decode()

    def get(self) -> int:
        buf = ctypes.create_string_buffer(1)
        r = self.lib.libusb_control_transfer(self.h, REQTYPE_IN, SIO_GET_LATENCY_TIMER,
                                             0, 0, buf, 1, TIMEOUT_MS)
        if r != 1:
            raise OSError(f"GET_LATENCY_TIMER failed: {self._err(r) if r < 0 else r}")
        return buf.raw[0]

    def set(self, ms: int) -> None:
        if not 1 <= ms <= 255:
            raise ValueError("latency must be 1..255 ms")
        r = self.lib.libusb_control_transfer(self.h, REQTYPE_OUT, SIO_SET_LATENCY_TIMER,
                                             ms, 0, None, 0, TIMEOUT_MS)
        if r < 0:
            raise OSError(f"SET_LATENCY_TIMER failed: {self._err(r)}")

    def close(self):
        if self.h:
            self.lib.libusb_close(self.h)
            self.h = None
        if self.ctx:
            self.lib.libusb_exit(self.ctx)
            self.ctx = None


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("get", "set"):
        sys.exit(__doc__)
    dev = FtdiLatency()
    try:
        if sys.argv[1] == "get":
            print(f"latency timer = {dev.get()} ms")
        else:
            ms = int(sys.argv[2])
            before = dev.get()
            dev.set(ms)
            after = dev.get()
            print(f"latency timer: {before} ms -> {after} ms (requested {ms})")
            if after != ms:
                sys.exit("device did not accept the new value")
    finally:
        dev.close()


if __name__ == "__main__":
    main()
