# pi_bci — headless auditory BCI for the Raspberry Pi

Browser-driven version of `main.py`. No Qt GUI: the Pi runs headless and a small
Flask server exposes the controls and the live EEG trace.

## Deploy

```bash
./pi_bci/deploy.sh            # copies to bcipi:/home/pi/AuditoryBCI/
ssh bcipi 'sudo systemctl restart bci-server'
```

The server runs as the systemd unit **`bci-server.service`** (installed from
`pi_bci/bci-server.service`), enabled at boot, so normally there is nothing to start
by hand — power the Pi on and open the page.

```bash
ssh bcipi 'systemctl status bci-server'      # state
ssh bcipi 'journalctl -u bci-server -f'      # live log
ssh bcipi 'sudo systemctl stop bci-server'   # free the audio device / serial port
```

Stop the service before running anything that needs the sound card or the Cyton
(`timing_test.py`, a manual `python3 bci_server.py`) — both are opened exclusively.

Then open **<http://raspberrypi.local/>** — no IP address, no port number.

That name is answered by the Pi's avahi over mDNS. Its IPv4 answer is the direct-cable
address `192.168.137.20`, so the browser always goes over the LAN cable rather than WiFi;
the server binds `0.0.0.0` (IPv4 only), so a browser that tries the Pi's WiFi-side IPv6
address just falls straight back to IPv4. `http://192.168.137.20/` still works if mDNS
is ever unavailable.

Port 80 is used so the URL needs no `:8080`. The server runs as `pi`, not root — the Pi
has `net.ipv4.ip_unprivileged_port_start=80` (`/etc/sysctl.d/50-unprivileged-port80.conf`).
If port 80 cannot be bound the server falls back to 8080 on its own, and
`BCI_WEB_PORT=<n>` overrides the choice.

`deploy.sh` also pushes `model.py`, `impedance_check.py`, `svm_model.pkl` and the
beep / Japanese announcement WAVs, since the Pi app imports and plays them directly.

## What the browser page does

- **Left / Center / Right** — plays that tone for a listening check *and* makes it
  the target, exactly like the Play Stimulus buttons in `main.py`.
- **刺激音量 / デバイス音量** — software gain applied to everything the stream plays,
  and the USB card's own ALSA playback volume (`amixer -c <card> sset Speaker`).
  Both are written to `settings.json` and reapplied at startup: ALSA resets the card to
  its default on reboot, and the presentation level is an experimental condition that
  must not depend on system state.
- **レコーディング（予測なし）** — records one trial and writes the labelled CSV. No classification.
- **レコーディング（予測あり）** — same recording, then runs the selected model on the
  CSV and announces the result.
- **予測モデル** — model sets live in `models/<name>/`, one per experiment. Pick the
  active one with the radio button, or upload a new set (name + `svm_model.pkl`, plus
  `model.py` only if the preprocessing differs, plus an optional note). The table shows
  what is inside each bundle — training folder, preset, LOOCV accuracy — so sets are
  distinguishable at a glance. Predict-mode recordings write a `_predict.json` sidecar
  naming the model that produced the result.
- **中止** — aborts the running trial.
- **インピーダンス** — runs `ImpedanceCheck` per channel over the existing brainflow session.
- **リアルタイム EEG** — last 6 s, 0.5–40 Hz, decimated to 50 Hz, drawn on a canvas.
- **保存データ** — lists every recording with its target, size and timestamp; tick rows to
  delete them, or rename one in place. The `_timing.json` sidecar moves/goes with the CSV,
  and the matching `eeg_cache/*.npy` entries are dropped so a reused name can never pick up
  a previous recording's cached waveform. Names are restricted to `[A-Za-z0-9_.-]+.csv`
  resolved inside `data/`, and both operations are refused while a trial is running.

CSVs land in `~/AuditoryBCI/data/` as `YYYYmmdd_HHMMSS_<target>.csv` with the same
columns as the Mac version (`Timestamp, Ch1..Ch8, Label, Seq`). EEG is stored **unfiltered**
in µV; the 0.5–40 Hz filter is applied only to the browser trace.

Each recording also gets a `_timing.json` sidecar with the measured sample intervals,
the audio SOA and the label SOA. Nothing in the CSV is altered to produce it — the
timestamps stay exactly as brainflow reported them.

## Differences from main.py

| | Mac `main.py` | Pi `bci_server.py` |
|---|---|---|
| Acquisition | pyOpenBCI callback, `time.time()` per sample | brainflow `get_board_data()`, brainflow timestamp channel |
| Stimulus timing | `pygame.mixer` + sleep loop | persistent `sounddevice` stream, onsets scheduled on the DAC clock |
| Onset timestamp | wall clock just before `play()` | wall-clock equivalent of the actual DAC output time |
| UI | PyQt5 windows | Flask + HTML page |
| LED / tactile ESP32 | supported | not ported |

## Timing

The FTDI dongle's `latency_timer` **must** be 1 ms. At the Linux default of 16 ms the
driver batches serial data, so the 250 Hz samples arrive in bursts and brainflow's
timestamps go staircase-shaped — which pushes the label onto the wrong row. `bci_server.py`
lowers it on every connect (same approach as `AugmentedArms/ABMI_Utils.set_latency_timer`),
and `/etc/udev/rules.d/99-ftdi-latency.rules` on the Pi sets it at plug-in as a backstop.

Measured over full 40-stimulus trials (nominal 500 ms SOA):

| | `latency_timer` = 16 | = 1 | = 1, display paused |
|---|---|---|---|
| sample interval SD | 14.03 ms | 0.67 ms | **0.59 ms** |
| sample interval range | 0.18 – 59.6 ms | 1.24 – 7.06 ms | 0.31 – 7.37 ms |
| audio SOA SD | — | 1.49 ms | **0.08 ms** |
| label SOA SD | 40.1 ms | 2.42 ms | **1.49 ms** |
| label SOA range | 304 – 539 ms | 494.8 – 504.8 ms | 496.8 – 503.9 ms |
| dropped samples | — | 0 | 0 |

Stimulus onsets are scheduled on integer DAC frame counts, so the acoustic SOA cannot
drift. What *can* wobble is the wall-clock estimate written for each onset, which is
derived per audio callback from `outputBufferDacTime - currentTime`; under display load
that estimate carried ~1.5 ms of noise, which is why the display is suspended during
recording (see below).

### Suspending the display during recording

While a trial records, the reader thread skips the per-channel display filter, the ring
buffer updates and the derived frame/stats entirely, and the browser stops polling
`/api/eeg`. The sampling-rate estimate keeps updating. The page greys the trace out and
shows an explicit "⏸ 記録中は表示を停止しています" badge, so a frozen trace can never be
mistaken for a fault.

### The sampling rate reads 250.3, not 250.0

That residual is real hardware: the Cyton's oscillator runs ~1200 ppm fast relative to
the Pi's clock, so 250.3 samples arrive per host second. No amount of load reduction
changes it. (An earlier apparent 251.5 Hz *was* a bug — the live estimator summed N+1
chunks' samples over N chunk intervals, a +1/N ≈ +0.4 % bias at a 20 ms read interval.
Both that estimator and `effective_fs` in the timing report now use n-1 intervals.)

## Notes

- The USB audio device is opened exclusively (`hw:3,0`), so announcements are mixed
  into the same stream rather than played through a second `sd.play()` stream.
- To keep the acquisition thread from being starved, the reader prepares the decimated
  trace and the per-channel signal stats every 200 ms; the HTTP handlers only hand back
  what is already built. Werkzeug's per-request logging is silenced for the same reason.
- Impedance measurement pauses the stream, so it is rejected while a trial is running.
- `scikit-learn` on the Pi is pinned to 1.7.1 to match the version that wrote
  `svm_model.pkl`; the Debian package (1.2.1) cannot unpickle it.
- brainflow is built from source (tag 5.18.1) at `~/brainflow`; there is no aarch64 wheel.
