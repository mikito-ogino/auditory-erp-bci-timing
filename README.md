# auditory-erp-bci-timing

Acquisition and stimulus-presentation software for a low-cost auditory
ERP brain-computer interface (BCI) built on an OpenBCI Cyton board.

This repository accompanies the manuscript

> Ogino, M. *A protocol for stimulus presentation and EEG recording in an
> auditory ERP-based brain-computer interface using OpenBCI Cyton.*
> Submitted to the *Journal of Neuroscience Methods*.

The paper describes how the equipment and software around the EEG device
are set up: the operating system, the USB driver, the audio path and the
code that drives it, the radio link and the electrode montage. Each setting
is paired with a quantity that can be read from the stored recording, so
that a laboratory reproducing the protocol can confirm from its own data
that the stimuli were placed on the EEG samples as intended. This
repository holds the software side of that protocol.

## Contents

| Path | Contents |
|---|---|
| `pi_bci/bci_server.py` | Headless acquisition and presentation server (Python, HTTP). Reserves all stimulus onsets of a trial on the sound card's clock, records the EEG with the driver's timestamps, applies the FTDI latency timer on Linux, monitors the effective sampling rate, and writes a `_timing.json` next to every trial with the measured stimulus intervals, effective rate and dropped-sample count |
| `pi_bci/index.html` | Browser client (trace display, impedance panel, session controls) |
| `pi_bci/devices.py` | EEG device back-ends (OpenBCI Cyton via brainflow; Polymate Pocket on Windows) |
| `impedance_check.py` | ADS1299 lead-off impedance measurement (6 nA at 31.2 Hz, 2.2 kΩ series resistor), run from the server on a headless host |
| `model.py`, `hyperparam_search.py` | Classifier and its parameter search, used by the server's training and prediction buttons |
| `latency_probe/ftdi_latency.py` | Reads and writes the FTDI latency timer over USB on macOS, where the Apple driver exposes no setting |
| `latency_probe/latency_timer_probe.py` | Infers the effective latency timer from the arrival pattern of serial data |
| `pi_bci/deploy.sh`, `pi_bci/bci-server.service` | Deployment to a Raspberry Pi and start at boot |
| `*.wav` | Beep and spoken target / result announcements (Japanese) |
| `pi_bci/README.md` | Operating notes for the server (partly in Japanese) |

The server also contains a second recording mode ("Ne-utsuri", `pi_bci/neutsuri*.py`)
for a different project. It shares the acquisition code and is not used in the paper.

## Running

```bash
pip install -r requirements.txt
python3 pi_bci/bci_server.py        # then open http://<host>:8080/
```

On a Raspberry Pi, `pi_bci/deploy.sh` copies the files to the Pi and
`pi_bci/bci-server.service` starts the server at boot. On macOS, set the
latency timer once per session before starting the server:

```bash
python3 latency_probe/ftdi_latency.py set 1
python3 latency_probe/ftdi_latency.py get
```

Recordings are written per subject to `pi_bci/data/<subject>/` as
`YYYYmmdd_HHMMSS_<target>.csv` (`Timestamp, Ch1..Ch8, Label, Seq`; unfiltered
microvolts) with a `_timing.json` sidecar. Trained models go to
`pi_bci/models/<subject>/`. Neither is part of this repository.

## Configuration checklist

The settings the paper asks a reproducing laboratory to make, with how each
one is verified. The section numbers refer to the paper.

| Item | Setting | Verification |
|---|---|---|
| Stimulus scheduling | All onsets of a trial computed before the first tone and reserved at once on one output stream, in sound-card frames; audio buffers prepared when the stream is opened | Successive onsets within one 4 ms sampling period of 500 ms |
| Radio link | Dongle on a USB extension cable, housed with the board a few centimetres away | Per-trial rate within 250 ± 1 Hz |
| USB latency timer | 1 ms (default 16 ms). Windows: Device Manager → Ports → Advanced. Linux: write 1 to sysfs `latency_timer` at each connect, read back. macOS: `ftdi_latency.py` once per session, read back | Inter-packet interval at one sampling period, 4 ms |
| Trace display and other processes | Plotting stopped during trials; other applications closed, notifications and updaters disabled | Sampling rate stays inside the band during trials |
| Sampling-rate acceptance | Rate shown over a five-second window; trial accepted only within 250 ± 1 Hz; session paused if it leaves the band | Per-trial rate over the analysis interval within 250 ± 1 Hz |
| Board power | Battery fully charged, checked with a capacity meter before each session | Rate inside the band |
| Audio path | Passive splitter to participant and experimenter earphones; level set by the software and reapplied at start-up | Experimenter hears every trial |
| Electrode application | 10-20 sites marked with tape measure and pencil before preparation; firm wet-wipe abrasion; Ten20 paste thin and wide, hair parted; leads run to the occiput | Impedance check below |
| Impedance | Per channel, one at a time, 6 nA at 31.2 Hz minus 2.2 kΩ; any channel more than 10 kΩ above the rest re-prepared; start when all < 50 kΩ | Logged before each session |
| Host on Raspberry Pi | Headless server started at boot; browser client on a separate computer over a direct wired link | Same interval and rate figures as on the laptops |

## Data

The EEG recordings are not redistributed here, because the consent obtained
from the patients does not cover public release of their data. Requests for
access should be addressed to the corresponding author.

## Licence

MIT. See `LICENSE`.

## Contact

Mikito Ogino — ogino-m@g.ecc.u-tokyo.ac.jp
