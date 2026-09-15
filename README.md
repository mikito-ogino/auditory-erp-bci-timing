# auditory-erp-bci-timing

Configuration protocol and timing diagnostics for a low-cost auditory
ERP brain–computer interface (BCI) built on an OpenBCI Cyton board.

This repository accompanies the manuscript

> Ogino, M. *Millisecond stimulus–EEG alignment on a low-cost ERP BCI is a
> configuration problem: a protocol and its timestamp-based verification.*
> Submitted to the *Journal of Neuroscience Methods*.

The paper's claim is that the stimulus–electroencephalogram (EEG) alignment
error of a \$1,200 open-source ERP BCI is not a hardware limit but a set of
host-side settings, and that the alignment can be held inside one 250 Hz
sampling period (4 ms) and verified from the host timestamps a recording
already stores, with no added instrumentation.

## Status

The name is reserved and the code is being prepared for release. The
acquisition and presentation software is being separated from the
participant recordings before it is published here.

## Planned contents

| Path | Contents |
|---|---|
| `acquisition/` | EEG acquisition, stimulus presentation and the HTTP/browser interface (Python), including the FTDI latency-timer write and the sampling-rate watchdog |
| `impedance/` | ADS1299 lead-off impedance measurement, usable from a headless host |
| `diagnostics/` | Timestamp-based reconstruction of stimulus–sample asynchrony, packet arrival and effective sampling rate |
| `analysis/` | Scripts reproducing every figure and table of the paper |
| `checklist.yaml` | The configuration checklist of the paper's appendix, machine-readable |

## Configuration checklist

The settings the paper asks a reproducing laboratory to make, with how each
one is verified.

| Item | Setting | Verification |
|---|---|---|
| FTDI latency timer | 1 ms (default 16 ms). Windows: Device Manager → Ports → Advanced. Linux: write `1` to `/sys/.../latency_timer`. macOS: D2XX `FT_SetLatencyTimer` | Inter-packet interval histogram unimodal near 4 ms |
| Stimulus scheduling | Absolute deadlines from one monotonic reference, no `sleep` between stimuli, and audio buffers prepared before *t*₀ | Asynchrony independent of stimulus index |
| Sampling rate monitor | Dedicated thread, sliding-window estimate, run rejected outside 250 ± 1 Hz | Logged per run |
| Radio link | Dongle on a USB extension cable, co-housed with the board a few centimetres away | Missing-sample count per 10 min |
| Rendering | All plotting suspended during trials, other applications closed, and notifications disabled | Sampling rate stable during trials |
| Host on Raspberry Pi | Headless server with a browser client on a separate machine, and a Pi 4 is sufficient | Sampling rate inside band |
| Audio path | Passive splitter to participant and experimenter, and host not muted | Experimenter hears every trial |
| Battery | Fully charged before session | — |
| Electrode preparation | Dermatographic 10–20 marking with tape measure, degrease, thin and wide paste, leads routed to the occiput | — |
| Impedance | Per channel, one at a time, 6 nA at 31.2 Hz, minus 2.2 kΩ series, all channels < 50 kΩ | Logged at session start and end |
| Electrode removal | One at a time, pulled from the board end of the lead | — |

## Data

The EEG recordings are not redistributed here, because the consent obtained
from the patients does not cover public release of their data. Requests for
access should be addressed to the corresponding author.

## Licence

To be added before the code is published.

## Contact

Mikito Ogino — ogino-m@g.ecc.u-tokyo.ac.jp
