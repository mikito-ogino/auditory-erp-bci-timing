# pi_bci — headless **Auditory BCI & Ne-utsuri** for the Raspberry Pi

Browser-driven version of `main.py`. No Qt GUI: the Pi runs headless and a small
Flask server exposes the controls and the live EEG trace.

The page hosts two independent modes:

| | driven by | records | stimuli |
|---|---|---|---|
| **Auditory BCI** | `bci_server.Experiment` | `YYYYmmdd_HHMMSS_<target>.csv` | four tones, labelled |
| **Ne-utsuri** | `neutsuri.NeutsuriEngine` | `YYYYmmdd_HHMMSS_neutsuri.csv` | none — EEG + ESP32 piezo |

Only one of them may record at a time (they share the amplifier); each refuses to start
while the other is running, and the buttons grey out accordingly.

## EEG アンプ（Cyton / Polymate Pocket）

アンプは **EEG アンプ** パネルで切り替える。切り替えられるのは未接続のときだけで、
選択は `settings.json` の `device` に残るので次の起動でも同じ機種で立ち上がる。

| | サンプリング | チャンネル | ポート | 実装 |
|---|---|---|---|---|
| **OpenBCI Cyton** | 250 Hz | 8ch | FTDI ドングルを自動検出 | brainflow |
| **Polymate Pocket (MP208)** | 1000 Hz | 8ch + **EXT 2ch** | COM番号（空欄なら総当たり） | `polymate.pocket` |

機種ごとの違い（開き方・読み方・インピーダンス計測・レート・補助チャンネル）は
**`devices.py`** に閉じ込めてあり、`Acquisition` はそこだけを触る。レートが変わると
表示バッファ長・間引き率（`DISPLAY_RATE = 50` Hz に合わせる）・表示フィルタの係数も
すべて作り直される。

**Polymate の EXT1/EXT2 には刺激音そのものを入れる。** `Label` 列は「DAC に出た時刻に
最も近い行」に後付けするものなので、音響波形から取ったオンセットのほうが真値に近い。
EXT 列がある CSV では `model.py` が既定でそちら（`detect_onsets_ext()`）を使い、
ラベルも EXT1/EXT2 の相対エネルギーから決める（`USE_EXT_ONSET = False` で Label 列に戻せる）。
無音（音4）は音が出ないので EXT からは検出されず、エポックも作られない。
移動RMS窓が 30 ms の対称窓なので、検出されるオンセットは真のオンセットより
10〜15 ms ほど早い側に寄る。全エポックに同じだけ効くので窓が一様にずれるだけだが、
潜時をミリ秒で議論するときは効いていることを覚えておくこと。

**Polymate API は Windows 専用**（`polymate/pocket.pyd`）。メーカー配布の
`PythonAPI_PolymatePocket` を `pi_bci/reference/PythonAPI_PolymatePocket/` に置くと、
実行中の Python のバージョンに合う `py3XX/` を `devices.py` が自動で `sys.path` に足す。
Pi / Mac では読み込めないので、Polymate を選ぶと**なぜ読めないのかがそのままパネルに出る**
（一覧の選択肢にも「使用不可」と付く）。逆に brainflow が無い Windows 機でも起動できるよう、
`bci_server.py` の brainflow の import は失敗を許してある。

**Ne-utsuri は Cyton 専用。** 押下判定モデルが 250 Hz・2秒窓で学習されているので、
他のレートではモードONも記録開始も塞ぐ（理由をパネルに出す）。

## 被験者

記録データ・予測モデル・音量設定・電極配置の頭部計測値はすべて
**選択中の被験者** に紐づく。

```
data/<被験者>/     20260818_120000_1.csv  20260818_120000_1_timing.json  …
models/<被験者>/   svm_model.pkl  note.txt  best_params.json
settings.json      subjects, current_subject, subject_volumes, subject_head,
                   device, port_<device>
```

**使用中のモデルは常に「選択中の被験者のモデル」**。モデルを別に選べるようにすると
食い違ったまま実験してしまうので、`models/<ディレクトリ名>` は被験者名と1対1に固定した
（`/api/models/select` は被験者切り替えに読み替える）。

**音量は被験者ごとに覚える。** 刺激音量（ソフトウェアゲイン）とデバイス音量（ALSA）の
両方を `settings.json` の `subject_volumes` に保存し、被験者を選び直したときと
サーバー起動時に復元する。提示レベルは実験条件そのものなので、被験者間で持ち越さない。

**電極配置の計測値も被験者ごとに覚える。** 「電極配置」パネルの鼻根部→後頭結節と
左右耳介前点間（cm）は頭のかたちそのもので被験者ごとに違うので、入力した値を
`settings.json` の `subject_head` に保存し、被験者を選び直したときとページを開いた
ときに入れ直して図を描く（未登録の被験者は最後に入力された値を初期値にする）。

被験者を決めていないと保存先もモデルも決まらないので、記録ボタンは押せない。
旧レイアウト（`data/` 直下）の記録は被験者フォルダに入っていないため一覧に出ない。
起動時にその旨と移動コマンドをログに出すので、使いたければ手で移す:

```bash
mkdir -p data/<被験者名> && mv data/*.csv data/*.json data/<被験者名>/
```

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

Stop the service before running anything that needs the sound card or the amplifier
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

The bind address is IPv4-only by default for the reason above. On a machine whose avahi
advertises **only** an IPv6 address, that makes `<host>.local` unreachable while the raw
IPv4 address still works — start it with `BCI_WEB_HOST=::` there to serve both.

`deploy.sh` also pushes `devices.py`, `model.py`, `hyperparam_search.py`,
`impedance_check.py`, `svm_model.pkl` and the beep / Japanese announcement WAVs, since
the Pi app imports and plays them directly. `model.py` and `hyperparam_search.py` are the
two the in-browser training reloads from disk on every run, so pushing them is how a
pipeline change reaches the training button.
It pushes every model set under `pi_bci/models/<被験者名>/` too (`svm_model.pkl`,
`model.py`, `note.txt` — a set of the same name on the Pi is overwritten); see
[予測モデルを学習して配る](#予測モデルを学習して配る).

For the Ne-utsuri mode it additionally pushes `neutsuri.py` + `neutsuri_features.py`, the
per-subject `models/<被験者>/neutsuri_model.pkl` written by `neutsuri_train.py`, and — as the
subject-independent fallback — `neutsuri_model.pkl` if it sits in the repo root, copied over as `neutsuri_model.pkl`. Point somewhere else with
`NEUTSURI_MODEL=/path/to/model.pkl ./pi_bci/deploy.sh`; without it the mode still runs,
but the predicted probability stays 0 and the page says so.

## What the browser page does

- **Left / Center / Right** — plays that tone for a listening check *and* makes it
  the target, exactly like the Play Stimulus buttons in `main.py`.
- **刺激音量 / デバイス音量** — software gain applied to everything the stream plays,
  and the USB card's own ALSA playback volume (`amixer -c <card> sset Speaker`).
  Both are stored **per subject** in `settings.json` (`subject_volumes`) and reapplied
  when the subject is selected and at startup: ALSA resets the card to its default on
  reboot, and the presentation level is an experimental condition that must not depend on
  system state — nor carry over from the previous subject.

  **Which output counts as "USB"** is decided from ALSA, not from the product name:
  a card is USB when `/proc/asound/card<N>/usbid` exists. The server prefers a card it
  can open directly (`hw:N`, exclusive, cleanest DAC clock); where PipeWire/PulseAudio
  holds the card, PortAudio reports that device as 0-channel and the only way through is
  `default` → pulse, so the server takes that route and resolves the sink's card with
  `pactl` — if it lands on a USB card the page reports USB, because that is where the
  sound actually comes out. The ALSA volume follows the resolved card number, so the
  device-volume slider works on the pulse route too. Matching on the name `USB Audio`
  alone used to fail on adapters named anything else, and it also killed the volume
  slider whenever the device name carried no `hw:N`.

  PortAudio caches the device list when it initialises, so a sound card plugged in
  **after** the server started is invisible until **USB機器を再検出** is pressed
  (that re-inits PortAudio and re-detects the amplifier too). Symptom: audio is audible —
  pulse moved the stream to the new sink on its own — while the page still says
  「USBオーディオが未接続です」.
- **被験者** — pick the subject, or type a new name to register and select it. Everything
  downstream follows: `data/<被験者>/` for recordings, `models/<被験者>/` for the model
  used by predict mode, and the subject's own stimulus / device volumes
  and head measurements.
- **電極配置** — the nasion–inion and left–right preauricular distances (cm) that
  place the 10–20 electrodes. Head shape differs per subject, so the two boxes are
  stored **per subject** in `settings.json` (`subject_head`, via
  `GET/POST /api/subjects/head`) and reloaded when the subject is selected or the page
  is opened; a subject with no entry yet starts from the last values entered. The figure
  also shows the two ear-lobe reference electrodes — **white on the left ear (A1), black
  on the right ear (A2)** — so the cable colours match what is actually attached.
- **EEG アンプ** — Cyton か Polymate Pocket を選び、接続 / 切断する。COMポート欄は
  指定が必要な機種のときだけ出る（空欄なら見えているポートを順に試して、実際に
  サンプルが流れてきたものを採用する）。
- **レコーディング（予測なし）** — records one trial and writes the labelled CSV. No classification.
- **レコーディング（予測あり）** — same recording, then runs the **current subject's**
  model on the CSV and announces the result.
- **予測モデル** — one set per subject in `models/<被験者>/`. Two buttons train it
  in place:
  - **選択中の被験者のデータで学習** — `model.py:train()` on `data/<被験者>/`, with the
    stimulus-set count taken from the 計測 panel's selection.
  - **ハイパーパラメータ探索して学習** — `hyperparam_search.py:run_search()` for the
    chosen time budget (10分〜24時間), then trains the final model with what it found.
    The best params land in `models/<被験者>/best_params.json` and progress is written
    to `hyperparam_search_checkpoint.json` every 2 minutes, so a long run can be
    inspected while it goes. Recording is blocked while it runs.
    The score a candidate is ranked by is the **mean of several cross-validations**
    (`hyperparam_search.CV_SCHEMES` — leave-one-day-out, 3-fold, 10-fold and
    one-trial-out over trials), not one-trial-out alone: with n-1 trials always in the
    training set, one-trial-out alone favours params that overfit. Schemes that cannot
    run on the data at hand are dropped and the rest are averaged — leave-one-day-out
    falls back to leave-one-session-out when there is only one recording day and is
    skipped entirely when there is only one session, and k-fold is skipped when the
    subject has k trials or fewer. Which schemes were used is logged at the start of
    the run and stored per candidate in `best_params.json` (`cv_per_scheme_*`).

  Progress and the result appear under the buttons (`/api/state` の `training_message`).
  The table shows what is inside each bundle — training folder, **sampling rate**, sets
  per trial, file count, LOOCV accuracy. The rate matters: a 250 Hz model will happily
  accept a 1000 Hz recording without raising, so check it before trusting a prediction
  (`model.py` also prints a warning on mismatch). Upload is only for importing a model
  trained elsewhere — give it the subject's name. Predict-mode recordings write a
  `_predict.json` sidecar naming the model that produced the result.
  `train_model_set.py` still builds a set offline — see
  [予測モデルを学習して配る](#予測モデルを学習して配る).
- **中止** — aborts the running trial.
- **インピーダンス** — Cyton は `ImpedanceCheck` を既存の brainflow セッション上で
  1chずつ回す。Polymate は `pp.start_impd()` で全ch一括に測る。どちらの手順も
  `devices.py` 側にあり、`ImpedanceRunner` は投げるだけ。
- **平均ERP** — the condition-averaged ERP of one recording, sitting directly above the live
  trace: the Pi's stand-in for the `plot_erp.py` window in `main.py`. All eight channels are
  drawn as a grid, one line per tone (mean ± SEM, baseline = the 100 ms before onset), with
  the target tone thickened. It refreshes itself when a recording finishes, or pick any
  earlier recording from the dropdown. Two presets: **0.5–30 Hz / −100–500 ms** (one SOA,
  same band as `make_erp_report.py`) and **0.5–5 Hz / −100–1000 ms** (what `main.py` plots).
  `erp_view.py` does the epoching and averaging and `/api/erp` returns numbers only — the
  server host never loads matplotlib, and the drawing happens in the browser. Results are
  cached per file + preset, and the endpoint refuses to run while either mode is recording.
  The sampling rate is measured from the CSV's `Timestamp` column and snapped to a rate the
  hardware can actually produce, so the same code handles 250 Hz and 1000 Hz recordings
  (`make_erp_report.py` does the same).
- **リアルタイム EEG** — last 6 s, 0.5–40 Hz, decimated to 50 Hz, drawn on a canvas.
- **Ne-utsuri モード** — the second mode: ESP32 piezo + EEG co-recording and the
  0–100 % EEG/sensor priority mix. Details in its own section below.
- **保存データ** — lists every recording with its target, size and timestamp; tick rows to
  delete them, or rename one in place. The `_timing.json` sidecar moves/goes with the CSV,
  and the matching `eeg_cache/*.npy` entries are dropped so a reused name can never pick up
  a previous recording's cached waveform (the cache key includes the sampling rate for the
  same reason). `_predict.json` and `_impedance.csv` travel with the CSV too. Names are
  restricted to `[A-Za-z0-9_.-]+.csv` resolved inside `data/<被験者>/`, and both operations
  are refused while a trial is running.

- **Pi の時刻** — the Pi has no RTC and the direct-LAN link has no route to an NTP
  server, so after a power cycle its clock resumes from whatever `fake-hwclock` saved at
  shutdown; recordings would then be filed under a date that is days old. The page posts
  its own PC clock to `/api/time` on load and every 60 s, and the server sets the system
  time (`sudo date -u -s @<epoch>`, then `fake-hwclock save` so the next boot starts
  there) when the two differ by more than 5 s. Only the absolute time is taken — the Pi
  keeps its own time zone, so a PC abroad still files recordings in JST. The adjustment is
  refused while a trial, an impedance sweep, a Ne-utsuri recording or a training run is in
  progress: moving the clock mid-run would desynchronise brainflow's `Timestamp` from the
  stimulus schedule and would stretch or cut short the hyperparameter search's time budget.
  The browser retries on the next tick, so it lands as soon as the run ends.

## 予測モデルを学習して配る

`pi_bci/train_model_set.py` turns a folder of labelled recordings into one model set,
so 「レコーディング（予測あり）」 can use it:

```bash
python pi_bci/train_model_set.py --folder <subject> --tuned   # → pi_bci/models/<subject>/
./pi_bci/deploy.sh                                          # → bcipi:~/AuditoryBCI/models/
ssh bcipi 'sudo systemctl restart bci-server'
```

Then select that subject on the page — the set in `models/<被験者名>/` becomes the model
used by predict mode (`current_subject` is stored in `settings.json` and survives a
reboot). Usually the on-page **学習** button is enough; `train_model_set.py` is for
building a set offline from a folder that is not a subject folder. A set is three files:

| file | |
|---|---|
| `svm_model.pkl` | the bundle written by `model.py:train_svm()` — per-tone target/non-target `LinearSVC`s, the preprocessing `params`, and the LOOCV score the table shows |
| `model.py` | ten lines that load the *bundled* `model.py` and pin `BACKEND = "svm"` |
| `note.txt` | the one-line memo shown in the 予測モデル table (cut at 200 chars) |

The set needs its own `model.py` because the bundled one defaults to
`BACKEND = "legacy"` (`MODEL_BACKEND` env var), and legacy predicts from
`clfs.pkl` / `model.pkl`, which the Pi does not have — prediction would return `-1`.
The set's `model.py` does not *copy* the bundled one, it `exec`s it, so it cannot drift
from the real pipeline; it only flips the backend and registers itself as the module
`model` so `svm_model.pkl` (which pickles `model.TrialCalibratedSVC`) restores no matter
what `sys.path` looks like. Preprocessing comes from the `params` inside the bundle, so
the Pi-side `SVM_PRESET` is irrelevant.

`--preset current|filter_fixed|improved` picks one of `model.py`'s presets;
`--tuned` uses `<folder>/tuning_<folder>.pkl` instead (write it first with
`model.tune_svm()`). Every run re-loads the set in a **separate process** the way
`bci_server.get_predictor()` does and predicts a few CSVs, so a bundle that cannot be
restored on the far side fails here rather than mid-experiment. Pass
`--verify-csv <pi_recording.csv>` to put an actual Pi recording through it as well.

## Ne-utsuri モード（ESP32ピエゾ + EEG 同時記録）

Port of the same-named features in `Ne-utsuri/main.py`. The mode always starts **OFF**
— turn it ON in its panel each time you use it; while OFF nothing is claimed and no serial
port is opened (優先度としきい値だけは `settings.json` に残る). While an auditory trial runs,
the mode stands by automatically.

- **ESP32 接続** — the board is found by VID:PID, explicitly skipping the port the Cyton
  dongle holds. Two wirings are accepted: a **CP2102 bridge** (`10C4:EA60`, `/dev/ttyUSB*`
  — the same node family as the Cyton dongle, hence the skip) and **Espressif's built-in
  USB** (`303A:****`, `/dev/ttyACM*`) as on the ESP32-C6/C3/S3. On the built-in-USB boards
  the RTS/DTR reset pulse is *not* sent — it would reset the chip that hosts the USB
  device itself and drop the port we just opened — so no `READY` arrives on connect and
  the firmware is confirmed by `BTN` / *接続テスト* instead. The panel shows the port,
  the last line received with its age, the press / send counts, **and the BLE keyboard
  state**: the firmware answers `SENT` or `NO_BLE` to every `SPACE`, so *接続テスト* tells
  serial-down from BLE-unpaired apart instead of leaving one bug looking like the other.
  A pulled cable is detected and retried every 3 s.
- **判定の優先度（脳波利用率）0–100%** — at every 0.1 s decision tick the engine draws a
  number and consults the **EEG model** with probability *usage*%, the **real piezo** with
  the rest. 0% = every press passes straight through; 100% = only the model can fire.
  A press that arrives on an EEG tick is *latched*, not discarded, so real operations are
  never lost (Mac版と同じ挙動). The slider value is saved with the recording's sidecar,
  since it is an experimental condition.
- **予測しきい値** — the probability must cross *high* upward to fire once, and must fall
  back below *high − 0.20* before it can fire again (0.4 s refractory on top for a v2 model,
  0.8 s for v1). A v2 model additionally re-fires when the probability dips and rises again by
  ≥ 0.2 without having reached *low* — presses come as fast as 0.3 s apart and the burst of
  the second one otherwise sits on top of the first. The panel shows the threshold the
  training script recommends for the loaded model; the slider is not changed automatically.
- **記録開始 / 停止** — records until stopped, unlike the trial-shaped auditory mode.
- **トレース** — presses, probability and sent commands over the same 6 s window as the EEG
  canvas above, written at the same ring positions so the two line up sample-for-sample.

CSV: `data/YYYYmmdd_HHMMSS_neutsuri.csv`

| column | meaning |
|---|---|
| `Timestamp` | brainflow's own timestamp, unmodified |
| `Ch1..Ch8` | **unfiltered** µV — the same input the model was trained on |
| `PiezoVoltage` | `1.0` on the single row nearest the press, else 0 (the v1 script reads this as the label) |
| `Prediction` | latest probability; a 10 Hz staircase |
| `Command` | `1.0` sensor-driven SPACE, `2.0` EEG-driven SPACE, else 0 |

A `<name>.json` sidecar records the session's conditions (usage %, thresholds, model,
ESP32/BLE state) and the measured sample-interval statistics; it is moved and deleted
along with the CSV.

### 押下判定モデル（`neutsuri_train.py`）

`models/<被験者>/neutsuri_model.pkl` is preferred, `neutsuri_model.pkl` next to the app is
the fallback; the model is re-resolved when the subject changes. Two file formats load:

| | v2 (`neutsuri_train.py`) | v1 (the earlier Ne-utsuri script) |
|---|---|---|
| window | 0.3 s (last 75 of the 500-sample ring) | 2 s |
| features | log band power 4–12 / 12–20 / 20–45 / 55–95 / 105–124 Hz × 8 ch (`neutsuri_features.py`) | 116 statistics / Hjorth / correlations / band powers |
| normalisation | causal: median / MAD of the previous 60 s of features (reset on mode ON) | StandardScaler fitted at training |
| classifier | shrinkage LDA (`predict_proba`) | RandomForest |
| label at training | press 0.05–0.35 s before the window end | any press inside the 2 s window |
| refractory / re-arm | 0.4 s / re-fire on a ≥ 0.2 rise | 0.8 s / none |

Why: the bite shows up as a ~0.4 s broadband EMG burst (20–45 and 60–125 Hz) around the press,
and presses arrive about once per second. A 2 s window labelled "contains a press" is positive
most of the time and the burst is diluted — on the earlier sessions the v1 model is at chance
(event F1 ≈ 0.01, the probability sits at ≈ 0.6 all day). The v2 model reaches event F1 ≈ 0.80
(hit ≈ 0.82, precision ≈ 0.77) across sessions it has never seen (`neutsuri_bite_analysis.py`
in the repo root has the comparison and writes the report figures).

```bash
python pi_bci/neutsuri_train.py <subject>        # data/<subject>/*.csv → models/<subject>/neutsuri_model.pkl (+ .json)
python pi_bci/neutsuri_train.py <subject> --cv     # leave-one-session-out numbers only
```

The pkl is a dict (`format: "neutsuri-v2"`) carrying the feature configuration, the
estimator, the recommended threshold and the LOSO results, so `neutsuri.py` needs nothing
but the file. Both CSV layouts (Mac `main.py` and Pi `_neutsuri.csv`) are accepted as
training data. `neutsuri_features.py` is shared verbatim between training and the Pi — if it
changes, retrain.

### Differences from `Ne-utsuri/main.py`

- Inference runs on its own 10 Hz thread, not inside the sample callback. On the Pi a
  ~20 ms RandomForest call inside the acquisition path would drop samples outright.
- Presses are placed on the row whose timestamp is closest, because the Pi reads in
  ~20 ms chunks; appending "wherever we are now" would smear them by up to a chunk.
- `predict_proba` is used instead of `predict`, so the threshold control actually does
  something (the Mac SVM path returned only 0/1).
- The draw is `random.random()*100 < usage`; `randint(0, 100) < usage` never quite
  reached 100 %.
- `feature_buffer` becomes a 500-sample ring (2 s). A v1 model consumes all of it through
  the verbatim copy of the v1 feature code; a v2 model takes the last 0.3 s
  through `neutsuri_features.py`.

CSVs land in `~/AuditoryBCI/data/` as `YYYYmmdd_HHMMSS_<target>.csv` with the same
columns as the Mac version (`Timestamp, Ch1..Ch8, Label, Seq`). EEG is stored **unfiltered**
in µV; the 0.5–40 Hz filter is applied only to the browser trace.

Each recording also gets a `_timing.json` sidecar with the measured sample intervals,
the audio SOA and the label SOA. Nothing in the CSV is altered to produce it — the
timestamps stay exactly as brainflow reported them.

## Differences from main.py

| | Mac `main.py` | Pi `bci_server.py` |
|---|---|---|
| Amplifier | Cyton only | Cyton or Polymate Pocket, switchable from the page (`devices.py`) |
| Acquisition | pyOpenBCI callback, `time.time()` per sample | brainflow `get_board_data()` / `polymate.pocket`, driver timestamps |
| Stimulus timing | `pygame.mixer` + sleep loop | persistent `sounddevice` stream, onsets scheduled on the DAC clock |
| Onset timestamp | wall clock just before `play()` | wall-clock equivalent of the actual DAC output time (Polymate additionally records the audio itself on EXT1/EXT2) |
| Data / models | one folder, one model | per subject: `data/<被験者>/`, `models/<被験者>/` |
| Training | `python model.py` on the Mac | also from the page (plain, or with a time-budgeted hyperparameter search) |
| UI | PyQt5 windows | Flask + HTML page |
| LED / tactile ESP32 | supported | not ported (the Ne-utsuri piezo ESP32 *is* — see below) |

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

The state panel shows the measured rate next to the amplifier's nominal rate for that
reason: if samples are being dropped, only the measured one falls.

Polymate's API hands over a block of samples without arrival times, so `devices.py`
stamps the last sample "now" and spaces the rest at exactly 1/1000 s. Timestamps within
a block are therefore nominal rather than measured — unlike Cyton, where brainflow's own
timestamp channel is used. `_timing.json` also omits `dropped_samples` for Polymate,
since the API exposes no packet counter to compare against.

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
