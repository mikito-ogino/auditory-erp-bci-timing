#!/bin/bash
# Push the Pi-side app and everything it needs to the Raspberry Pi.
# Usage: ./pi_bci/deploy.sh          (run from the repo root, or anywhere)
set -e

HOST=${1:-bcipi}
DEST=/home/pi/AuditoryBCI
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> creating $DEST on $HOST"
ssh "$HOST" "mkdir -p $DEST/data"

echo "==> app"
scp "$REPO/pi_bci/bci_server.py" "$REPO/pi_bci/devices.py" "$REPO/pi_bci/neutsuri.py" \
    "$REPO/pi_bci/neutsuri_features.py" \
    "$REPO/pi_bci/erp_view.py" "$REPO/pi_bci/index.html" "$HOST:$DEST/"

# model.py は「ブラウザからの学習」でも毎回ディスクから読み直される。
# hyperparam_search.py は「ハイパーパラメータ探索して学習」が使う。
echo "==> shared modules"
scp "$REPO/model.py" "$REPO/hyperparam_search.py" "$REPO/impedance_check.py" "$HOST:$DEST/"

echo "==> trained model"
scp "$REPO/svm_model.pkl" "$HOST:$DEST/"

# 予測モデルのセット（pi_bci/models/<被験者名>/）。使用中のモデルは常に
# 「選択中の被験者のモデル」なので、ディレクトリ名は被験者名と一致させる。
# ブラウザの「学習」ボタンでも作られるが、リポジトリで管理しているものはここで配る。
# 同名のセットは上書きになる。
echo "==> model sets"
pushed=0
for d in "$REPO"/pi_bci/models/*/; do
  [ -d "$d" ] || continue
  d=${d%/}
  name=$(basename "$d")
  if [ ! -f "$d/svm_model.pkl" ] && [ ! -f "$d/neutsuri_model.pkl" ]; then
    echo "    $name: skipped (no svm_model.pkl / neutsuri_model.pkl)"
    continue
  fi
  ssh "$HOST" "mkdir -p $DEST/models/$name"
  if [ -f "$d/svm_model.pkl" ]; then
    scp "$d/svm_model.pkl" "$HOST:$DEST/models/$name/"
  fi
  for extra in model.py note.txt best_params.json neutsuri_model.pkl; do
    if [ -f "$d/$extra" ]; then
      scp "$d/$extra" "$HOST:$DEST/models/$name/"
    fi
  done
  echo "    $name"
  pushed=$((pushed + 1))
done
if [ "$pushed" -eq 0 ]; then
  echo "    none (ブラウザの「学習」ボタン、または pi_bci/train_model_set.py で作る)"
fi

# Ne-utsuri モードの共通（被験者非依存）押下判定モデル。被験者ごとのものは上の
# models/<被験者>/neutsuri_model.pkl が優先される（neutsuri_train.py が書く）。
# 共通のものは旧 Ne-utsuri リポジトリの pkl を隣に置いてあれば拾う。無ければ黙って
# 飛ばさず、何が起きたか出す（Pi 側では「モデル未配置」と表示され、確率は常に0になる）。
NEUTSURI_MODEL=${NEUTSURI_MODEL:-$REPO/neutsuri_model.pkl}
if [ -f "$NEUTSURI_MODEL" ]; then
  echo "==> Ne-utsuri model ($NEUTSURI_MODEL)"
  scp "$NEUTSURI_MODEL" "$HOST:$DEST/neutsuri_model.pkl"
else
  echo "==> Ne-utsuri model: skipped (not found at $NEUTSURI_MODEL)"
  echo "    put it there, or run: NEUTSURI_MODEL=/path/to/model.pkl ./pi_bci/deploy.sh"
fi

echo "==> announcement / beep wavs"
scp "$REPO/beep.wav" \
    "$REPO/target_left_jp.wav" "$REPO/target_center_jp.wav" "$REPO/target_right_jp.wav" \
    "$REPO/result_left_jp.wav" "$REPO/result_center_jp.wav" "$REPO/result_right_jp.wav" \
    "$HOST:$DEST/"

echo "==> done. start it with:"
echo "    ssh $HOST 'cd $DEST && python3 bci_server.py'"
echo "    then open  http://raspberrypi.local/"
