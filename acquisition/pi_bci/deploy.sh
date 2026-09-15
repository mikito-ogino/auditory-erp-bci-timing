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
scp "$REPO/pi_bci/bci_server.py" "$REPO/pi_bci/index.html" "$HOST:$DEST/"

echo "==> shared modules"
scp "$REPO/model.py" "$REPO/impedance_check.py" "$HOST:$DEST/"

echo "==> trained model"
scp "$REPO/svm_model.pkl" "$HOST:$DEST/"

echo "==> announcement / beep wavs"
scp "$REPO/beep.wav" \
    "$REPO/target_left_jp.wav" "$REPO/target_center_jp.wav" "$REPO/target_right_jp.wav" \
    "$REPO/result_left_jp.wav" "$REPO/result_center_jp.wav" "$REPO/result_right_jp.wav" \
    "$HOST:$DEST/"

echo "==> done. start it with:"
echo "    ssh $HOST 'cd $DEST && python3 bci_server.py'"
echo "    then open  http://raspberrypi.local/"
