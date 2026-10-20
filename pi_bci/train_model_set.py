#!/usr/bin/env python3
"""データフォルダから Pi 用のモデルセット `pi_bci/models/<name>/` を作る。

Pi 側 (`bci_server.py`) は「レコーディング（予測あり）」で
`models/<active>/svm_model.pkl` を読み、そのセットの `model.py` の
`predict(csv_path)` を呼ぶ。このスクリプトはその 3 点セットを一括で作る:

    models/<name>/svm_model.pkl … 学習済みバンドル（model.py の train_svm() 形式）
    models/<name>/model.py      … 同梱 model.py を読み込み backend を svm に固定する薄い層
    models/<name>/note.txt      … 学習フォルダ・プリセット・LOOCV 精度の覚え書き

使い方:

    python pi_bci/train_model_set.py --folder <subject>                 # 既定プリセット
    python pi_bci/train_model_set.py --folder <subject> --preset improved
    python pi_bci/train_model_set.py --folder <subject> --tuned         # tune_svm の best を使う
    python pi_bci/train_model_set.py --folder <subject> --verify-csv <Piで録ったCSV>
    python pi_bci/train_model_set.py --folder <subject> --note "精度の読み方の注意"

作ったセットは `./pi_bci/deploy.sh` で Pi に配られる。Pi 側の「使用中のモデル」は
常に「選択中の被験者のモデル」なので、`--name`（既定は `--folder` 名）は
被験者名と一致させること。ブラウザの「学習」ボタンでも同じ場所に同じ形式で書かれる
ので、こちらは被験者フォルダ以外のデータからセットを作りたいときに使う。

リポジトリ直下の `model.py` は変更しない（学習・前処理はそのまま再利用する）。
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# セットに置く model.py。同梱 model.py（Pi では /home/pi/AuditoryBCI/model.py、
# ローカルではリポジトリ直下）の中身をこのモジュールの名前空間で実行し、
# backend だけ svm に固定する。丸ごとコピーしないので本体と食い違わない。
#
# なぜ必要か: 同梱 model.py の `predict()` は BACKEND（既定 "legacy"）で分岐し、
# legacy は clfs.pkl / model.pkl を要求する。Pi にはそれが無いので予測が -1 になる。
# セット内 model.py があれば `get_predictor()` はそちらを優先して読む。
SHIM = '''\
"""{name} 用の予測層（{stamp} に train_model_set.py が生成）。

同梱 model.py をこのモジュールに読み込み直し、`predict()` の backend を
svm（このセットの svm_model.pkl = train_svm() のバンドル）に固定する。
前処理は svm_model.pkl の "params" に入っているものが使われるので、
同梱 model.py の SVM_PRESET には依存しない。
"""
import os as _os

# 同梱 model.py の場所。Pi では models/ の親（/home/pi/AuditoryBCI）、
# このリポジトリでは pi_bci/models/ の 3 つ上（リポジトリ直下）にある。
_here = _os.path.dirname(_os.path.abspath(__file__))
_up = lambda n: _os.path.join(*([_here] + [".."] * n + ["model.py"]))
for _bundled in (_up(2), _up(3)):
    _bundled = _os.path.normpath(_bundled)
    if _os.path.exists(_bundled):
        break
else:
    raise FileNotFoundError("同梱 model.py が見つかりません: " + _here)

with open(_bundled) as _fp:
    exec(compile(_fp.read(), _bundled, "exec"), globals())

BACKEND = "svm"      # ← 予測を svm バックエンド（このセットの pkl）に固定する

# svm_model.pkl は自作クラスを `model.TrialCalibratedSVC` として参照する。
# `import model` が通るかは起動時の sys.path 次第（Pi は bci_server.py と同じ
# ディレクトリに model.py があるので通るが、他の起動方法では通らない）。
# 中身は同じなので、このモジュール自身を `model` としても登録しておく。
# 本物が既に読み込まれていればそれを尊重する（setdefault）。
import sys as _sys
_self = _sys.modules.get(__name__)
if _self is not None:
    _sys.modules.setdefault("model", _self)
'''


def build_note(bundle, folder, n_csv, stamp, extra=None):
    """モデル一覧に出す 1 行メモ

    `list_models()` が note.txt を 200 文字で切るので、1 行に収める。
    """
    acc, sd = bundle.get("loocv_acc"), bundle.get("loocv_acc_sd")
    p = bundle.get("params", {})
    labels = bundle.get("label_candidates") or []
    chance = f"{1.0 / len(labels):.3f}" if labels else "?"
    score = (f"LOOCV {acc:.3f}±{sd:.3f} (chance {chance})" if acc is not None
             else "LOOCV 未評価")
    note = (f"{os.path.basename(folder.rstrip('/'))} {n_csv}本 labels{labels} / "
            f"{bundle.get('preset')}: {p.get('use_filter')} "
            f"{p.get('low_cutoff')}-{p.get('high_cutoff')}Hz "
            f"bl{p.get('baseline_length')} ds{p.get('downsampling_rate')} "
            f"C{p.get('C')} / {score}")
    if extra:
        note += f" / {extra}"
    return note + f" / {stamp}\n"


# Pi の get_predictor() と同じ手順でセットを読み、CSV を予測する検証用スクリプト。
# 学習に使ったプロセスとは別プロセスで動かして、pickle が独立に復元できることまで見る。
VERIFY = r'''
import importlib.util, json, os, sys

# 起動ディレクトリを sys.path から外す。こうしておくと「`import model` が
# 通らない環境」でも pkl が復元できることまで確かめられる（セットの model.py が
# 自分を `model` として登録するので通る）。Pi は通る側なので、これは厳しめの条件。
sys.path = [p for p in sys.path if p not in ("", ".", os.getcwd())]

import numpy as np

set_dir, csvs = sys.argv[1], sys.argv[2:]

def load_module_from(path, module_name):        # bci_server.load_module_from と同じ
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def alias_model_classes(mod):                   # bci_server.alias_model_classes と同じ
    main = sys.modules["__main__"]
    for attr, obj in vars(mod).items():
        if isinstance(obj, type) and getattr(obj, "__module__", "") == mod.__name__:
            setattr(main, attr, obj)

name = os.path.basename(set_dir.rstrip("/"))
mod = load_module_from(os.path.join(set_dir, "model.py"), f"bci_model_{name}")
mod.model_path = os.path.join(set_dir, "svm_model.pkl")
alias_model_classes(mod)

out = []
for path in csvs:
    pred = int(np.asarray(mod.predict(path)).ravel()[0])
    out.append({"file": os.path.basename(path), "pred": pred})
print(json.dumps({"backend": mod.BACKEND, "results": out}))
'''


def verify(set_dir, csvs, cwd):
    """別プロセスで Pi と同じ読み込み手順を踏み、予測が返ることを確かめる"""
    proc = subprocess.run([sys.executable, "-c", VERIFY, set_dir, *csvs],
                          capture_output=True, text=True, cwd=cwd,
                          env={**os.environ, "OMP_NUM_THREADS": "1"})
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"検証に失敗しました (exit {proc.returncode})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True,
                    help="学習データのフォルダ（CSV の末尾 _1/-1 等が注意音ラベル）")
    ap.add_argument("--name", default=None, help="モデルセット名（既定: フォルダ名）")
    ap.add_argument("--preset", default="filter_fixed",
                    help="model.py の SVM_PRESET（current / filter_fixed / improved）")
    ap.add_argument("--tuned", action="store_true",
                    help="<folder>/tuning_<folder>.pkl の best_params で学習する")
    ap.add_argument("--no-eval", action="store_true",
                    help="one-trial-out LOOCV を省く（速いが note に精度が残らない）")
    ap.add_argument("--note", default=None,
                    help="note.txt に足す一言（精度の読み方の注意など）。一覧は 200 字で切られる")
    ap.add_argument("--verify-csv", action="append", default=[],
                    help="追加で予測を通す CSV（Pi で録った CSV を指定すると配布前に確認できる）")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"フォルダがありません: {folder}")
    name = args.name or os.path.basename(folder.rstrip("/"))

    # model.py はプリセットを import 時に確定するので、import より先に環境変数で渡す
    os.environ["SVM_PRESET"] = args.preset
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    os.chdir(REPO)          # eeg_cache と相対パスをリポジトリ直下に揃える
    sys.path.insert(0, REPO)
    import joblib
    import model as M

    set_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(set_dir, exist_ok=True)
    pkl = os.path.join(set_dir, "svm_model.pkl")

    M.data_folder = folder      # 学習対象 & eeg_cache のキー
    M.model_path = pkl          # train_svm() の保存先

    csvs = sorted(f for f in os.listdir(folder)
                  if f.endswith(".csv") and not f.endswith("impedance.csv"))
    stamp = time.strftime("%Y-%m-%d")
    how = ("tuned (%s)" % os.path.basename(M.default_tuning_path(folder))
           if args.tuned else args.preset)
    print(f"=== {name}: {folder} ({len(csvs)} CSV) params={how} ===")

    M.train_svm(evaluate_trial=not args.no_eval, use_tuned=args.tuned)

    bundle = joblib.load(pkl)
    with open(os.path.join(set_dir, "note.txt"), "w") as f:
        f.write(build_note(bundle, folder, len(csvs), stamp, args.note))
    with open(os.path.join(set_dir, "model.py"), "w") as f:
        f.write(SHIM.format(name=name, stamp=stamp))

    # 検証: 学習フォルダの先頭数本（in-sample なので健全性の確認だけ）＋ 指定 CSV
    probe = [os.path.join(folder, c) for c in csvs[:3]] + \
            [os.path.abspath(p) for p in args.verify_csv]
    res = verify(set_dir, probe, REPO)
    print(f"\n=== 読み込み検証 (別プロセス, backend={res['backend']}) ===")
    for r in res["results"]:
        print(f"  {r['file']}: 予測 {r['pred']}")

    print(f"\nモデルセット: {set_dir}")
    for f in sorted(os.listdir(set_dir)):
        print(f"  {f}  ({os.path.getsize(os.path.join(set_dir, f))} bytes)")
    print("\n配布: ./pi_bci/deploy.sh   →  Pi の「予測モデル」で選択")


if __name__ == "__main__":
    main()
