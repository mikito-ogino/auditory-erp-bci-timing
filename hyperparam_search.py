#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model.py の svm バックエンド向け、時間予算つきハイパーパラメータ探索。

`tune_svm()` は決め打ちのグリッドを最後まで回す。こちらは「何分（何時間）で終わって
ほしい」を指定してその中で当たりを探す用途で、`bci_server.py` の
「ハイパーパラメータ探索して学習」ボタンから被験者ごとに呼ばれる。

探索方法:
  * 既定は Optuna の TPE（Tree-structured Parzen Estimator）によるベイズ最適化。
    最初の `n_startup_trials` 回だけランダムに撒いて分布を作り、以降は
    「良かった組み合わせの分布 / 悪かった組み合わせの分布」の比が大きいところを
    優先して提案する。時間予算が短く試行回数が数十回しか取れない本用途では、
    一様ランダムより当たりに寄りやすい。
    `multivariate=True, group=True` にしてあるので「filtfilt のときだけ numtaps を
    振る」という条件つき探索空間もそのまま扱える。
    optuna が入っていない環境（Raspberry Pi など）では自動でランダムサーチに落ちる。
  * 現行プリセット（model.params）は必ず1回目に評価する（enqueue）ので、
    探索結果が今の設定より悪くなることは原理的にない。

評価:
  * データセット構築（フィルタ → エポック化 → 特徴量）は離散パラメータだけで決まるので、
    同じ組み合わせは一度しか作らずキャッシュする。C はここに影響しない。
  * 評価は複数の交差検証を平均した値。one-trial-out だけだと学習データが毎回 n-1 試行
    そろってしまい、過学習気味の組み合わせでも高く出る。日をまたぐ汎化（leave-one-day-out）
    と、学習データを削った k 分割（3-fold / 10-fold）を混ぜて平均する（`CV_SCHEMES`）。
    1 日分しかない・試行が少ないなど成立しない検証は自動で外し、残りの平均を使う。
    探索中は seed 1 本、最後に見つかった best だけ seed を増やして確認評価する。

CLI:
    python hyperparam_search.py --folder ./data/<subject> --budget 30m
    python hyperparam_search.py --folder ./data/<subject> --budget 30m --cv day,3fold,10fold
    python hyperparam_search.py --folder ./data/<subject> --budget 30m --sampler random
"""
import argparse
import json
import os
import re
import time
import warnings

import numpy as np

import model as m

# ==== 探索空間 ====
# baseline_length は model.py と同じく 250 Hz 基準のサンプル数で書く
# （実レートは model.resolve_params が読み替える）。
#
# フィルタは (use_filter, low_cutoff) をペアで動かす。
#   iir      : 4次バターワースのゼロ位相バンドパス（sosfiltfilt）。low_cutoff は実効値。
#   filtfilt : firwin のゼロ位相バンドパス。**ハイパスは実質効かない**ので low_cutoff は
#              名目値でしかない（下の注記を参照）。実質「0〜high_cutoff のローパス」。
#
# ※ firwin のハイパスが効かない件（実測 / fs=250Hz, high_cutoff=30Hz の DC ゲイン）:
#       numtaps=11 → 1.150,  21 → 1.043,  31 → 0.985,  41 → 0.983   (low_cutoff=0.1Hz)
#       numtaps=11 → 1.151,  21 → 1.019,  31 → 0.935,  41 → 0.914   (low_cutoff=0.5Hz)
#    どれも DC がそのまま通っており、ドリフトは一切落ちていない。0.5Hz を本当に切るには
#    numtaps≈801（DC 0.008, 必要データ長 3N=2403 点）以上が要る。
#    したがって旧探索空間の filtfilt/0.1 と filtfilt/0.5 は「ハイパス無しのローパス」を
#    ゲインだけ 7% 変えて 2 回試していたのと同じ（ゲイン差は C の変化と等価で、C は
#    別途探索している）。ここでは 1 本に統合した。ハイパス無しが有利な被験者もあり得る
#    ので選択肢としては残す。
FILTER_CHOICES = [
    {"use_filter": "iir", "low_cutoff": 0.3},
    {"use_filter": "iir", "low_cutoff": 0.5},
    {"use_filter": "iir", "low_cutoff": 1.0},
    {"use_filter": "filtfilt", "low_cutoff": 0.5},   # 実質ハイパス無効（上記注記）
]

# numtaps は filtfilt（firwin）のときだけ効く。iir では butter() が使うのは
# low/high_cutoff だけで numtaps は完全に無視されるため、iir のときは
# IIR_NUMTAPS に固定する（固定しないと同一データセットを numtaps の本数だけ
# 別物として作り直し・再評価してしまう。旧実装ではこれで離散点の 45% が無駄だった）。
IIR_NUMTAPS = 21

SEARCH_SPACE = {
    "filter": FILTER_CHOICES,
    "high_cutoff": [10, 15, 20, 30, 40],
    "numtaps": [11, 21, 31, 41],          # filtfilt のときだけ振る
    "downsampling_rate": [5, 10, 25],     # 窓を何点に落とすか（= 特徴量の時間点数）
    "baseline_length": [5, 50, 200],      # 刺激前の平均区間（20ms / 200ms / 800ms 相当）
}
C_LOG_RANGE = (-3, 2)                     # C = 10^x, x ∈ [-3, 2]

# 平均する交差検証。"day" は日ごと（leave-one-day-out）、"<k>fold" は試行を k 分割、
# "trial" は 1 試行ずつ（one-trial-out）。成立しないものは実行時に除外する。
CV_SCHEMES = ("day", "3fold", "10fold", "trial")

RNG_SEED = 0
TIME_BUDGET_SECONDS = 30 * 60
FINAL_EVAL_RESERVE_SECONDS = 120
MAX_ITER_SEARCH = 100_000
CHECKPOINT_EVERY_SECONDS = 120
N_SEED_SEARCH = 1
N_SEED_FINAL = 3
SAMPLER = "tpe"          # "tpe"（ベイズ最適化）/ "random"
N_STARTUP_TRIALS = 10    # TPE が分布を作り始めるまでのランダム試行数

# データセット構築に効くキー（= キャッシュのキー）
DATASET_KEYS = ("use_filter", "low_cutoff", "high_cutoff", "numtaps",
                "downsampling_rate", "baseline_length")


def filter_label(entry):
    """フィルタ選択肢を Optuna の categorical 値（文字列）にする。"""
    return f"{entry['use_filter']}@{entry['low_cutoff']}"


FILTER_BY_LABEL = {filter_label(f): f for f in FILTER_CHOICES}


def apply_filter_choice(p, entry):
    """フィルタのペアを p に反映し、numtaps の有効／無効も揃える。"""
    p.update(entry)
    if p["use_filter"] != "filtfilt":
        p["numtaps"] = IIR_NUMTAPS      # iir では未使用。固定してキャッシュを共有する
    return p


# ---- 使えない組み合わせの検査 ----

def invalid_reason(p, fs=None):
    """データセットを作る前に弾ける「使えない組み合わせ」なら理由を返す（無ければ None）。

    ランダム／ベイズどちらの提案でも、組み合わせ次第で成立しない値が出る。
    例外が飛んでから捨てるとその分の読み込み時間が丸ごと無駄になるので先に検査する。
    """
    fs = m.original_sampling_rate if fs is None else int(fs)
    nyq = fs / 2.0
    lo, hi = float(p["low_cutoff"]), float(p["high_cutoff"])
    if not 0 < lo < hi < nyq:
        return f"カットオフが不正: {lo}–{hi} Hz (Nyquist {nyq} Hz)"

    # compute_erp は窓を reshape(-1, factor, ch) で平均するので、割り切れないと落ちる
    win = int(round(m.WINDOW_SECONDS * fs))
    factor = win // int(p["downsampling_rate"])
    if factor < 1 or win % factor:
        return (f"downsampling_rate={p['downsampling_rate']} は窓 {win} 点を"
                f"割り切れない (factor={factor})")

    q = m.resolve_params(p, fs)
    if p.get("use_filter") in ("filtfilt", "lfilter"):
        if q["numtaps"] % 2 == 0:
            return f"firwin の numtaps={q['numtaps']} が偶数"
        if q["numtaps"] < 3:
            return f"firwin の numtaps={q['numtaps']} が短すぎる"
    if int(q["baseline_length"]) < 1:
        return f"baseline_length={q['baseline_length']} が 0 点になる"
    return None


def highpass_dc_gain(p, fs=None):
    """firwin バンドパスの DC ゲイン（1 に近いほどハイパスが効いていない）。

    iir（sosfiltfilt）では常に 0 なので None を返す。
    """
    if p.get("use_filter") not in ("filtfilt", "lfilter"):
        return None
    from scipy.signal import firwin, freqz
    fs = m.original_sampling_rate if fs is None else int(fs)
    q = m.resolve_params(p, fs)
    b = firwin(q["numtaps"],
               [p["low_cutoff"] / (fs / 2), p["high_cutoff"] / (fs / 2)],
               pass_zero=False)
    _w, h = freqz(b, worN=[0.0], fs=fs)
    return float(abs(h[0]))


# ---- パラメータの提案 ----

def sample_params(rng, base):
    """base（model.params 相当）を土台に1組ランダムサンプリングする（フォールバック用）。"""
    p = dict(base)
    entry = SEARCH_SPACE["filter"][int(rng.integers(len(SEARCH_SPACE["filter"])))]
    p["numtaps"] = SEARCH_SPACE["numtaps"][int(rng.integers(len(SEARCH_SPACE["numtaps"])))]
    apply_filter_choice(p, entry)
    for key in ("high_cutoff", "downsampling_rate", "baseline_length"):
        choices = SEARCH_SPACE[key]
        p[key] = choices[int(rng.integers(len(choices)))]
    p["C"] = float(10 ** rng.uniform(*C_LOG_RANGE))
    return p


def suggest_params(trial, base):
    """Optuna の trial から1組提案する。numtaps は filtfilt のときだけ振る。"""
    p = dict(base)
    label = trial.suggest_categorical("filter", list(FILTER_BY_LABEL))
    entry = FILTER_BY_LABEL[label]
    if entry["use_filter"] == "filtfilt":
        p["numtaps"] = trial.suggest_categorical("numtaps", SEARCH_SPACE["numtaps"])
    apply_filter_choice(p, entry)
    for key in ("high_cutoff", "downsampling_rate", "baseline_length"):
        p[key] = trial.suggest_categorical(key, SEARCH_SPACE[key])
    p["C"] = trial.suggest_float("C", 10.0 ** C_LOG_RANGE[0], 10.0 ** C_LOG_RANGE[1],
                                 log=True)
    return p


def base_as_trial(base):
    """現行プリセットを Optuna の enqueue 用の辞書にする（空間外なら None）。"""
    entry = next((f for f in FILTER_CHOICES
                  if f["use_filter"] == base.get("use_filter")
                  and float(f["low_cutoff"]) == float(base.get("low_cutoff", -1))), None)
    if entry is None:
        return None
    fixed = {"filter": filter_label(entry), "C": float(base.get("C", 1.0))}
    for key in ("high_cutoff", "downsampling_rate", "baseline_length"):
        if base.get(key) not in SEARCH_SPACE[key]:
            return None
        fixed[key] = base[key]
    if entry["use_filter"] == "filtfilt":
        if base.get("numtaps") not in SEARCH_SPACE["numtaps"]:
            return None
        fixed["numtaps"] = base["numtaps"]
    return fixed


def dataset_key(p):
    return tuple(p[k] for k in DATASET_KEYS)


def build_dataset_cached(folder, p, cache):
    """(X, tones, attended, trial_idx, filenames) をキャッシュ付きで作る。"""
    key = dataset_key(p)
    if key not in cache:
        try:
            cache[key] = m.build_epoch_dataset(folder, params_=p)
        except Exception as exc:      # noqa: BLE001 - 組み合わせが悪いだけなら飛ばす
            print(f"[search] dataset build failed for {key}: "
                  f"{type(exc).__name__}: {exc}")
            cache[key] = None
    return cache[key]


# ---- 交差検証の分割づくり ----

def file_day(name):
    """ファイル名から記録日を取り出す（"2026-07-10" 形式と "20260710" 形式の両対応）。"""
    base = os.path.basename(name)
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", base)
    if match:
        return "".join(match.groups())
    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", base)
    return match.group(1) if match else None


def file_session(name):
    """ファイル名からセッション（1 回の装着）を取り出す。

    pi_bci の記録は "<セッション開始時刻>__<試行時刻>-<ラベル>.csv" なので、
    "__" の前がセッション ID になる。持たない形式では日付で代用する。
    """
    base = os.path.basename(name)
    if "__" in base:
        return base.split("__", 1)[0]
    return file_day(name)


def trial_info(dataset):
    """試行番号 -> 注意ラベル / 日 / セッション の対応を作る。"""
    _X, _tones, attended, trial_idx, filenames = dataset
    labels, days, sessions = {}, {}, {}
    for t in sorted(set(trial_idx.tolist())):
        mask = trial_idx == t
        labels[t] = int(attended[mask][0])
        days[t] = file_day(filenames[t])
        sessions[t] = file_session(filenames[t])
    return labels, days, sessions


def _stratified_trial_folds(k, labels, seed):
    """試行を k 分割する。注意ラベルごとにばらして各フォールドのラベル比を揃える。"""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in sorted(set(labels.values())):
        trials = [t for t in sorted(labels) if labels[t] == label]
        for i, j in enumerate(rng.permutation(len(trials))):
            folds[i % k].append(trials[j])
    return [np.array(sorted(f)) for f in folds if f]


def build_folds(scheme, info, seed):
    """(folds, 説明) を返す。その検証が成立しないときは (None, 理由)。"""
    labels, days, sessions = info
    trials = sorted(labels)

    if scheme == "trial":
        return [np.array([t]) for t in trials], f"1試行ずつ({len(trials)}分割)"

    if scheme == "day":
        # 1 日分しかないときは、同じ日の中のセッション（装着）単位で代用する。
        for key, unit in ((days, "日"), (sessions, "セッション")):
            if any(key[t] is None for t in trials):
                continue
            groups = sorted({key[t] for t in trials})
            if len(groups) < 2:
                continue
            folds = [np.array([t for t in trials if key[t] == g]) for g in groups]
            return folds, f"{unit}ごと({len(groups)}{unit})"
        return None, "日もセッションも 1 つしかない"

    match = re.fullmatch(r"(\d+)fold", scheme)
    if match is None:
        raise ValueError(f"未知の交差検証: {scheme}")
    k = int(match.group(1))
    if len(trials) <= k:
        # k >= 試行数 では one-trial-out と同じになるので回さない
        return None, f"試行 {len(trials)} 本では {k} 分割にならない"
    return _stratified_trial_folds(k, labels, seed), f"{k}分割"


def describe_schemes(dataset, schemes, seed=0):
    """どの検証が使えるかを 1 行にまとめる（探索開始時のログ用）。"""
    info = trial_info(dataset)
    parts = []
    for scheme in schemes:
        folds, note = build_folds(scheme, info, seed)
        parts.append(f"{scheme}={note}" if folds else f"{scheme}=除外({note})")
    return ", ".join(parts)


def evaluate(dataset, p, n_seed, schemes=CV_SCHEMES):
    """複数の交差検証を平均した精度を返す。

    戻り値: (平均精度, {検証名: 精度}) — 成立しなかった検証はキーごと落ちる。
    """
    X, tones, attended, trial_idx, _ = dataset
    label_candidates = sorted(set(attended.tolist()))
    if len(label_candidates) < 2:
        return 0.0, {}

    info = trial_info(dataset)
    per_scheme = {}
    for scheme in schemes:
        accs = []
        for seed in range(n_seed):
            folds, _note = build_folds(scheme, info, seed)
            if folds is None:
                break
            trues, preds = m.cv_by_folds(X, tones, attended, trial_idx,
                                         label_candidates, folds, params_=p,
                                         verbose=False, seed=seed)
            if len(trues):
                accs.append(float(np.mean(trues == preds)))
        if accs:
            per_scheme[scheme] = float(np.mean(accs))

    if not per_scheme:
        return 0.0, {}
    return float(np.mean(list(per_scheme.values()))), per_scheme


def scheme_text(per_scheme):
    return " ".join(f"{k}={v:.3f}" for k, v in per_scheme.items())


def save_checkpoint(path, results, it, elapsed):
    ranked = sorted(results, key=lambda r: r["cv_accuracy"], reverse=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"iterations": it, "elapsed_s": round(elapsed, 1),
                   "n_results": len(results), "top10": ranked[:10]},
                  f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_search(folder, time_budget_seconds,
               final_reserve_seconds=FINAL_EVAL_RESERVE_SECONDS,
               n_seed_search=N_SEED_SEARCH, n_seed_final=N_SEED_FINAL,
               rng_seed=RNG_SEED, max_iter=MAX_ITER_SEARCH,
               checkpoint_path=None, checkpoint_every=CHECKPOINT_EVERY_SECONDS,
               log_every=15, log=print, use_sequence_override=None,
               base_params=None, cv_schemes=CV_SCHEMES,
               sampler=SAMPLER, n_startup_trials=N_STARTUP_TRIALS):
    """`folder` の CSV で `time_budget_seconds` 以内に探索を終える。

    最終確認評価と保存の時間 (`final_reserve_seconds`) も予算に含める。
    `bci_server.py` の被験者ごと学習と、このファイルの CLI 実行の両方から使う。

    `sampler="tpe"` なら Optuna のベイズ最適化、`"random"` ならランダムサーチ。
    optuna が入っていない環境では自動でランダムに落ちる。

    評価は `cv_schemes` の交差検証を平均した値（成立しないものは自動で除外）。

    戻り値: {"params", "cv_accuracy_search", "cv_accuracy_final", "n_files",
             "iterations", "top10", "use_sequence", "sampling_rate",
             "cv_schemes", "cv_per_scheme_search", "cv_per_scheme_final",
             "sampler", "n_skipped"}
    """
    prev_use_sequence = m.use_sequence
    if use_sequence_override is not None:
        m.use_sequence = int(use_sequence_override)
    base = dict(m.params if base_params is None else base_params)

    try:
        cache = {}
        results = []
        state = {"fs": m.original_sampling_rate, "n_files": 0,
                 "schemes_logged": False, "skipped": 0, "iters": 0}

        cv_schemes = tuple(cv_schemes)
        search_budget = max(0.0, time_budget_seconds - final_reserve_seconds)

        sampler = (sampler or "tpe").lower()
        optuna = None
        if sampler == "tpe":
            try:
                import optuna as _optuna
                optuna = _optuna
            except Exception as exc:      # noqa: BLE001
                log(f"[search] optuna が使えないためランダムサーチにします "
                    f"({type(exc).__name__}: {exc})")
                sampler = "random"

        name = "ベイズ最適化(TPE)" if sampler == "tpe" else "ランダムサーチ"
        log(f"[search] {folder} を{name}（最大 {search_budget/60:.1f} 分, "
            f"CV {'+'.join(cv_schemes)} の平均, seed {n_seed_search} 本）")
        # ハイパスが効かないフィルタ設定は結果の解釈を誤らせるので一度だけ知らせる
        for entry in FILTER_CHOICES:
            probe = apply_filter_choice(dict(base), dict(entry))
            probe["high_cutoff"] = SEARCH_SPACE["high_cutoff"][-1]
            gain = highpass_dc_gain(probe, state["fs"])
            if gain is not None and gain > 0.5:
                log(f"[search] 注意: {filter_label(entry)} は numtaps"
                    f"{SEARCH_SPACE['numtaps']} の範囲では DC ゲイン ~{gain:.2f} で"
                    f"ハイパスが効きません（実質 0–{probe['high_cutoff']}Hz ローパス）")

        t0 = time.time()
        marks = {"log": t0, "ckpt": t0}

        def score_params(p):
            """1 組を評価して (score, per_scheme) を返す。使えない組み合わせは None。"""
            state["iters"] += 1
            bad = invalid_reason(p, state["fs"])
            if bad:
                state["skipped"] += 1
                log(f"[search] skip: {bad}")
                return None
            dataset = build_dataset_cached(folder, p, cache)
            if dataset is None:
                state["skipped"] += 1
                return None
            state["fs"] = m.original_sampling_rate      # 実レートが判明した
            state["n_files"] = len(dataset[4])
            if not state["schemes_logged"]:
                log(f"[search] 試行 {len(dataset[4])} 本の内訳: "
                    f"{describe_schemes(dataset, cv_schemes)}")
                state["schemes_logged"] = True
            score, per_scheme = evaluate(dataset, p, n_seed_search, cv_schemes)
            results.append({**{k: p[k] for k in DATASET_KEYS},
                            "C": p["C"], "cv_accuracy": score,
                            "cv_per_scheme": per_scheme})
            return score, per_scheme

        def periodic_report():
            now = time.time()
            if results and now - marks["log"] >= log_every:
                best = max(results, key=lambda r: r["cv_accuracy"])
                log(f"[search] it={state['iters']} 組合せ={len(cache)} "
                    f"best={best['cv_accuracy']:.3f} "
                    f"経過 {now-t0:.0f}s / {search_budget:.0f}s")
                marks["log"] = now
            if checkpoint_path and results and now - marks["ckpt"] >= checkpoint_every:
                save_checkpoint(checkpoint_path, results, state["iters"], now - t0)
                marks["ckpt"] = now

        if sampler == "tpe":
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            with warnings.catch_warnings():
                # multivariate/group は optuna では experimental 扱いで毎回警告が出る。
                # 学習ログが読みにくくなるだけなので黙らせる。
                warnings.simplefilter("ignore")
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(
                        seed=rng_seed, n_startup_trials=n_startup_trials,
                        multivariate=True, group=True),
                )
            # 現行プリセットを 1 回目に評価する（探索が今の設定を下回らないように）
            seed_trial = base_as_trial(base)
            if seed_trial:
                study.enqueue_trial(seed_trial, skip_if_exists=True)
                log(f"[search] 1回目は現行設定を評価: {seed_trial}")

            def objective(trial):
                out = score_params(suggest_params(trial, base))
                if out is None:
                    raise optuna.TrialPruned()
                score, per_scheme = out
                trial.set_user_attr("cv_per_scheme", per_scheme)
                return score

            def callback(_study, _trial):
                periodic_report()

            study.optimize(objective, n_trials=max_iter, timeout=search_budget,
                           callbacks=[callback], catch=())
        else:
            rng = np.random.default_rng(rng_seed)
            while state["iters"] < max_iter and time.time() - t0 < search_budget:
                score_params(sample_params(rng, base))
                periodic_report()

        if not results:
            raise RuntimeError("有効なパラメータの組み合わせが見つかりませんでした"
                               "（データが少なすぎる／時間予算が短すぎる可能性）")

        results.sort(key=lambda r: r["cv_accuracy"], reverse=True)
        if checkpoint_path:
            save_checkpoint(checkpoint_path, results, state["iters"], time.time() - t0)
        log(f"[search] 探索終了: {state['iters']} 回（うち不成立 {state['skipped']} 回）, "
            f"有効 {len(results)} 件, データセット {len(cache)} 通り, "
            f"{time.time()-t0:.0f}s")

        best = results[0]
        best_params = dict(base)
        best_params.update({k: best[k] for k in DATASET_KEYS})
        best_params["C"] = best["C"]

        # 探索中の best は「たくさん試した中の最大値」なので選択バイアスで
        # 楽観側に出る。seed を増やして測り直した値も一緒に返す。
        log(f"[search] best を seed {n_seed_final} 本で確認評価中…")
        final, final_per = evaluate(build_dataset_cached(folder, best_params, cache),
                                    best_params, n_seed_final, cv_schemes)
        log(f"[search] best={best_params}")
        log(f"[search] 確認評価 CV 平均 = {final:.3f}"
            f"（探索時 {best['cv_accuracy']:.3f}） {scheme_text(final_per)}")

        return {
            "params": best_params,
            "cv_accuracy_search": best["cv_accuracy"],
            "cv_accuracy_final": final,
            "n_files": state["n_files"],
            "iterations": state["iters"],
            "n_skipped": state["skipped"],
            "sampler": sampler,
            "top10": results[:10],
            "use_sequence": m.use_sequence,
            "sampling_rate": m.original_sampling_rate,
            "cv_schemes": list(final_per.keys()),
            "cv_per_scheme_search": best.get("cv_per_scheme", {}),
            "cv_per_scheme_final": final_per,
        }
    finally:
        m.use_sequence = prev_use_sequence


def parse_budget(text):
    """"30m" / "2h" / "900" を秒に直す。"""
    text = str(text).strip().lower()
    if text.endswith("h"):
        return float(text[:-1]) * 3600
    if text.endswith("m"):
        return float(text[:-1]) * 60
    if text.endswith("s"):
        return float(text[:-1])
    return float(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default=m.data_folder, help="学習データのフォルダ")
    ap.add_argument("--budget", default="30m", help="探索の時間予算（例: 30m, 2h）")
    ap.add_argument("--out", default=None, help="結果 JSON の保存先")
    ap.add_argument("--train", action="store_true",
                    help="見つかったパラメータで最終モデルまで学習する")
    ap.add_argument("--use-sequence", type=int, default=None,
                    help="1試行あたりの刺激セット数")
    ap.add_argument("--cv", default=",".join(CV_SCHEMES),
                    help="平均する交差検証（例: day,3fold,10fold,trial）")
    ap.add_argument("--sampler", default=SAMPLER, choices=("tpe", "random"),
                    help="探索方法（tpe=ベイズ最適化 / random=ランダムサーチ）")
    ap.add_argument("--n-startup", type=int, default=N_STARTUP_TRIALS,
                    help="TPE が分布を作り始めるまでのランダム試行数")
    args = ap.parse_args()

    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hyperparam_search_checkpoint.json")
    result = run_search(args.folder, parse_budget(args.budget),
                        checkpoint_path=ckpt,
                        use_sequence_override=args.use_sequence,
                        cv_schemes=[c for c in args.cv.split(",") if c.strip()],
                        sampler=args.sampler, n_startup_trials=args.n_startup)

    print("\nTop 10:")
    for r in result["top10"]:
        print(f"  acc={r['cv_accuracy']:.3f}  filter={r['use_filter']:>8} "
              f"lc={r['low_cutoff']} hc={r['high_cutoff']} nt={r['numtaps']} "
              f"ds={r['downsampling_rate']} bl={r['baseline_length']} "
              f"C={r['C']:.4g}  [{scheme_text(r.get('cv_per_scheme', {}))}]")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "best_params.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[search] saved -> {out}")

    if args.train:
        stats = m.train(data_folder_override=args.folder,
                        params_=result["params"],
                        use_sequence_override=result.get("use_sequence"))
        print(f"[search] final model: {stats}")


if __name__ == "__main__":
    main()
