#%%
import os
import tempfile
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.signal import firwin, lfilter, find_peaks
from scipy.signal import filtfilt
import re
import joblib
import pickle
from collections import Counter
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from scipy.signal import butter, sosfiltfilt

# ==== バックエンド選択フラグ ====
# "svm"    → 本ファイルの FIR + LinearSVC パイプライン（既存）
# "legacy" → model_legacy.Model のパイプライン（LASSO / SKLDA / ElasticNet / SVM）
# 環境変数 MODEL_BACKEND でも上書き可能（例: MODEL_BACKEND=legacy python model.py）
BACKEND = os.environ.get("MODEL_BACKEND", "legacy")

# データフォルダのパス（記録 CSV を置いたフォルダ）
data_folder = "./data"
file_prefix = ""  # 解析対象とするファイル名の先頭文字列（空文字なら全ファイル）

# ==== サンプリングレート ====
# Cyton は 250 Hz、Polymate Pocket は 1000 Hz。CSV の Timestamp から実測して
# 読み込みのたびに切り替える（load_filtered_eeg が set_sampling_rate を呼ぶ）ので、
# ここの値は「CSV を読まずに使うときの既定」でしかない。
#
# params の 'baseline_length' と 'numtaps' は「サンプル数」指定なので、
# レートが変わると意味する時間が変わってしまう。既存のプリセット・チューニング結果を
# そのまま使えるよう、値は 250 Hz 基準のまま置いておき、resolve_params() で
# 実レートに読み替える（= 時間長を保つ）。
REFERENCE_SAMPLING_RATE = 250       # params の「サンプル数」指定の基準レート
WINDOW_SECONDS = 1.0                # 刺激後の解析窓（秒）。窓長 = fs × これ
KNOWN_SAMPLING_RATES = (250, 500, 1000, 2000)

original_sampling_rate = 250
window_size = 250
diff_downsampling_rate = 25
use_sequence = 10
# FIRフィルタの遅れを補正するための変数
fir_delay = 0


def atomic_save(path, write):
    """同じディレクトリの一時ファイルに書いてから置き換える。

    joblib.dump / pickle.dump / np.save は出力先を先に truncate するので、
    書き込み中に失敗すると「壊れたファイル」がその場に残る。モデルの pkl だと
    bci_server のモデル一覧には「モデルあり」と出たまま、実験中の予測で初めて
    落ちる。同じディレクトリに書いてから os.replace で差し替えれば、
    成功したときだけ中身が入れ替わり、失敗しても元のファイルが残る。

    write は一時ファイルのパスを受け取って書き込む関数。
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # 拡張子は残す（np.save は .npy が付いていないと勝手に足してしまう）
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_",
                               suffix=os.path.basename(path))
    os.close(fd)
    try:
        write(tmp)
        # mkstemp は 0600 で作る。置き換えた後も従来通り読めるようにしておく。
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:       # 中断（Ctrl-C）でも書きかけを残さない
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def save_pickle(path, obj):
    """pickle をアトミックに書く（atomic_save 参照）。"""
    def write(tmp):
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
    return atomic_save(path, write)


def set_sampling_rate(fs):
    """解析レートを切り替える（窓長も追随させる）。"""
    global original_sampling_rate, window_size
    fs = int(round(fs))
    original_sampling_rate = fs
    window_size = int(round(WINDOW_SECONDS * fs))
    return fs


def detect_sampling_rate(df, default=None):
    """CSV の Timestamp 列から実サンプリングレートを推定して既知の値に丸める。

    実測値をそのまま使うと 999.6 Hz のような端数になり、窓長がファイルごとに
    1 サンプルずれて特徴量の次元が揃わなくなる。既知レートに寄せて防ぐ。
    """
    default = original_sampling_rate if default is None else default
    if "Timestamp" not in getattr(df, "columns", []):
        return default
    t = np.asarray(df["Timestamp"].values, dtype=float)
    if len(t) < 2:
        return default
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return default
    fs = 1.0 / dt
    return min(KNOWN_SAMPLING_RATES, key=lambda k: abs(k - fs))


def resolve_params(p, fs=None):
    """params の「サンプル数」指定を実レート向けに読み替えた辞書を返す。

    baseline_length（刺激前の平均区間）と numtaps（FIR のタップ数）は
    REFERENCE_SAMPLING_RATE 基準で書かれているので、時間長が変わらないよう
    レート比で伸縮する。低次カットオフは正規化周波数で使われるため触らない。
    """
    fs = original_sampling_rate if fs is None else int(round(fs))
    if fs == REFERENCE_SAMPLING_RATE:
        return dict(p)
    ratio = fs / REFERENCE_SAMPLING_RATE
    q = dict(p)
    q['baseline_length'] = max(1, int(round(p['baseline_length'] * ratio)))
    if 'numtaps' in p:
        taps = int(round(p['numtaps'] * ratio))
        q['numtaps'] = taps + 1 if taps % 2 == 0 else taps   # firwin は奇数が無難
    return q

model_path = "svm_model.pkl"
scaler_path = "scaler.pkl"

# ==== svm バックエンドの前処理プリセット ====
# "current"      → 旧設定（バックアップ）。firwin(numtaps=21, 0.1Hz) は DC gain≈1.043 で
#                  ハイパスが実質まったく効いていない（実測: DC/0.1/0.5/1Hz すべて 1.043）。
# "filter_fixed" → 既定。current からフィルタだけを 0.5Hz の IIR ハイパス（4次バターワース,
#                  sosfiltfilt）に差し替えた版。DC gain 0.00000 で legacy と同じドリフト処理。
#                  ベースライン長・窓長・ダウンサンプルは current と同一（フィルタのみ変更）。
# "improved"     → filter_fixed に加えてベースラインを刺激前 800ms にした版。
# 環境変数 SVM_PRESET でも上書き可能。
# ※旧ベースライン値（interface_with_raspberrypi/CLAUDE.md の「元 model.py 手法」）を
#   再現したいときは SVM_PRESET=current を指定する。
# ※プリセットを切り替えたら train_svm() で再学習が必要（特徴量が変わるため）。
SVM_PRESET = os.environ.get("SVM_PRESET", "filter_fixed")

params_current = {
    'downsampling_rate': 10,
    'numtaps': 21,
    'low_cutoff': 0.1,
    'high_cutoff': 30,
    'baseline_length': 5,
    'epoch_pattern': 7,
    'C': 1,
    'use_filter': "filtfilt",   # "filtfilt" / "lfilter" / "iir" / None
    'balance': True,            # target/non-target の学習数を最小クラス数に揃える
    'balance_global': True,     # さらに全分類器で同じ枚数に揃える
    'balance_trials': True,     # 注意ラベルごとの試行数も最小数に揃える
    'prob_method': "platt_trial",  # predict_proba で尤度比較（"platt_trial"/"platt"/"logreg"）
}

params_filter_fixed = {
    'downsampling_rate': 10,
    'numtaps': 21,               # use_filter="iir" のときは未使用
    'low_cutoff': 0.5,           # ← ここだけが current との差
    'high_cutoff': 30,
    'baseline_length': 5,
    'epoch_pattern': 7,
    'C': 1,
    'use_filter': "iir",         # ← ここだけが current との差
    'balance': True,             # target/non-target の学習数を最小クラス数に揃える
    'balance_global': True,      # さらに全分類器で同じ枚数に揃える
    'balance_trials': True,      # 注意ラベルごとの試行数も最小数に揃える
    'prob_method': "platt_trial",   # predict_proba で尤度比較（"platt_trial"/"platt"/"logreg"）
}

params_improved = {
    'downsampling_rate': 10,
    'numtaps': 21,               # use_filter="iir" のときは未使用
    'low_cutoff': 0.5,
    'high_cutoff': 30,
    'baseline_length': 200,      # 刺激前 800ms（250Hz × 0.8s）
    'epoch_pattern': 7,
    'C': 1,
    'use_filter': "iir",
    'balance': True,             # target/non-target の学習数を最小クラス数に揃える
    'balance_global': True,      # さらに全分類器で同じ枚数に揃える
    'balance_trials': True,      # 注意ラベルごとの試行数も最小数に揃える
    'prob_method': "platt_trial",   # predict_proba で尤度比較（"platt_trial"/"platt"/"logreg"）
}

svm_presets = {
    'current': params_current,
    'filter_fixed': params_filter_fixed,
    'improved': params_improved,
}

if SVM_PRESET not in svm_presets:
    raise ValueError(f"Unknown SVM_PRESET: {SVM_PRESET}")
params = svm_presets[SVM_PRESET]

# ==== legacy バックエンド（model_legacy.Model）用の設定 ====
legacy_clfs_path = "clfs.pkl"
legacy_model_path = "model.pkl"

legacy_params = {
    'kind': "LASSO",            # "LASSO" / "SKLDA" / "ElasticNet" / "SVM"
    'LASSO_param': 0.013627698441233689,
    'decimation_value': 25,
    'upper_frequency': 120,
    'start_eeg': -200,          # ベースライン区間の長さ（サンプル、負値）
    'end_eeg': 250,             # 刺激後の解析区間（サンプル）
    'testSequence': use_sequence,
    'useChannel': [1, 2, 3, 4, 5, 6, 7, 8],  # legacy の filtering は Ch1 から連続指定のみ正しく動作
    'label_candidates': None,   # None なら学習データのターゲットから自動決定
    'evaluateTrial': True,      # 学習時に one-trial-out LOOCV を実行
    'graph': False,
    'acc': False,
}


# ==== EXT1/EXT2（刺激音そのもの）からのオンセット検出 ====
# Polymate Pocket は補助入力に刺激音のステレオ信号をそのまま録れる。Label 列は
# 「DAC に出た時刻に最も近い行」に後付けするものなので、音響波形から取ったオンセットの
# ほうが真値に近い。EXT 列がある CSV では既定でこちらを使う。
USE_EXT_ONSET = True        # False にすると EXT 列があっても Label 列を使う

ENV_WIN_MS = 30      # エンベロープの移動RMS窓 [ms]
MIN_DIST_MS = 300    # ピーク間の最小距離 [ms]（SOA 500ms より短く取る）
SNR_FACTOR = 5       # height 閾値 = ノイズフロア × これ
PROM_FACTOR = 3      # prominence 閾値 = ノイズフロア × これ
MIN_ABS_THR = 1.0    # 絶対的な最低閾値


def detect_onsets_ext(df, fs):
    """EXT1/EXT2 のステレオ音響信号からオンセット位置と提示音ラベルを返す。

    ラベルは EXT1/EXT2 の相対エネルギーから決める（Label 列は使わない）。
    Label 列と突き合わせて検証した結果、EXT1/EXT2 は実際の左右チャンネルに対して
    入れ替わって記録されていることが分かっているので、次のように対応させる:
      Stim 1 (left)   : EXT2 >> EXT1 → label 1
      Stim 2 (center) : EXT1 ≈ EXT2  → label 2
      Stim 3 (right)  : EXT1 >> EXT2 → label 3
      Stim 4 (silence): 音が無いので検出されない（エポックも作られない）
    """
    ext1 = np.asarray(df["EXT1"].values, dtype=float)
    ext2 = np.asarray(df["EXT2"].values, dtype=float)

    # DC 除去（中央値。無音区間が半分以上あることを前提にする）
    ext1_ac = ext1 - np.median(ext1)
    ext2_ac = ext2 - np.median(ext2)

    win = max(1, int(ENV_WIN_MS / 1000 * fs))
    kernel = np.ones(win) / win
    env1 = np.sqrt(np.convolve(ext1_ac ** 2, kernel, mode="same"))
    env2 = np.sqrt(np.convolve(ext2_ac ** 2, kernel, mode="same"))
    env_sum = env1 + env2

    noise_floor = np.median(env_sum)
    thr_height = max(noise_floor * SNR_FACTOR, MIN_ABS_THR)
    thr_prom = max(noise_floor * PROM_FACTOR, MIN_ABS_THR * 0.5)
    min_dist = max(1, int(MIN_DIST_MS / 1000 * fs))

    peaks, _ = find_peaks(env_sum, height=thr_height, distance=min_dist,
                          prominence=thr_prom)

    # ピークから遡り、閾値を超え始めた点をオンセットとする
    onset_thr = noise_floor * 2
    back = int(0.20 * fs)
    onset_indices = []
    for pk in peaks:
        start = max(0, pk - back)
        seg = env_sum[start:pk + 1]
        below = np.where(~(seg > onset_thr))[0]
        onset_indices.append(start + (below[-1] + 1 if len(below) else 0))
    onset_indices = np.array(onset_indices, dtype=int)

    if len(onset_indices) == 0:
        return onset_indices, np.array([], dtype=int)

    # 音の鳴っている区間の左右エネルギー比でラベルを決める
    sound_samples = max(1, int(0.15 * fs))
    RATIO_THR = 3.0
    labels = np.zeros(len(onset_indices), dtype=int)
    for i, oi in enumerate(onset_indices):
        end = min(oi + sound_samples, len(ext1_ac))
        e1 = np.sqrt(np.mean(ext1_ac[oi:end] ** 2)) + 1e-10
        e2 = np.sqrt(np.mean(ext2_ac[oi:end] ** 2)) + 1e-10
        if e1 > RATIO_THR * e2:
            labels[i] = 3        # EXT1 優勢 = 実際は右チャンネル
        elif e2 > RATIO_THR * e1:
            labels[i] = 1        # EXT2 優勢 = 実際は左チャンネル
        else:
            labels[i] = 2        # 両ch = センター
    return onset_indices, labels


# ファイル名末尾の数字からラベルを取得する関数
# 命名規則が2種類あるため両方に対応する（例: ..._1.csv / ...-1.csv）
def extract_label(file_name):
    stem = os.path.basename(file_name).rsplit(".", 1)[0]
    match = re.search(r"[_-](\d+)$", stem)
    if match is None:
        raise ValueError(f"Cannot extract label from file name: {file_name}")
    return int(match.group(1))

# 各刺激に対する事象関連電位を計算する関数（エポック毎のサンプルを返す）
def compute_erp(data, stimulus_onsets, params):
    # baseline_length は 250 Hz 基準のサンプル数なので、実レートに読み替える
    params = resolve_params(params)
    downsampling_rate = params['downsampling_rate']
    numtaps = params['numtaps']
    low_cutoff = params['low_cutoff'] / (original_sampling_rate / 2)
    high_cutoff = params['high_cutoff'] / (original_sampling_rate / 2)
    baseline_length = params['baseline_length']

    downsampling_factor = window_size // downsampling_rate
    erp = []
    features = []
    for onset in stimulus_onsets:
        if onset + window_size <= data.shape[0]:
            # ベースライン区間を計算
            baseline_start = max(0, onset - baseline_length - fir_delay)
            baseline_end = onset - fir_delay
            baseline = data[baseline_start:baseline_end].mean(axis=0)
            segment = data[onset - fir_delay:onset + window_size - fir_delay] - baseline
            # # 区間を平均してダウンサンプリング
            downsampled_segment = segment.reshape(-1, downsampling_factor, segment.shape[1]).mean(axis=1)
            
            # diff_downsampled_segment = segment.reshape(-1, diff_downsampling_factor, segment.shape[1]).mean(axis=1)
            
            if False:  # 変化率特徴量を計算する場合はTrueに変更
                # 変化率特徴量を計算
                rate_of_change = np.diff(diff_downsampled_segment, axis=0)

                # 刺激後1秒間のデータを取得
                combined_features = np.concatenate((downsampled_segment, rate_of_change), axis=0)
                features.append(combined_features)
            else:
                features.append(downsampled_segment)
                
    return np.array(features)

# CSV を読み込み、フィルタ済み EEG と刺激オンセットを返す関数（キャッシュ付き）
def load_filtered_eeg(file_path, params):
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)
    data = df.values

    # このファイルの実サンプリングレートに合わせる（Cyton 250Hz / Polymate 1000Hz）
    fs = set_sampling_rate(detect_sampling_rate(df))
    params = resolve_params(params, fs)
    numtaps = params['numtaps']
    low_cutoff = params['low_cutoff'] / (original_sampling_rate / 2)
    high_cutoff = params['high_cutoff'] / (original_sampling_rate / 2)

    # 刺激ラベルの列を取得（EXT列が増えると列位置が動くので名前で引く）
    stimulus_labels = np.nan_to_num(
        np.asarray(df["Label"].values, dtype=float), nan=0)

    # 刺激オンセットの取得。EXT1/EXT2 に刺激音そのものが入っているならそちらから
    # 検出する（Label 列は最近傍の行に後付けしたものなので誤差を含む）。
    onset_src = "label"
    if USE_EXT_ONSET and "EXT1" in df.columns and "EXT2" in df.columns:
        stimulus_onsets, onset_labels = detect_onsets_ext(df, fs)
        if len(stimulus_onsets):
            onset_src = "ext"
            # 以降は「オンセット行にだけラベルが立っている配列」として扱う。
            # 呼び出し側は stimulus_labels[stimulus_onsets] でラベルを引くので、
            # ここで検出結果を書き戻せば Label 列由来と同じ形で流せる。
            stimulus_labels = np.zeros(len(df), dtype=float)
            stimulus_labels[stimulus_onsets] = onset_labels
    if onset_src == "label":
        stimulus_onsets = np.where(np.diff(stimulus_labels) != 0)[0] + 1
        stimulus_onsets = stimulus_onsets[stimulus_labels[stimulus_onsets] != 0]

    # 1試行あたり use_sequence セット分だけを使う。EXT 検出では無音（音4）が
    # 挙がらないので 1 セット 3 個、Label 由来なら 4 個で数える。
    stimulus_onsets = stimulus_onsets[:(3 if onset_src == "ext" else 4) * use_sequence]

    # 保存フォルダのパス
    eeg_cache_folder = "./eeg_cache"
    if not os.path.exists(eeg_cache_folder):
        os.makedirs(eeg_cache_folder)

    # フィルタリング方法（プリセットで切り替え。未指定なら従来通り filtfilt）
    use_filter = params.get('use_filter', "filtfilt")

    # キャッシュファイル名を生成（フィルタ種別・タップ数もキーに含める）
    # レートが違えば波形も変わるので fs もキーに含める（古いキャッシュの誤用防止）
    # カットオフは「Hz のまま」書く。正規化値（lc/(fs/2)）を 3 桁に丸めると
    # fs=1000Hz では 0.3Hz も 0.5Hz も 0.001 になって別設定が同じファイル名に
    # 衝突し、片方がもう片方のフィルタ済みデータを黙って使い回してしまう。
    # フォルダ名も、data_folder（モジュール変数）ではなくこのファイル自身の
    # 親フォルダから取る（build_epoch_dataset に別フォルダを渡したときの取り違え防止）。
    folder_tag = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    cache_name = (
        f"{folder_tag}_{os.path.basename(file_path).replace('.csv','')}"
        f"_lc{params['low_cutoff']:g}Hz_hc{params['high_cutoff']:g}Hz"
        f"_{use_filter}{numtaps}_fs{fs}.npy"
    )
    cache_path = os.path.join(eeg_cache_folder, cache_name)

    if os.path.exists(cache_path):
        eeg_data = np.load(cache_path)
    else:
        # バンドパスフィルタの設計（FIR系のときのみ使用）
        if use_filter in ("filtfilt", "lfilter"):
            b = firwin(numtaps, [low_cutoff, high_cutoff], pass_zero=False)

        # EEGデータを取得（多極対応）
        use_all_channels = True  # Trueの場合はCh1〜Ch8を使用、Falseの場合は1つだけ指定
        if use_all_channels:
            selected_channels = [f"Ch{i}" for i in range(1, 9)]  # Ch1〜Ch8を使用
        else:
            selected_channels = ["Ch5"]  # ここで使用するCHを指定
        eeg_data = data[:, [i for i, col in enumerate(df.columns) if col in selected_channels]] * -1
        # EEGデータを標準化
        eeg_data = StandardScaler().fit_transform(eeg_data)

        if use_filter == "filtfilt":
            eeg_data = filtfilt(b, 1.0, eeg_data, axis=0)
        elif use_filter == "lfilter":
            eeg_data = lfilter(b, 1.0, eeg_data, axis=0)
        elif use_filter == "iir":
            sos = butter(4, [params["low_cutoff"], params["high_cutoff"]], btype='bandpass', fs=original_sampling_rate, output='sos')
            eeg_data = sosfiltfilt(sos, eeg_data, axis=0)
        elif use_filter is None:
            pass

        atomic_save(cache_path, lambda tmp: np.save(tmp, eeg_data))

    # 窓が記録末尾に収まらないオンセットは落とす（EXTの余分検出・末尾不足を吸収）
    valid = (stimulus_onsets + window_size) <= eeg_data.shape[0]
    if not np.all(valid):
        print(f"[model] {int((~valid).sum())} onset(s) dropped (窓が末尾に収まらない): "
              f"{os.path.basename(file_path)}")
        stimulus_onsets = stimulus_onsets[valid]

    return eeg_data, stimulus_onsets, stimulus_labels


# データ処理と特徴量抽出を行う関数
def process_file(file_path, params, use_mean_features=True):
    epoch_pattern = params['epoch_pattern']

    eeg_data, stimulus_onsets, stimulus_labels = load_filtered_eeg(file_path, params)

    # 刺激ラベルの0を削除し、連続するものを1つにする
    processed_stimulus_labels = stimulus_labels[stimulus_labels != 0]
    processed_stimulus_labels = np.concatenate(([processed_stimulus_labels[0]], processed_stimulus_labels[1:][processed_stimulus_labels[1:] != processed_stimulus_labels[:-1]]))

    # エポック毎のERPを計算
    erp_epochs = compute_erp(eeg_data, stimulus_onsets, params)

    features = []
    if use_mean_features:
        # 刺激ラベルが1と3のエポックを分ける
        erp_label_1 = erp_epochs[stimulus_labels[stimulus_onsets] == 1]
        erp_label_2 = erp_epochs[stimulus_labels[stimulus_onsets] == 2]
        erp_label_3 = erp_epochs[stimulus_labels[stimulus_onsets] == 3]
        erp_label_4 = erp_epochs[stimulus_labels[stimulus_onsets] == 4]

        # 平均を計算
        mean_erp_label_1 = np.mean(erp_label_1, axis=0)
        mean_erp_label_2 = np.mean(erp_label_2, axis=0)
        mean_erp_label_3 = np.mean(erp_label_3, axis=0)
        mean_erp_label_4 = np.mean(erp_label_4, axis=0)

        # ラベルごとの flatten 済み特徴量をリストにまとめる（インデックス 0 → label_1）
        label_features = [
            mean_erp_label_1.flatten(),
            mean_erp_label_2.flatten(),
            mean_erp_label_3.flatten(),
            mean_erp_label_4.flatten(),
        ]

        # 差を計算して特徴量に追加
        if(0):
            diff_erp = (mean_erp_label_1 - mean_erp_label_3).flatten()
        
        # 各音に対する特徴量を統合
        if(1):
            # epoch_pattern に対応する特徴量のインデックスを定義
            epoch_combinations = {
                1: [0],
                2: [1],
                3: [2],
                4: [0, 2],
                5: [0, 1],
                6: [1, 2],
                7: [0, 1, 2],
                8: [0, 1, 2, 3],
            }

            # epoch_pattern を指定して diff_erp を生成
            if epoch_pattern in epoch_combinations:
                selected_features = [label_features[i] for i in epoch_combinations[epoch_pattern]]
                diff_erp = np.concatenate(selected_features)
            else:
                raise ValueError(f"Unknown epoch_pattern: {epoch_pattern}")
            # diff_erp = np.concatenate((mean_erp_label_1.flatten(), mean_erp_label_3.flatten()))

        # 刺激ラベルを特徴量に追加
        if(0):
            diff_erp = np.concatenate((diff_erp, processed_stimulus_labels[:len(diff_erp)]))

        features.append(diff_erp)

    else:
        # 刺激ラベルが1と3のエポックを分ける
        erp_label_1 = erp_epochs[stimulus_labels[stimulus_onsets] == 1]
        erp_label_3 = erp_epochs[stimulus_labels[stimulus_onsets] == 3]
        
        # 各エポックの電極平均を計算して特徴量に追加
        for epoch_1, epoch_3 in zip(erp_label_1, erp_label_3):
            mean_epoch_1 = np.mean(epoch_1, axis=1)  # 電極平均
            mean_epoch_3 = np.mean(epoch_3, axis=1)  # 電極平均
            concatenated_erp = np.concatenate((
                mean_epoch_1.flatten(), 
                mean_epoch_3.flatten()
            ))
            features.append(concatenated_erp)
    
    return np.vstack(features)

# ==== svm バックエンド: エポック単位 target / non-target 定式化 ====
# legacy（model_legacy.Model）に寄せた定式化。
#   学習: エポックを平均しない。提示音ごとに「その音が注意音か（target=1）/ 否か（non-target=0）」
#         の二値分類器を別々に作る。1ファイル40エポックがそのまま学習サンプルになる。
#   予測: テスト試行のエポックを提示音ごとに平均し、各音の分類器が返す
#         「その音が注意音である確率」(predict_proba) を比較して argmax を注意音とする。
# 旧定式化（process_file: 1ファイル=1サンプル・音1〜3の平均ERPを連結した多クラス分類）は
# process_file 側にそのまま残してある（epoch_pattern / use_mean_features は旧定式化専用）。

def extract_epochs(file_path, params):
    """1ファイルから エポック単位の特徴量 (n_epoch, dim) と提示音ラベル (n_epoch,) を返す"""
    eeg_data, stimulus_onsets, stimulus_labels = load_filtered_eeg(file_path, params)
    epochs = compute_erp(eeg_data, stimulus_onsets, params)
    # compute_erp は窓が記録末尾を超えるオンセットを末尾から落とすので、同数に切り詰めて対応させる
    tones = stimulus_labels[stimulus_onsets][:len(epochs)].astype(int)
    return epochs.reshape(len(epochs), -1), tones


def build_epoch_dataset(folder=None, prefix=None, params_=None):
    """フォルダ内の全 CSV をエポック単位で読み込む

    戻り値: (X, tones, attended, trial_idx, filenames)
      X         : (n_epoch, dim) エポック特徴量
      tones     : (n_epoch,)     そのエポックで提示された音
      attended  : (n_epoch,)     そのファイルの注意音（= 正解ラベル）
      trial_idx : (n_epoch,)     ファイル（試行）番号。one-trial-out の分割単位
    """
    folder = data_folder if folder is None else folder
    prefix = file_prefix if prefix is None else prefix
    params_ = params if params_ is None else params_

    X, tones, attended, trial_idx, filenames = [], [], [], [], []
    rates = {}          # 実サンプリングレート -> そのレートのファイル
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if not file.endswith(".csv"):
                continue
            if not file.startswith(prefix):
                continue
            if file.endswith("impedance.csv"):
                continue
            label = extract_label(file)
            ep, tn = extract_epochs(os.path.join(root, file), params_)
            # extract_epochs → load_filtered_eeg がこのファイルのレートを反映済み
            rates.setdefault(original_sampling_rate, []).append(file)
            if len(ep) == 0:
                print(f"skip (no epoch): {file}")
                continue
            X.append(ep)
            tones.append(tn)
            attended.extend([label] * len(ep))
            trial_idx.extend([len(filenames)] * len(ep))
            filenames.append(file)

    if len(X) == 0:
        raise ValueError(f"No CSV files found in {folder}")

    # レートが混ざると窓長が変わり特徴量の次元も揃わない。どのファイルが
    # どちらなのかまで出さないと直せないので列挙して止める。
    if len(rates) > 1:
        detail = "; ".join(
            f"{fs}Hz: {len(fl)}本（例 {fl[0]}）" for fs, fl in sorted(rates.items()))
        raise ValueError(
            f"{folder} にサンプリングレートの違う記録が混在しています（{detail}）。"
            f"アンプごとにフォルダを分けてください。")

    return (np.vstack(X), np.concatenate(tones), np.array(attended),
            np.array(trial_idx), filenames)


def balanced_indices(y, groups=None, seed=0, n_keep=None):
    """y の最小クラス数に各クラスを合わせる（多い側をダウンサンプル）インデックスを返す

    groups を渡すと、間引く側を groups（注意音ラベル）ごとに均等に取る。
    non-target は「他の注意音のファイル」由来なので、これで注意音の偏りも同時に消える。
    n_keep を渡すと、各クラスをその枚数まで揃える（分類器間で枚数を揃えるために使う）。
    """
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_keep = int(counts.min()) if n_keep is None else min(int(n_keep), int(counts.min()))

    out = []
    for c in classes:
        ci = np.where(y == c)[0]
        if len(ci) <= n_keep:
            out.append(ci)
            continue
        if groups is None:
            out.append(rng.choice(ci, n_keep, replace=False))
            continue
        # groups ごとに均等割り（端数は先頭グループから配る）
        g = groups[ci]
        gvals = np.unique(g)
        base, rem = divmod(n_keep, len(gvals))
        sel = []
        for j, gv in enumerate(gvals):
            gi = ci[g == gv]
            k = min(base + (1 if j < rem else 0), len(gi))
            sel.append(rng.choice(gi, k, replace=False))
        sel = np.concatenate(sel)
        if len(sel) < n_keep:  # あるグループが不足した分を残りから補う
            rest = np.setdiff1d(ci, sel)
            sel = np.concatenate([sel, rng.choice(rest, n_keep - len(sel), replace=False)])
        out.append(sel)

    return np.sort(np.concatenate(out))


def fit_svm_clfs(X, tones, attended, label_candidates, params_=None, seed=0, trial_idx=None):
    """提示音ごとに target / non-target の二値 LinearSVC を学習する（エポック平均なし）

    均衡化は 3 段階（いずれも既定 True）。
      balance         : 各分類器の target / non-target を最小クラス数に揃える
      balance_global  : さらに全分類器を通しての最小枚数に揃える
                        （target / non-target の枚数が音を通しても完全に同数になる）
      balance_trials  : 注意ラベルごとの「試行数」も最小数に揃える（trial_idx が必要）
                        エポック枚数を揃えても、元になる試行数は 13/12/16 のように
                        不揃いのままなので、試行の多様性まで揃えるのがこれ。
    """
    params_ = params if params_ is None else params_
    balance = params_.get('balance', True)
    balance_global = params_.get('balance_global', True)
    balance_trials = params_.get('balance_trials', True)

    # 注意ラベルごとの試行数を最小数に揃える（エポックを間引く前に行う）
    if balance_trials and trial_idx is not None:
        rng = np.random.RandomState(seed)
        trials = np.array(sorted(set(trial_idx.tolist())))
        tlab = np.array([attended[trial_idx == t][0] for t in trials])
        counts = [int((tlab == c).sum()) for c in label_candidates]
        if counts and min(counts) > 0 and len(set(counts)) > 1:
            n_tr = min(counts)
            sel = np.concatenate([rng.choice(trials[tlab == c], n_tr, replace=False)
                                  for c in label_candidates])
            keep_ep = np.isin(trial_idx, np.sort(sel))
            X, tones, attended = X[keep_ep], tones[keep_ep], attended[keep_ep]
            trial_idx = trial_idx[keep_ep]

    # 学習可能な音と、その音での min(target, non-target) を先に数える
    usable = {}
    for t in label_candidates:
        mask = tones == t
        yt = (attended[mask] == t).astype(int)
        if len(np.unique(yt)) < 2:
            # その音の target 例か non-target 例が片方しか無い場合は分類器を作れない
            continue
        usable[t] = min(int(yt.sum()), int((1 - yt).sum()))

    # 全分類器で共通に使う枚数（各クラスあたり）
    n_keep = min(usable.values()) if (balance and balance_global and usable) else None

    clfs = {}
    for t in usable:
        mask = tones == t
        Xt_all = X[mask]                       # 校正用（間引く前の全エポック）
        yt_all = (attended[mask] == t).astype(int)
        gt = attended[mask]
        tr_all = trial_idx[mask] if trial_idx is not None else None

        if balance:
            keep = balanced_indices(yt_all, gt, seed=seed, n_keep=n_keep)
            Xt, yt = Xt_all[keep], yt_all[keep]
        else:
            Xt, yt = Xt_all, yt_all

        clfs[t] = fit_binary_clf(Xt, yt, params_,
                                 X_cal=Xt_all, y_cal=yt_all, trial_cal=tr_all)
    return clfs


class TrialCalibratedSVC:
    """エポック単位で学習した LinearSVC を「試行平均スコア」でシグモイド校正したもの

    学習は単一エポック・予測は 1 試行 10 エポックの平均、という粒度差があるため、
    校正も試行平均スコアで行う必要がある。単一エポックのスコア分布でシグモイドを
    当てはめると、平均済みのテスト入力は分散が縮んで確率が 0.5 付近に潰れる
    （実測: 123456789 で 0.473 → 0.288 まで落ちる）。
    predict_proba は「その音が注意音である確率」を返す。
    """

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def decision_function(self, X):
        return self.base.decision_function(X)

    def predict_proba(self, X):
        d = np.asarray(self.base.decision_function(X)).reshape(-1, 1)
        return self.calibrator.predict_proba(d)


def fit_binary_clf(X, y, params_=None, X_cal=None, y_cal=None, trial_cal=None):
    """target / non-target の二値分類器を学習して返す

    argmax で音を選ぶには 3 分類器のスコアが同じ尺度でなければならないので、
    predict_proba（その音が注意音である尤度）を返せる形にする。
      prob_method="platt_trial" → 既定。LinearSVC を試行平均スコアでシグモイド校正
      prob_method="platt"       → LinearSVC を CalibratedClassifierCV で確率化
      prob_method="logreg"      → LogisticRegression（確率をそのまま出す）
    """
    params_ = params if params_ is None else params_
    method = params_.get('prob_method', "platt_trial")
    C = params_["C"]

    if method == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=C, max_iter=2000).fit(X, y)

    if method == "platt":
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(penalty="l2", dual=False, C=C, random_state=0)
        return CalibratedClassifierCV(base, cv=params_.get('platt_cv', 5)).fit(X, y)

    if method == "platt_trial":
        from sklearn.linear_model import LogisticRegression
        base = LinearSVC(penalty="l2", dual=False, C=C, random_state=0).fit(X, y)

        if trial_cal is None:
            # 試行情報が無いときは通常の Platt にフォールバック
            from sklearn.calibration import CalibratedClassifierCV
            b2 = LinearSVC(penalty="l2", dual=False, C=C, random_state=0)
            return CalibratedClassifierCV(b2, cv=params_.get('platt_cv', 5)).fit(X, y)

        # 校正用データ: 学習試行ごとに、その音のエポックを平均したスコア
        # （試行平均を一括で作ってから decision_function を 1 回だけ呼ぶ）
        uniq = np.unique(trial_cal)
        means = np.empty((len(uniq), X_cal.shape[1]), dtype=X_cal.dtype)
        lab_cal = np.empty(len(uniq), dtype=int)
        for i, tt in enumerate(uniq):
            m = trial_cal == tt
            means[i] = X_cal[m].mean(axis=0)
            lab_cal[i] = int(y_cal[m][0])
        d_cal = np.asarray(base.decision_function(means)).reshape(-1, 1)

        if len(np.unique(lab_cal)) < 2:
            from sklearn.calibration import CalibratedClassifierCV
            b2 = LinearSVC(penalty="l2", dual=False, C=C, random_state=0)
            return CalibratedClassifierCV(b2, cv=params_.get('platt_cv', 5)).fit(X, y)

        # C を大きくして実質正則化なしの 1 次元ロジスティック（= Platt scaling）
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(d_cal, lab_cal)
        return TrialCalibratedSVC(base, cal)

    raise ValueError(f"Unknown prob_method: {method}")


def score_trial(epochs, tones, clfs):
    """1試行のエポックを音ごとに平均して target 尤度を出し、argmax を返す

    平均するのは予測時だけ（学習時はエポックを平均しない）。
    スコアは predict_proba による「その音が注意音である確率」で、
    3 分類器で同じ 0〜1 スケールなので比較できる。
    """
    scores = {}
    for t, clf in clfs.items():
        mask = tones == t
        if not np.any(mask):
            continue
        mean_epoch = epochs[mask].mean(axis=0, keepdims=True)
        scores[t] = float(clf.predict_proba(mean_epoch)[0, 1])
    if not scores:
        return -2
    return max(scores, key=scores.get)


def cv_by_folds(X, tones, attended, trial_idx, label_candidates, folds,
                params_=None, verbose=False, seed=0):
    """任意の試行分割で試行単位 CV を回す（one-trial-out / k-fold / 日ごとの共通実装）

    folds : 各フォールドのテスト試行番号（配列）のリスト。学習側はその補集合。
            [[t] for t in trials] を渡せば one_trial_out_svm と同じになる。
    戻り値: (trues, preds)。学習側で分類器が立たなかったフォールドの試行は
            score_trial が -2 を返すので、除外せず不正解として数える。
    """
    trues, preds = [], []
    for te_trials in tqdm(folds, disable=not verbose):
        te = np.isin(trial_idx, te_trials)
        tr = ~te
        clfs = fit_svm_clfs(X[tr], tones[tr], attended[tr], label_candidates, params_,
                            seed=seed, trial_idx=trial_idx[tr])
        for t in np.unique(np.asarray(te_trials)):
            mask = trial_idx == t
            if not np.any(mask):
                continue
            preds.append(score_trial(X[mask], tones[mask], clfs))
            trues.append(int(attended[mask][0]))
    return np.array(trues), np.array(preds)


def one_trial_out_svm(X, tones, attended, trial_idx, label_candidates, params_=None,
                      verbose=True, seed=0):
    """試行（ファイル）単位の LOOCV。legacy の oneTrialOut と同じ分割単位"""
    folds = [np.array([t]) for t in sorted(set(trial_idx.tolist()))]
    return cv_by_folds(X, tones, attended, trial_idx, label_candidates, folds,
                       params_=params_, verbose=verbose, seed=seed)


def loocv_accuracy(X, tones, attended, trial_idx, label_candidates, params_=None,
                   n_seed=3, verbose=False):
    """ダウンサンプルの乱数を変えて one-trial-out を n_seed 回まわす

    クラス数を最小数に揃える際のダウンサンプルは乱数依存で、精度が実測 ±0.07 揺れる。
    ハイパーパラメータ探索でこの揺れを拾わないよう、複数 seed の平均で評価する。
    戻り値: (mean, sd, 混同行列の合計, seed 毎の精度リスト)
    """
    accs, cm_sum = [], None
    for s in range(n_seed):
        trues, preds = one_trial_out_svm(X, tones, attended, trial_idx, label_candidates,
                                         params_, verbose=verbose, seed=s)
        accs.append(float(np.mean(trues == preds)))
        cm = confusion_matrix(trues, preds, labels=label_candidates)
        cm_sum = cm if cm_sum is None else cm_sum + cm
    return float(np.mean(accs)), float(np.std(accs)), cm_sum, accs


# 未知のデータを分類する関数（svm バックエンド）
def predict_svm(file_path):
    bundle = joblib.load(model_path)
    if not (isinstance(bundle, dict) and "clfs" in bundle):
        raise ValueError(
            f"{model_path} が旧形式（単一 SVM）です。train_svm() で再学習してください。"
        )

    epochs, tones = extract_epochs(file_path, bundle.get("params", params))
    trained_fs = bundle.get("sampling_rate")
    if trained_fs and int(trained_fs) != int(original_sampling_rate):
        print(f"[model] 警告: モデルは {trained_fs} Hz で学習されていますが、"
              f"この記録は {original_sampling_rate} Hz です。予測は信頼できません。")
    if len(epochs) == 0:
        return np.array([-2])
    return np.array([score_trial(epochs, tones, bundle["clfs"])])


# ==== legacy バックエンド ====

def build_legacy_model(label_candidates=None):
    """legacy_params の内容を反映した model_legacy.Model を生成する"""
    from model_legacy import Model

    myModel = Model()
    myModel.data_name = data_folder
    myModel.kind = legacy_params['kind']
    myModel.LASSO_param = legacy_params['LASSO_param']
    myModel.decimation_value = legacy_params['decimation_value']
    myModel.upper_frequency = legacy_params['upper_frequency']
    myModel.start_eeg = legacy_params['start_eeg']
    myModel.end_eeg = legacy_params['end_eeg']
    myModel.testSequence = legacy_params['testSequence']
    myModel.useChannel = list(legacy_params['useChannel'])
    if label_candidates is not None:
        myModel.label_candidates = list(label_candidates)
    return myModel


def load_legacy_data(folder=None, prefix=None):
    """フォルダ内の CSV を読み込み、(データ, ターゲットラベル, ファイル名) を返す"""
    folder = data_folder if folder is None else folder
    prefix = file_prefix if prefix is None else prefix

    datas = []
    targets = []
    filenames = []
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".csv"):
            continue
        if not file.startswith(prefix):
            continue
        if file.endswith("impedance.csv"):
            continue
        datas.append(pd.read_csv(os.path.join(folder, file)))
        targets.append(extract_label(file))
        filenames.append(file)

    return datas, targets, filenames


def train_legacy():
    """model_legacy.Model で学習し、clfs.pkl / model.pkl を保存する"""
    datas, targets, filenames = load_legacy_data()
    print("Data Size:", len(datas))
    if len(datas) == 0:
        raise ValueError(f"No CSV files found in {data_folder}")

    label_candidates = legacy_params['label_candidates']
    if label_candidates is None:
        label_candidates = sorted(set(targets))
        print("label_candidates (auto):", label_candidates)
    elif set(label_candidates) != set(targets):
        raise ValueError(
            f"label_candidates {sorted(set(label_candidates))} と targets {sorted(set(targets))} が一致しません"
        )

    myModel = build_legacy_model(label_candidates)

    training_data, label, targets, trials, runs, noise_data = myModel.filtering(datas, targets)
    dataPD = myModel.epoching(training_data, label, targets, trials, runs, noise_data)

    clfs = myModel.train(
        dataPD,
        evaluateTrial=legacy_params['evaluateTrial'],
        graph=legacy_params['graph'],
        acc=legacy_params['acc'],
    )

    save_pickle(legacy_clfs_path, clfs)
    save_pickle(legacy_model_path, myModel)
    print(f"Model saved to {legacy_clfs_path} / {legacy_model_path}")

    return myModel, clfs


# 学習済み legacy モデルのキャッシュ（(mtime, mtime) -> (myModel, clfs)）
_legacy_cache = {}


def load_legacy_trained():
    """保存済みの model.pkl / clfs.pkl をロードする（mtime が変わったら読み直す）"""
    if not (os.path.exists(legacy_clfs_path) and os.path.exists(legacy_model_path)):
        return None, None

    key = (os.path.getmtime(legacy_clfs_path), os.path.getmtime(legacy_model_path))
    if key not in _legacy_cache:
        with open(legacy_clfs_path, "rb") as f:
            clfs = pickle.load(f)
        with open(legacy_model_path, "rb") as f:
            myModel = pickle.load(f)
        _legacy_cache.clear()
        _legacy_cache[key] = (myModel, clfs)

    return _legacy_cache[key]


def predict_legacy(file_path):
    """legacy バックエンドで 1 ファイルを分類する

    戻り値は svm バックエンドと同じく配列（要素 1 個）。
    -1: clfs.pkl / model.pkl が存在しない、-2: ノイズ除去でデータが 0 件
    """
    myModel, clfs = load_legacy_trained()
    if myModel is None:
        return np.array([-1])

    data = pd.read_csv(file_path)
    data = data.drop("Seq", axis=1, errors="ignore")

    training_data, label, targets, trials, runs, noise_data = myModel.filtering([data])
    dataPD = myModel.epoching(training_data, label, targets, trials, runs, noise_data)
    result = myModel.predict(dataPD, clfs)
    return np.array([result])


def predict(file_path, backend=None):
    """BACKEND（または引数 backend）に応じて予測を実行する"""
    backend = BACKEND if backend is None else backend
    if backend == "legacy":
        return predict_legacy(file_path)
    elif backend == "svm":
        return predict_svm(file_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ==== ハイパーパラメータのグリッド探索 ====
# 探索は被験者（データフォルダ）ごとに行い、結果はそのフォルダ内に保存する。
def default_tuning_path(folder=None):
    """<data_folder>/tuning_<フォルダ名>.pkl を返す（被験者ごとに1ファイル）"""
    folder = data_folder if folder is None else folder
    return os.path.join(folder, f"tuning_{os.path.basename(folder)}.pkl")


tuning_path = None  # None なら default_tuning_path() を使う

# フィルタは (use_filter, low_cutoff) をペアで動かす
tune_filters = [
    {'use_filter': "iir",      'low_cutoff': 0.5},   # 0.5Hz IIR ハイパス（filter_fixed 相当）
    {'use_filter': "filtfilt", 'low_cutoff': 0.1},   # 旧 firwin（ハイパス不作動、current 相当）
]

# baseline_length: 刺激前の平均区間（サンプル）。20ms / 200ms / 800ms
# downsampling_rate: 窓 250 サンプルを何点に落とすか（= 特徴量の時間点数。dim = rate × 8ch）
tune_grid = {
    'baseline_length': [5, 50, 200],
    'downsampling_rate': [5, 10, 25],
    'C': [0.01, 0.1, 1.0, 10.0],
}


def tune_svm(folder=None, grid=None, filters=None, base_params=None,
             save=True, out_path=None, verbose=True, n_seed=3):
    """one-trial-out でハイパーパラメータをグリッド探索し、結果を pickle に保存する

    データ構築（フィルタ・ベースライン・ダウンサンプル）が同じ組み合わせは 1 回だけ作り、
    その上で C を回す。戻り値と pickle には全組み合わせの結果を残す。

    ※ここで得られる best の精度は「探索した中の最大値」なので選択バイアスで楽観側に偏る。
      素の汎化性能として引用しないこと（nested CV か新規データで確認する）。
    """
    global data_folder
    folder = data_folder if folder is None else folder
    grid = tune_grid if grid is None else grid
    filters = tune_filters if filters is None else filters
    base = dict(params if base_params is None else base_params)
    out_path = (tuning_path or default_tuning_path(folder)) if out_path is None else out_path

    prev_folder = data_folder
    data_folder = folder  # load_filtered_eeg のキャッシュ名に使われる
    try:
        results = []
        combos = [(f, bl, ds) for f in filters
                  for bl in grid['baseline_length']
                  for ds in grid['downsampling_rate']]
        for f_spec, bl, ds in tqdm(combos, disable=not verbose):
            p = dict(base)
            p.update(f_spec)
            p['baseline_length'] = bl
            p['downsampling_rate'] = ds

            X, tones, attended, trial_idx, filenames = build_epoch_dataset(folder, params_=p)
            label_candidates = sorted(set(attended.tolist()))
            chance = 1.0 / len(label_candidates)

            for C in grid['C']:
                p_c = dict(p)
                p_c['C'] = C
                acc, sd, cm, per_seed = loocv_accuracy(X, tones, attended, trial_idx,
                                                       label_candidates, p_c, n_seed=n_seed)
                results.append({
                    'use_filter': f_spec['use_filter'],
                    'low_cutoff': f_spec['low_cutoff'],
                    'baseline_length': bl,
                    'downsampling_rate': ds,
                    'C': C,
                    'dim': int(X.shape[1]),
                    'acc': acc,
                    'acc_sd': sd,
                    'acc_per_seed': per_seed,
                    'confusion': cm,
                    'params': p_c,
                })
    finally:
        data_folder = prev_folder

    results.sort(key=lambda r: -r['acc'])
    best = results[0]

    out = {
        'folder': folder,
        'n_trial': len(filenames),
        'label_candidates': label_candidates,
        'chance': chance,
        'grid': grid,
        'filters': filters,
        'results': results,
        'best': best,
        'best_params': best['params'],
    }

    out['n_seed'] = n_seed

    if verbose:
        print(f"\n=== tuning: {os.path.basename(folder)} "
              f"(n={out['n_trial']}, chance={chance:.3f}, {len(results)} combos, "
              f"seed平均 {n_seed}回) ===")
        print(f"{'filter':>9} {'lc':>4} {'bl':>4} {'ds':>3} {'dim':>4} {'C':>6}  {'acc':>6} {'sd':>5}")
        for r in results[:10]:
            print(f"{r['use_filter']:>9} {r['low_cutoff']:>4} {r['baseline_length']:>4} "
                  f"{r['downsampling_rate']:>3} {r['dim']:>4} {r['C']:>6}  "
                  f"{r['acc']:>6.3f} {r['acc_sd']:>5.3f}")
        print(f"  ... worst = {results[-1]['acc']:.3f}, "
              f"median = {np.median([r['acc'] for r in results]):.3f}")

    if save:
        save_pickle(out_path, out)
        if verbose:
            print(f"Tuning result saved to {out_path}")

    return out


def _trial_cv_accuracy(X, tones, attended, trial_idx, label_candidates, params_,
                       trials, k=5, seed=0):
    """指定した試行集合を k 分割して試行単位 CV の精度を返す（tune の内側ループ用）"""
    all_trials = np.array(sorted(trials))
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(all_trials)), k)

    ok = tot = 0
    for f in folds:
        te_trials = all_trials[f]
        m_te = np.isin(trial_idx, te_trials)
        m_tr = np.isin(trial_idx, np.setdiff1d(all_trials, te_trials))
        clfs = fit_svm_clfs(X[m_tr], tones[m_tr], attended[m_tr], label_candidates,
                            params_, seed=seed, trial_idx=trial_idx[m_tr])
        if not clfs:
            continue
        for t in te_trials:
            m = trial_idx == t
            ok += int(score_trial(X[m], tones[m], clfs) == int(attended[m][0]))
            tot += 1
    return ok / tot if tot else 0.0


def nested_cv_svm(folder=None, grid=None, filters=None, base_params=None,
                  inner_k=5, seed=0, verbose=True):
    """外側 one-trial-out・内側 k-fold でハイパーパラメータを選ぶ nested CV

    tune_svm の best 精度は「72通りの最大値」なので選択バイアスで楽観側に偏る。
    こちらはハイパーパラメータの選択を各外側フォールドの学習データ内だけで行うので、
    「チューニングを含めた手続き全体」の汎化性能の推定値になる。
    ※試行数 × 組み合わせ数だけ計算するので、n の大きいフォルダでは重い。
    """
    global data_folder
    folder = data_folder if folder is None else folder
    grid = tune_grid if grid is None else grid
    filters = tune_filters if filters is None else filters
    base = dict(params if base_params is None else base_params)

    prev_folder = data_folder
    data_folder = folder
    try:
        # データ構築が必要な組み合わせ（フィルタ × ベースライン × ダウンサンプル）を先に全部作る
        datasets = []
        build_combos = [(f, bl, ds) for f in filters
                        for bl in grid['baseline_length']
                        for ds in grid['downsampling_rate']]
        for f_spec, bl, ds in build_combos:
            p = dict(base)
            p.update(f_spec)
            p['baseline_length'] = bl
            p['downsampling_rate'] = ds
            datasets.append((p, build_epoch_dataset(folder, params_=p)))

        _, (_, _, attended0, trial_idx0, filenames) = datasets[0]
        label_candidates = sorted(set(attended0.tolist()))
        trials = np.array(sorted(set(trial_idx0.tolist())))

        trues, preds, chosen = [], [], []
        for held in tqdm(trials, disable=not verbose):
            inner_trials = np.setdiff1d(trials, [held])

            best_acc, best = -1.0, None
            for p, (X, tones, attended, trial_idx, _) in datasets:
                for C in grid['C']:
                    p_c = dict(p)
                    p_c['C'] = C
                    a = _trial_cv_accuracy(X, tones, attended, trial_idx, label_candidates,
                                           p_c, inner_trials, k=inner_k, seed=seed)
                    if a > best_acc:
                        best_acc, best = a, (p_c, X, tones, attended, trial_idx)

            p_c, X, tones, attended, trial_idx = best
            m_tr = trial_idx != held
            m_te = trial_idx == held
            clfs = fit_svm_clfs(X[m_tr], tones[m_tr], attended[m_tr], label_candidates,
                                p_c, seed=seed, trial_idx=trial_idx[m_tr])
            preds.append(score_trial(X[m_te], tones[m_te], clfs))
            trues.append(int(attended[m_te][0]))
            chosen.append({k: p_c[k] for k in
                           ('use_filter', 'low_cutoff', 'baseline_length',
                            'downsampling_rate', 'C')})
    finally:
        data_folder = prev_folder

    trues, preds = np.array(trues), np.array(preds)
    acc = float(np.mean(trues == preds))
    chance = 1.0 / len(label_candidates)

    out = {
        'folder': folder,
        'n_trial': len(trials),
        'chance': chance,
        'nested_acc': acc,
        'confusion': confusion_matrix(trues, preds, labels=label_candidates),
        'chosen_per_fold': chosen,
        'inner_k': inner_k,
        'seed': seed,
    }

    if verbose:
        print(f"\n=== nested CV: {os.path.basename(folder)} "
              f"(外側 one-trial-out n={len(trials)}, 内側 {inner_k}-fold) ===")
        print(f"nested accuracy = {acc:.4f}  (chance = {chance:.3f})")
        print(out['confusion'])
        picked = Counter(tuple(sorted(c.items())) for c in chosen)
        print("選ばれたハイパーパラメータの頻度 (上位5):")
        for combo, cnt in picked.most_common(5):
            print(f"  {cnt:>3}/{len(chosen)}  {dict(combo)}")

    return out


def load_tuned_params(out_path=None, folder=None):
    """tune_svm が保存した pickle から best_params を読む（フォルダ不一致なら警告）"""
    folder = data_folder if folder is None else folder
    out_path = (tuning_path or default_tuning_path(folder)) if out_path is None else out_path
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"{out_path} がありません。先に tune_svm() を実行してください。")
    with open(out_path, "rb") as fp:
        out = pickle.load(fp)
    if os.path.basename(out['folder']) != os.path.basename(folder):
        print(f"warning: {out_path} は {out['folder']} で探索した結果です "
              f"(現在の data_folder = {folder})")
    return out['best_params'], out


def train_svm(evaluate_trial=True, params_=None, use_tuned=False):
    """エポック単位 target/non-target の音別 LinearSVC を学習し svm_model.pkl に保存する

    evaluate_trial=True なら legacy と同じ試行単位 LOOCV（one-trial-out）で精度を出す。
    use_tuned=True なら svm_tuning.pkl の best_params を使う。
    """
    tuned = None
    if use_tuned:
        params_, tuned = load_tuned_params(folder=data_folder)
    p = params if params_ is None else params_

    X, tones, attended, trial_idx, filenames = build_epoch_dataset(params_=p)

    # 注意音の候補（ファイル名末尾の数字）。音4のように注意対象にならない音は分類器を作らない
    label_candidates = sorted(set(attended.tolist()))

    n_trial = len(filenames)
    print(f"Preset: {SVM_PRESET if params_ is None else ('tuned' if use_tuned else 'custom')}  "
          f"(filter={p.get('use_filter')}, low_cutoff={p['low_cutoff']}Hz, "
          f"baseline={p['baseline_length']}, downsample={p['downsampling_rate']}, C={p['C']}, "
          f"balance={p.get('balance', True)})")
    print("X shape:", X.shape, " (epoch単位)")
    print("Trials:", n_trial, " epochs/trial:", np.bincount(trial_idx).tolist()[:5], "...")
    print("label_candidates:", label_candidates)
    print("tone dist:", dict(sorted(Counter(tones.tolist()).items())))
    per_tone_min = {}
    for t in label_candidates:
        m = tones == t
        y = (attended[m] == t).astype(int)
        per_tone_min[t] = min(int(y.sum()), int((1 - y).sum()))
    n_keep = (min(per_tone_min.values())
              if p.get('balance', True) and p.get('balance_global', True) else None)
    for t in label_candidates:
        m = tones == t
        y = (attended[m] == t).astype(int)
        if not p.get('balance', True):
            desc = f"学習に使う数={int(m.sum())} (均衡化なし)"
        else:
            k = per_tone_min[t] if n_keep is None else n_keep
            desc = f"学習に使う数={2*k} (target={k} / non-target={k})"
        print(f"  clf[tone={t}]: n={m.sum()}  target={int(y.sum())}  "
              f"non-target={int((1-y).sum())}  -> {desc}")

    acc = cm = acc_sd = None
    if evaluate_trial:
        n_seed = 1 if not p.get('balance', True) else 3
        acc, acc_sd, cm, per_seed = loocv_accuracy(X, tones, attended, trial_idx,
                                                   label_candidates, p, n_seed=n_seed,
                                                   verbose=(n_seed == 1))
        print(f"\naccuracy for LOOCV (one-trial-out): {acc:.4f} ± {acc_sd:.4f}  "
              f"(chance = {1.0/len(label_candidates):.3f}, n = {n_trial}, "
              f"seed平均 {n_seed}回: {[round(v, 3) for v in per_seed]})")
        print(f"Confusion matrix (seed {n_seed}回の合計):")
        print(cm)

        # 誤分類ファイルの一覧は seed=0 のもの
        trues, preds = one_trial_out_svm(X, tones, attended, trial_idx, label_candidates,
                                         p, verbose=False, seed=0)
        wrong = [filenames[i] for i in range(n_trial) if trues[i] != preds[i]]
        print(f"Misclassified trials (seed=0): {len(wrong)}/{n_trial}")
        for f in wrong[:10]:
            print("  ", f)

    # 全試行で最終学習
    clfs = fit_svm_clfs(X, tones, attended, label_candidates, p, trial_idx=trial_idx)

    bundle = {
        "clfs": clfs,
        "label_candidates": label_candidates,
        "params": p,
        "preset": ("tuned" if use_tuned else SVM_PRESET),
        "data_folder": data_folder,
        "loocv_acc": acc,
        "loocv_acc_sd": acc_sd,
        "loocv_confusion": cm,
        "tuning": (tuned['best'] if tuned else None),
        # ---- 以下は bci_server の一覧表示と再現性のための付帯情報 ----
        # sampling_rate は「この pkl がどのアンプのデータで学習されたか」を示す。
        # 250Hz学習のモデルに1000Hzの記録を食わせても例外は出ないが結果は無意味なので、
        # predict 側でここと突き合わせて警告する。
        "sampling_rate": original_sampling_rate,
        "use_sequence": use_sequence,
        "n_files": n_trial,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv_accuracy": acc,          # loocv_acc の別名（UIはこちらを見る）
        "labels": label_candidates,  # label_candidates の別名
    }
    atomic_save(model_path, lambda tmp: joblib.dump(bundle, tmp))
    print(f"\nModel saved to {model_path}  (clfs for tones {sorted(clfs.keys())})")

    return clfs


def evaluate_folder(backend=None, folder=None):
    """学習に使ったフォルダを predict() で流し直して混同行列と accuracy を表示する"""
    folder = data_folder if folder is None else folder

    #predict_Test
    filenames = os.listdir(folder)
    predictions = []
    true_labels = []
    for filename in filenames:
        if filename.endswith(".csv"):
            if filename.endswith("impedance.csv"):
                continue
            file_path = os.path.join(folder, filename)
            predictions.append(predict(file_path, backend=backend))

            # 正解ラベルを取得
            label = extract_label(filename)
            true_labels.append(label)

    # 混同行列とaccuracyを表示
    # Flatten predictions to a 1D array of predicted labels
    flat_predictions = [pred[0] if isinstance(pred, np.ndarray) else pred for pred in predictions]
    cm = confusion_matrix(true_labels, flat_predictions)
    acc = np.mean(np.array(true_labels) == np.array(flat_predictions))
    print(f"Confusion matrix:\n{cm}")
    print(f"Accuracy: {acc:.3f}")
    return cm, acc


# ==== サーバーから呼ぶ学習エントリポイント ====
# bci_server.py（Pi / Windows）は被験者ごとに data/<被験者>/ を学習して
# models/<被験者>/svm_model.pkl に書く。このモジュールは data_folder / model_path /
# use_sequence / params をモジュール変数で持っているので、呼び出しの間だけ差し替える。
def train(data_folder_override=None, model_path_override=None,
          use_sequence_override=None, params_=None, evaluate_trial=True):
    """指定フォルダの CSV で svm バックエンドを学習し、要約を dict で返す。

    data_folder_override  … 学習に使うフォルダ（被験者のデータフォルダ）
    model_path_override   … 保存先の svm_model.pkl
    use_sequence_override … 1試行あたり何セット分の刺激を使うか（UIの選択値）
    params_               … 前処理・分類のパラメータ（None ならモジュールの params）

    戻り値: {"cv_accuracy", "cv_accuracy_sd", "n_files", "use_sequence",
             "sampling_rate", "labels", "model_path", "data_folder"}
    """
    global data_folder, model_path, use_sequence

    prev = (data_folder, model_path, use_sequence)
    if data_folder_override is not None:
        data_folder = data_folder_override
    if model_path_override is not None:
        model_path = model_path_override
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    if use_sequence_override is not None:
        use_sequence = int(use_sequence_override)

    try:
        train_svm(evaluate_trial=evaluate_trial, params_=params_)
        bundle = joblib.load(model_path)
        return {
            "cv_accuracy": bundle.get("cv_accuracy"),
            "cv_accuracy_sd": bundle.get("loocv_acc_sd"),
            "n_files": bundle.get("n_files"),
            "use_sequence": bundle.get("use_sequence"),
            "sampling_rate": bundle.get("sampling_rate"),
            "labels": bundle.get("label_candidates"),
            "model_path": model_path,
            "data_folder": data_folder,
        }
    finally:
        data_folder, model_path, use_sequence = prev


if __name__ == "__main__":
    print(f"Backend: {BACKEND}")

    # TUNE=1 でハイパーパラメータ探索 → その best_params で学習
    #   例: TUNE=1 MODEL_BACKEND=svm python model.py
    do_tune = os.environ.get("TUNE", "0") == "1"

    if BACKEND == "legacy":
        train_legacy()
    elif BACKEND == "svm":
        if do_tune:
            tune_svm()
            train_svm(use_tuned=True)
        else:
            train_svm()
    else:
        raise ValueError(f"Unknown backend: {BACKEND}")

    evaluate_folder(BACKEND)