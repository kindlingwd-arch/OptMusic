"""音频分析：音高追踪 → 音符切分，BPM / 下拍 / 小节网格估计。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import librosa
import numpy as np
import scipy.ndimage as ndi

SR = 22050
HOP = 256
FRAME = 2048


def load_mono(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# 节奏：BPM、拍点、下拍、小节网格
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    bpm: float
    beats_per_bar: int
    origin: float          # 第 1 小节第 1 拍的时间（秒），<= 0，代表音频之前的弱起
    beat_times: list       # 检测到的原始拍点（秒），用于可视化校验
    confidence: float

    def to_dict(self):
        return {
            "bpm": round(self.bpm, 3),
            "beats_per_bar": self.beats_per_bar,
            "origin": round(self.origin, 5),
            "beat_times": [round(t, 4) for t in self.beat_times],
            "confidence": round(self.confidence, 3),
        }


def estimate_grid(y: np.ndarray, sr: int = SR, beats_per_bar: int | None = None,
                  note_onsets: list | None = None) -> Grid:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                           hop_length=HOP, units="frames")
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 120.0

    if len(beat_times) >= 4:
        # 用线性回归拟合恒定速度，比单个 tempo 估计更准，也能得到相位
        idx = np.arange(len(beat_times))
        period, intercept = np.polyfit(idx, beat_times, 1)
        resid = beat_times - (idx * period + intercept)
        confidence = float(max(0.0, 1.0 - np.std(resid) / (period * 0.25)))
    else:
        period = 60.0 / (tempo or 120.0)
        intercept = float(beat_times[0]) if len(beat_times) else 0.0
        idx = np.arange(len(beat_times))
        confidence = 0.0

    bpm = 60.0 / period

    # 每个拍点的“重音强度”：起音强度 + 低频能量 + 音符起点数
    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME)
    low = S[freqs < 200].sum(axis=0)
    low = low / (low.max() + 1e-9)
    env = onset_env / (onset_env.max() + 1e-9)
    grid_beats = intercept + period * idx
    strength = []
    for t in grid_beats:
        f = int(round(t * sr / HOP))
        lo, hi = max(0, f - 2), min(len(env), f + 3)
        s = env[lo:hi].max() if hi > lo else 0.0
        s += 0.7 * (low[lo:hi].max() if hi > lo else 0.0)
        if note_onsets:
            s += 0.5 * sum(1 for o in note_onsets if abs(o - t) < period * 0.15)
        strength.append(s)
    strength = np.array(strength) if strength else np.zeros(1)

    def phase_scores(bpb):
        return np.array([strength[k::bpb].mean() if len(strength[k::bpb]) else 0
                         for k in range(bpb)])

    if beats_per_bar is None:
        # 3/4 与 4/4 之间粗略判断：比较“最强相位 / 平均”的对比度，偏向 4/4
        c4 = phase_scores(4)
        c3 = phase_scores(3)
        r4 = c4.max() / (c4.mean() + 1e-9)
        r3 = c3.max() / (c3.mean() + 1e-9)
        beats_per_bar = 3 if (len(strength) >= 12 and r3 > r4 * 1.15) else 4

    scores = phase_scores(beats_per_bar)
    k = int(np.argmax(scores))
    first_downbeat = intercept + period * k
    bar_len = period * beats_per_bar
    n = math.ceil(first_downbeat / bar_len - 1e-6)
    origin = first_downbeat - n * bar_len

    return Grid(bpm=bpm, beats_per_bar=beats_per_bar, origin=float(origin),
                beat_times=[float(t) for t in beat_times], confidence=confidence)


# ---------------------------------------------------------------------------
# 单音（旋律）模式：pYIN 音高曲线 + 切分
# ---------------------------------------------------------------------------

def _nanmedian_filter(x: np.ndarray, size: int) -> np.ndarray:
    out = x.copy()
    half = size // 2
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        w = x[max(0, i - half): i + half + 1]
        w = w[~np.isnan(w)]
        out[i] = np.median(w)
    return out


def transcribe_mono(y: np.ndarray, sr: int = SR, fmin: float = 65.0, fmax: float = 1400.0,
                    min_note: float = 0.07, split_semitones: float = 0.75):
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=FRAME, hop_length=HOP, fill_na=np.nan)
    midi = librosa.hz_to_midi(f0)
    voiced = voiced_flag & (voiced_prob > 0.25) & ~np.isnan(midi)
    midi = np.where(voiced, midi, np.nan)
    smooth = _nanmedian_filter(midi, 5)

    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) + 1e-12)
    n = min(len(smooth), len(rms_db))
    smooth, midi, voiced, rms_db, voiced_prob = (smooth[:n], midi[:n], voiced[:n],
                                                 rms_db[:n], voiced_prob[:n])

    onset_frames = set(librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP,
                                                  units="frames", backtrack=True).tolist())
    min_frames = max(2, int(round(min_note * sr / HOP)))
    persist = 6  # 音高变化需持续的帧数（约 70ms）才算换音

    # 1) 有声段（允许 2 帧以内的短暂断开）
    segments = []
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j + 1 < n and (voiced[j + 1] or gap < 2):
            j += 1
            gap = 0 if voiced[j] else gap + 1
        while j > i and not voiced[j]:
            j -= 1
        segments.append((i, j + 1))
        i = j + 1

    # 2) 段内按“音高稳定变化”与“起音点”切分
    raw_notes = []
    for s, e in segments:
        start = s
        vals = []
        k = s
        while k < e:
            v = smooth[k]
            if np.isnan(v):
                k += 1
                continue
            cur = np.median(vals) if vals else v
            split_at = None
            if vals and abs(v - cur) > split_semitones:
                run = smooth[k:k + persist]
                run = run[~np.isnan(run)]
                if len(run) >= persist - 1 and np.all(np.abs(run - cur) > split_semitones):
                    split_at = k
            if split_at is None and k in onset_frames and k - start >= min_frames and e - k >= min_frames:
                split_at = k
            if split_at is not None and split_at - start >= 1:
                raw_notes.append((start, split_at))
                start = split_at
                vals = []
            vals.append(v)
            k += 1
        raw_notes.append((start, e))

    notes = []
    for s, e in raw_notes:
        if e - s < min_frames:
            continue
        seg = midi[s:e]
        seg = seg[~np.isnan(seg)]
        if len(seg) < max(2, min_frames // 2):
            continue
        # 去掉首尾各 15% 的滑音再取中位数
        trim = int(len(seg) * 0.15)
        core = seg[trim: len(seg) - trim] if len(seg) - 2 * trim >= 3 else seg
        med = float(np.median(core))
        pitch = int(round(med))
        loud = float(np.max(rms_db[s:e]))
        velocity = int(np.clip(127 + loud * 2.2, 20, 127))
        notes.append({
            "start": s * HOP / sr,
            "end": e * HOP / sr,
            "pitch": pitch,
            "cents": round((med - pitch) * 100, 1),
            "velocity": velocity,
            "confidence": round(float(np.nanmean(voiced_prob[s:e])), 3),
        })

    # 3) 合并：同音高、无起音点、首尾相接的片段
    merged = []
    for nt in notes:
        if merged:
            p = merged[-1]
            f = int(round(nt["start"] * sr / HOP))
            if (p["pitch"] == nt["pitch"] and nt["start"] - p["end"] < 0.03
                    and not any(abs(f - o) <= 1 for o in onset_frames)):
                p["end"] = nt["end"]
                p["velocity"] = max(p["velocity"], nt["velocity"])
                continue
        merged.append(nt)

    # 4) 起点校正：pYIN 需要若干帧才判定“有声”，起点普遍偏晚。
    #    在 [起点-150ms, 起点+30ms] 内取最早的起音点（onset），且不越过上一个音符的结尾。
    #    取最早而非最近：起点只会偏晚；鼓等打击乐常在同一音头附近产生多个 onset。
    onset_times = np.array(sorted(onset_frames)) * HOP / sr
    prev_end = 0.0
    for nt in merged:
        cand = onset_times[(onset_times >= nt["start"] - 0.15) & (onset_times <= nt["start"] + 0.03)
                           & (onset_times >= prev_end - 0.01)]
        if len(cand):
            nt["start"] = float(cand.min())
        prev_end = nt["end"]

    curve = [None if np.isnan(v) else round(float(v), 2) for v in midi]
    return merged, {"hop_sec": HOP / sr, "midi": curve}


# ---------------------------------------------------------------------------
# 对比：原声片段与 VST 渲染的频谱相似度（检测八度 / 半音错误）
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 调音：整体调音参考 + 单个音符的精确音分（复音时用频谱峰值，而不是模型的 1/3 半音音高图）
# ---------------------------------------------------------------------------

def estimate_tuning(y: np.ndarray, sr: int = SR, seconds: float = 90) -> float:
    """整体调音偏差（音分，相对 A4=440Hz）。例：+8 表示整首歌大约按 A4=442Hz 调音。"""
    H = librosa.effects.harmonic(y[: int(sr * seconds)])
    return float(librosa.estimate_tuning(y=H, sr=sr, resolution=0.01, bins_per_octave=12) * 100)


def refine_cents(y: np.ndarray, sr: int, notes: list[dict], n_fft: int = 8192, max_win: float = 0.6):
    """用音频本身测每个音符的精确音高（音分偏差），就地写入 note["cents"]。

    在音符开头最多 0.6 秒内做高分辨率 FFT（8192 点，约 2.7 Hz/格），在第 1~4 泛音附近 ±50 音分内
    找最强的谱峰，用抛物线插值得到亚格精度的频率，再换算成相对该泛音理论频率的音分偏差。
    低音区基频分辨率不够时，高次泛音会自动胜出（它们的绝对频率高、相对误差小）。"""
    win = np.hanning(n_fft)
    for n in notes:
        s = int(n["start"] * sr)
        e = min(len(y), s + int(min(max_win, n["end"] - n["start"]) * sr))
        seg = y[s:e]
        if len(seg) < 2048:
            continue
        frames = [seg[i:i + n_fft] for i in range(0, max(1, len(seg) - n_fft + 1), n_fft // 4)]
        spec = np.zeros(n_fft // 2 + 1)
        for fr in frames:
            fr = np.pad(fr, (0, n_fft - len(fr)))
            spec += np.abs(np.fft.rfft(fr * win)) ** 2
        logs = np.log(spec + 1e-12)
        f_nom = 440.0 * 2 ** ((n["pitch"] - 69) / 12)
        best = None
        for h in (1, 2, 3, 4):
            fc = f_nom * h
            if fc > sr / 2 - 200:
                break
            lo = int(fc * 2 ** (-50 / 1200) * n_fft / sr)
            hi = int(np.ceil(fc * 2 ** (50 / 1200) * n_fft / sr))
            if hi - lo < 2 or lo < 1:
                continue
            k = lo + int(np.argmax(logs[lo:hi + 1]))
            if not 1 <= k < len(logs) - 1:
                continue
            a, b, c = logs[k - 1], logs[k], logs[k + 1]
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            f = (k + float(np.clip(delta, -0.5, 0.5))) * sr / n_fft
            cents = 1200 * np.log2(f / fc)
            strength = spec[k]
            if abs(cents) <= 50 and (best is None or strength > best[0]):
                best = (strength, cents)
        if best is not None:
            n["cents"] = round(float(best[1]), 1)
    return notes


PROFILE_LO, PROFILE_HI = 24, 120   # C1 ~ C9，每个半音一个频带
PROFILE_NFFT = 8192


def _semitone_matrix(sr: int, n_fft: int) -> np.ndarray:
    """把 FFT 频点映射到半音频带（每个半音 ±50 音分）的求和矩阵。"""
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    midi = np.full_like(freqs, -1.0)
    midi[1:] = 69 + 12 * np.log2(freqs[1:] / 440.0)
    band = np.round(midi).astype(int)
    M = np.zeros((PROFILE_HI - PROFILE_LO, len(freqs)), dtype=np.float32)
    for i, b in enumerate(range(PROFILE_LO, PROFILE_HI)):
        M[i, band == b] = 1.0
    return M


_SEMI_M = None


def cqt_profile(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """半音频谱轮廓（类 CQT，但用 FFT 实现，单次只需几毫秒）。
    分帧加窗 FFT → 功率谱平均 → 按半音频带求和 → 对数压缩 → 去均值归一化。"""
    global _SEMI_M
    if _SEMI_M is None:
        _SEMI_M = _semitone_matrix(sr, PROFILE_NFFT)
    if len(y) < PROFILE_NFFT:
        y = np.pad(y, (0, PROFILE_NFFT - len(y)))
    S = np.abs(librosa.stft(y, n_fft=PROFILE_NFFT, hop_length=PROFILE_NFFT // 4, center=False)) ** 2
    prof = np.log1p(1000 * (_SEMI_M @ S.mean(axis=1)))
    prof = prof - prof.mean()
    return prof / (np.linalg.norm(prof) + 1e-9)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
