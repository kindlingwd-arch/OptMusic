"""调性（调号）识别与和弦识别。

调性：Krumhansl-Kessler 调性轮廓 × 音级分布 的相关性，24 个大小调取最大。
和弦：以“拍”为单位统计音级能量 → 模板打分（含低音、调内先验）→ 动态规划平滑 → 合并。
详见 docs/算法原理.md
"""
from __future__ import annotations

import math

import numpy as np

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Krumhansl & Kessler (1982) 探测音实验得到的调性轮廓
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10, 11]  # 自然小调 + 和声小调的导音


# ---------------------------------------------------------------------------
# 音级分布（pitch-class profile）
# ---------------------------------------------------------------------------

def pitch_class_histogram(notes: list[dict]) -> np.ndarray:
    """按 时长 × 力度权重 累加每个音级。时长取 sqrt，避免个别长音主导。"""
    h = np.zeros(12)
    for n in notes:
        dur = max(0.0, n["end"] - n["start"])
        w = math.sqrt(dur) * (0.5 + 0.5 * n.get("velocity", 100) / 127)
        h[int(n["pitch"]) % 12] += w
    return h


# ---------------------------------------------------------------------------
# 调性 / 调号
# ---------------------------------------------------------------------------

def key_signature(tonic: int, mode: str) -> int:
    """返回调号：正数 = 升号个数，负数 = 降号个数。"""
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    fifths = (major_tonic * 7) % 12          # 在五度圈上的位置
    return fifths - 12 if fifths > 6 else fifths   # F#(6) 记为 6 个升号，Db 记为 5 个降号


def spell(pc: int, sig: int) -> str:
    return (FLAT if sig < 0 else SHARP)[pc % 12]


def detect_key(hist: np.ndarray) -> dict | None:
    if hist.sum() <= 0:
        return None
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            scores.append((float(r), tonic, mode))
    scores.sort(reverse=True)
    r, tonic, mode = scores[0]
    sig = key_signature(tonic, mode)
    name = spell(tonic, sig) + (" 大调" if mode == "major" else " 小调")
    sig_text = "无升降号" if sig == 0 else (f"{sig} 个升号" if sig > 0 else f"{-sig} 个降号")
    return {
        "tonic": tonic, "mode": mode, "name": name,
        "signature": sig, "signature_text": sig_text,
        "confidence": round(max(0.0, r), 3),
        "margin": round(r - scores[1][0], 3),
        "alternatives": [{"name": spell(t, key_signature(t, m)) + (" 大调" if m == "major" else " 小调"),
                          "r": round(s, 3)} for s, t, m in scores[1:4]],
    }


# ---------------------------------------------------------------------------
# 和弦识别
# ---------------------------------------------------------------------------

# 名称后缀, 音程(相对根音), 每个音的权重, 复杂度惩罚
CHORD_TYPES = [
    ("",     [0, 4, 7],      [1.0, 1.0, 0.8],       0.00),
    ("m",    [0, 3, 7],      [1.0, 1.0, 0.8],       0.00),
    ("dim",  [0, 3, 6],      [1.0, 1.0, 0.9],       0.06),
    ("aug",  [0, 4, 8],      [1.0, 1.0, 0.9],       0.10),
    ("sus2", [0, 2, 7],      [1.0, 0.9, 0.8],       0.08),
    ("sus4", [0, 5, 7],      [1.0, 0.9, 0.8],       0.06),
    ("7",    [0, 4, 7, 10],  [1.0, 1.0, 0.7, 0.9],  0.05),
    ("maj7", [0, 4, 7, 11],  [1.0, 1.0, 0.7, 0.9],  0.06),
    ("m7",   [0, 3, 7, 10],  [1.0, 1.0, 0.7, 0.9],  0.05),
    ("m7b5", [0, 3, 6, 10],  [1.0, 1.0, 0.9, 0.9],  0.09),
    ("dim7", [0, 3, 6, 9],   [1.0, 1.0, 0.9, 0.9],  0.12),
    ("6",    [0, 4, 7, 9],   [1.0, 1.0, 0.7, 0.8],  0.08),
    ("m6",   [0, 3, 7, 9],   [1.0, 1.0, 0.7, 0.8],  0.09),
]
NO_CHORD = -1
OUT_PENALTY = 0.6      # 和弦外音的扣分系数
MISSING_PENALTY = 0.12  # 和弦音完全缺失的扣分
BASS_ROOT_BONUS = 0.15
BASS_IN_CHORD_BONUS = 0.05
DIATONIC_BONUS = 0.08


def _all_chords():
    out = []
    for root in range(12):
        for ti, (suffix, ivs, ws, cx) in enumerate(CHORD_TYPES):
            pcs = [(root + i) % 12 for i in ivs]
            out.append({"root": root, "type": ti, "suffix": suffix, "pcs": pcs,
                        "w": dict(zip(pcs, ws)), "cx": cx})
    return out


CHORDS = _all_chords()


def _diatonic(chord, key) -> bool:
    if not key:
        return False
    scale = MAJOR_SCALE if key["mode"] == "major" else MINOR_SCALE
    allowed = {(key["tonic"] + s) % 12 for s in scale}
    return all(pc in allowed for pc in chord["pcs"])


def _emission(vec: np.ndarray, bass: int | None, key) -> np.ndarray:
    """对一个时间片的 12 维音级向量，给每个候选和弦打分。"""
    total = vec.sum()
    scores = np.empty(len(CHORDS))
    if total <= 0:
        return scores * 0
    v = vec / total
    for k, c in enumerate(CHORDS):
        inside = sum(c["w"][pc] * v[pc] for pc in c["pcs"])
        outside = sum(v[pc] for pc in range(12) if pc not in c["w"])
        missing = sum(1 for pc in c["pcs"] if v[pc] < 0.02)
        s = inside - OUT_PENALTY * outside - MISSING_PENALTY * missing - c["cx"]
        if bass is not None:
            if bass == c["root"]:
                s += BASS_ROOT_BONUS
            elif bass in c["w"]:
                s += BASS_IN_CHORD_BONUS
        if _diatonic(c, key):
            s += DIATONIC_BONUS
        scores[k] = s
    return scores


def detect_chords(notes: list[dict], bpm: float, beats_per_bar: int, origin: float,
                  key: dict | None = None, unit_beats: float = 1.0) -> list[dict]:
    if not notes:
        return []
    beat = 60.0 / bpm
    step = beat * unit_beats
    end_t = max(n["end"] for n in notes)
    k0 = math.floor((min(n["start"] for n in notes) - origin) / step)
    k1 = math.ceil((end_t - origin) / step)
    slices = [(origin + k * step, origin + (k + 1) * step, k) for k in range(k0, k1)]

    # 1) 每个时间片的音级向量 + 低音
    vecs, basses = [], []
    for s, e, _ in slices:
        v = np.zeros(12)
        bass, bass_w = None, 0.0
        lowest = None
        for n in notes:
            ov = min(e, n["end"]) - max(s, n["start"])
            if ov <= 0:
                continue
            w = ov / step * (0.5 + 0.5 * n.get("velocity", 100) / 127)
            v[int(n["pitch"]) % 12] += w
            if lowest is None or n["pitch"] < lowest:
                lowest, bass_w = n["pitch"], w
        if lowest is not None and bass_w > 0.25:
            bass = int(lowest) % 12
        vecs.append(v)
        basses.append(bass)

    # 2) 发射分数
    E = np.array([_emission(v, b, key) for v, b in zip(vecs, basses)])
    empty = np.array([v.sum() <= 1e-6 for v in vecs])

    # 3) 动态规划（Viterbi）平滑：换和弦有代价，强拍上换代价更低
    n_s, n_c = E.shape
    def switch_cost(k):
        pos = (k * unit_beats) % beats_per_bar
        if abs(pos) < 1e-6:
            return 0.10                       # 小节第一拍
        if beats_per_bar % 2 == 0 and abs(pos - beats_per_bar / 2) < 1e-6:
            return 0.20                       # 4/4 的第三拍
        return 0.35
    score = E[0].copy()
    back = np.zeros((n_s, n_c), dtype=int)
    for t in range(1, n_s):
        if empty[t]:
            # 空拍：沿用上一个和弦（延音/休止不改变和声）
            back[t] = np.arange(n_c)
            continue
        best_prev = int(np.argmax(score))
        stay = score
        move = score[best_prev] - switch_cost(slices[t][2])
        choose_stay = stay >= move
        back[t] = np.where(choose_stay, np.arange(n_c), best_prev)
        score = np.where(choose_stay, stay, move) + E[t]
    path = [int(np.argmax(score))]
    for t in range(n_s - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    path.reverse()

    # 4) 合并相同和弦，生成名称（含转位：C/E）
    sig = key["signature"] if key else 0
    out = []
    for t, (s, e, _) in enumerate(slices):
        c = CHORDS[path[t]]
        has_notes = not empty[t]
        if out and out[-1]["_k"] == path[t]:
            out[-1]["end"] = e
            if has_notes:
                out[-1]["_bass"].append(basses[t])
                out[-1]["_score"].append(E[t][path[t]])
            continue
        out.append({"start": s, "end": e, "_k": path[t], "root": c["root"], "suffix": c["suffix"],
                    "_bass": [basses[t]] if has_notes else [], "_score": [E[t][path[t]]] if has_notes else []})

    result = []
    for seg in out:
        c = CHORDS[seg["_k"]]
        if not seg["_score"]:
            name = "N"
            conf = 0.0
        else:
            name = spell(c["root"], sig) + c["suffix"]
            bs = [b for b in seg["_bass"] if b is not None]
            if bs:
                bass = max(set(bs), key=bs.count)
                if bass != c["root"] and bass in c["w"]:
                    name += "/" + spell(bass, sig)
            conf = float(np.clip(np.mean(seg["_score"]), 0, 1))
        result.append({"start": round(max(0.0, seg["start"]), 4), "end": round(seg["end"], 4),
                       "name": name, "root": c["root"], "quality": c["suffix"] or "maj",
                       "confidence": round(conf, 3)})
    return result


def analyze_harmony(notes, bpm, beats_per_bar, origin, audio_chroma: np.ndarray | None = None):
    hist = pitch_class_histogram(notes)
    if audio_chroma is not None and hist.sum() > 0:
        hist = 0.5 * hist / hist.sum() + 0.5 * audio_chroma / (audio_chroma.sum() + 1e-9)
    key = detect_key(hist)
    # 只有旋律（同一时刻最多一个音）时，单拍信息太少，改为半小节一个和弦
    unit = 1.0
    if max_polyphony(notes) <= 1:
        unit = beats_per_bar / 2 if beats_per_bar % 2 == 0 else float(beats_per_bar)
    chords = detect_chords(notes, bpm, beats_per_bar, origin, key, unit_beats=unit)
    return key, chords


def max_polyphony(notes) -> int:
    ev = sorted([(n["start"] + 0.02, 1) for n in notes] + [(n["end"], -1) for n in notes])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best
