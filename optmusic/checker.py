"""VST 逐音校验：判断每个 MIDI 音符的音高是否与原声一致。

做法（“解释掉”法，类似 NMF 频谱分解）：
1. 每个音高 p 用当前 VST 渲染一个参考音，提取“半音频谱”（每个半音一个频带的幅度）作为模板 T[p]；
2. 对一个音符，取原声里它开头最多 1 秒的片段，算出同样的半音频谱 S；
3. 找出这段时间里**同时在响的其他音符**（复音音频必须考虑，否则别的和弦音会干扰判断）；
4. 对候选音高 c ∈ {p, p±1, p±12}：用非负最小二乘求 S ≈ Σ a_k·T[其他音] + a_c·T[c]，记录残差 r_c；
5. 若某个候选 c ≠ p 的残差比 p 明显更小（相对改善 > margin），判为“疑似错音”，建议音高 = c；
   否则若实测音分偏差超出容差，判为“音准偏”；否则“正确”。

单音音频没有“其他音”，第 4 步退化为“哪个候选模板最像原声”。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .analysis import PROFILE_HI, PROFILE_LO, PROFILE_NFFT, _semitone_matrix

_M = {}


def semitone_spectrum(y: np.ndarray, sr: int) -> np.ndarray:
    """半音频谱（线性幅度，C1~B8 共 96 个频带），单位化。"""
    import librosa
    if sr not in _M:
        _M[sr] = _semitone_matrix(sr, PROFILE_NFFT)
    if len(y) < PROFILE_NFFT:
        y = np.pad(y, (0, PROFILE_NFFT - len(y)))
    P = np.abs(librosa.stft(y, n_fft=PROFILE_NFFT, hop_length=PROFILE_NFFT // 4, center=False)) ** 2
    v = np.sqrt(_M[sr] @ P.mean(axis=1))
    return v / (np.linalg.norm(v) + 1e-12)


def candidates(p: int) -> list[int]:
    return [c for c in (p, p - 12, p + 12, p - 1, p + 1) if 21 <= c <= 108]


def needed_pitches(notes: list[dict]) -> set[int]:
    return {c for n in notes for c in candidates(int(n["pitch"]))}


def check_notes(y: np.ndarray, sr: int, notes: list[dict], all_notes: list[dict], templates: dict[int, np.ndarray],
                cents_tolerance: float = 30.0, margin: float = 0.04, gain_ratio: float = 1.2,
                max_win: float = 1.0) -> list[dict]:
    """notes：要校验的音符；all_notes：工程里全部音符（用于找同时发声的其他音）。
    templates：音高 → semitone_spectrum（来自 VST 渲染）。"""
    results = []
    for n in notes:
        p = int(n["pitch"])
        s, e = n["start"], min(n["end"], n["start"] + max_win)
        seg = y[int(s * sr): int(e * sr)]
        if len(seg) < 512 or np.max(np.abs(seg)) < 1e-4:
            results.append({"id": n["id"], "status": "silent", "match": None, "suggest": None})
            continue
        S = semitone_spectrum(seg, sr)
        win = e - s
        others = sorted({int(m["pitch"]) for m in all_notes
                         if m.get("id") != n.get("id") and int(m["pitch"]) in templates
                         and min(e, m["end"]) - max(s, m["start"]) >= 0.4 * win})
        base_cols = [templates[o] for o in others]

        def residual(extra: int | None) -> float:
            cols = base_cols + ([templates[extra]] if extra is not None else [])
            if not cols:
                return 1.0
            _, r = nnls(np.column_stack(cols), S)
            return float(r)

        r_none = residual(None)
        res = {c: residual(c) for c in candidates(p) if c in templates}
        if p not in res:
            results.append({"id": n["id"], "status": "unchecked", "match": None, "suggest": None})
            continue
        best = min(res, key=res.get)
        explained_p = max(0.0, r_none - res[p])            # 这个音帮忙“解释”了多少能量
        explained_best = max(0.0, r_none - res[best])
        cents = float(n.get("cents") or 0.0)
        if best != p and res[best] < res[p] * (1 - margin) and explained_best > gain_ratio * explained_p:
            status = "suspect"
        elif abs(cents) > cents_tolerance:
            status = "offpitch"
        else:
            status = "ok"
        results.append({"id": n["id"], "status": status, "match": round(1 - res[p], 3),
                        "suggest": best if status == "suspect" else None,
                        "residuals": {str(k): round(v, 3) for k, v in res.items()}})
    return results
