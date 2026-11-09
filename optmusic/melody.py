"""单旋律引擎（人声、独奏）：复音模型主干 + pYIN 补漏 / 去矛盾 + 沿 pYIN 音高延伸 + 合并碎片 / 漂移 / 滑音 + 按重新起音切开。

服务端与评测脚本（tests/vocal_bench.py、tests/vocal_report.py）共用这一份实现。
算法与评测见 docs/算法原理.md 1.5。
"""
from __future__ import annotations

import librosa
import numpy as np

from . import analysis, poly

VOICED_MIN = 0.02        # 补漏用的 pYIN 置信度下限（仅 pYIN 引擎为 0.25）
EXT_TOL = 1.0            # 延伸时允许的音高偏差（半音）
EXT_BACK = 0.04          # 起点最多往前延伸（秒）
EXT_RMS_DROP = 12.0      # 响度比该音峰值低超过这么多 dB 就停止延伸（避免伸进混响尾巴 / 换气）
EXT_MAX_GAP = 2          # 延伸时允许连续“不符合”的帧数（pYIN 偶尔断一两帧）
MERGE_GAP = 0.08         # 同音高相邻碎片：间隙 ≤ 80 ms 才考虑合并
MERGE_DIP_DB = 3.0       # 交界处响度下凹 < 3 dB 才合并（真正重唱 / 换字会有明显下凹）
DRIFT_SEMI = 0.6         # 相邻相差 1 半音的两段，pYIN 中段中位数相差 < 0.6 半音 → 其实是同一个音（音高漂移）
GLIDE_MAX = 0.12         # 短于 120 ms、与相邻长音相差 1 半音的小音 → 滑音 / 装饰，并进长音
SPLIT_DIP_DB = 6.0       # 长音内部响度下凹 ≥ 6 dB → 重新起音（换字），切开
SPLIT_MIN = 0.10         # 切开后每段至少 100 ms
TRIM_START = 0.010       # 最后统一修正：起点后移 10 ms、终点前移 10 ms（抵消延伸带来的系统性偏长）
TRIM_END = 0.010


def extend_notes(notes: list[dict], pyin: tuple, rms_db: np.ndarray, hop: float,
                 tol: float = EXT_TOL, back: float = EXT_BACK, rms_drop: float = EXT_RMS_DROP,
                 max_gap: int = EXT_MAX_GAP) -> list[dict]:
    """把音符沿 pYIN 音高往后延伸（直到下一个音开始）、往前最多 back 秒：
    该帧 pYIN 判有声、与音符音高相差 ≤ tol 半音、响度不低于该音峰值 − rms_drop dB，就算这个音的延续。

    依据：vocadito 上没被覆盖的“在唱”帧里，80% 在两个音之间的短空隙、20% 在音的首尾，
    几乎没有整句漏掉——音符偏短（起得晚、收得早），而不是漏音。"""
    f0, vf, _ = pyin
    m = librosa.hz_to_midi(f0)
    n_fr = len(m)
    ns = sorted([dict(n) for n in notes], key=lambda n: n["start"])
    for i, n in enumerate(ns):
        nxt = ns[i + 1]["start"] if i + 1 < len(ns) else n_fr * hop
        prv = ns[i - 1]["end"] if i > 0 else 0.0
        p = n["pitch"] + (n.get("cents") or 0) / 100
        a0 = int(n["start"] / hop)
        b0 = max(a0 + 1, int(n["end"] / hop))
        peak = float(rms_db[a0:min(b0, len(rms_db))].max()) if a0 < len(rms_db) else -999.0

        def ok(k: int) -> bool:
            return (0 <= k < n_fr and vf[k] and not np.isnan(m[k]) and abs(m[k] - p) <= tol
                    and k < len(rms_db) and rms_db[k] >= peak - rms_drop)

        k, miss, last = int(n["end"] / hop), 0, n["end"]
        while k * hop < nxt and miss <= max_gap:
            if ok(k):
                last, miss = (k + 1) * hop, 0
            else:
                miss += 1
            k += 1
        n["end"] = min(max(n["end"], last), nxt)

        if back > 0:
            k, miss, first = a0 - 1, 0, n["start"]
            lim = max(prv, n["start"] - back)
            while k * hop >= lim and miss <= max_gap:
                if ok(k):
                    first, miss = k * hop, 0
                else:
                    miss += 1
                k -= 1
            n["start"] = max(min(n["start"], first), prv)
    return ns


def merge_fragments(notes: list[dict], rms_db: np.ndarray, hop: float,
                    gap: float = MERGE_GAP, dip_db: float = MERGE_DIP_DB) -> list[dict]:
    """把被切碎的同一个音合并回来：相邻两音同音高、间隙 ≤ gap，且交界处响度下凹 < dip_db。

    在 vocadito 上，“一个音被切成几段”是最大的编辑负担（每 100 音约 26 次合并）。
    无条件合并会把真正的重复音（同音高连唱两个字）也并掉；重唱 / 换字时交界处有辅音或短暂断开，
    响度会明显下凹，以此区分。"""
    ns = sorted([dict(n) for n in notes], key=lambda n: n["start"])
    res: list[dict] = []
    for n in ns:
        if res:
            p = res[-1]
            if n["pitch"] == p["pitch"] and n["start"] - p["end"] <= gap:
                a, k, b = int(p["start"] / hop), int(n["start"] / hop), int(n["end"] / hop)
                if b > k > a and b <= len(rms_db):
                    dip = min(rms_db[a:k].max(), rms_db[k:b].max()) - rms_db[max(a, k - 3):k + 3].min()
                    if dip < dip_db:
                        p["end"] = max(p["end"], n["end"])
                        if "velocity" in n and "velocity" in p:
                            p["velocity"] = max(p["velocity"], n["velocity"])
                        continue
        res.append(n)
    return res


def _seg_median(pm: np.ndarray, s: float, e: float, hop: float, trim: float = 0.2):
    d = e - s
    seg = pm[int((s + trim * d) / hop): int((e - trim * d) / hop) + 1]
    seg = seg[~np.isnan(seg)]
    return float(np.median(seg)) if len(seg) >= 3 else None


def merge_drift(notes: list[dict], pm: np.ndarray, hop: float, thresh: float = DRIFT_SEMI,
                gap: float = MERGE_GAP) -> list[dict]:
    """音高漂移：相邻两段相差 1 半音，但 pYIN 测到的真实音高（中段中位数）相差 < thresh 半音
    ——歌手的音高在两个半音之间晃，被取整成了两个音。合并，音高取合起来的 pYIN 中位数。"""
    ns = sorted([dict(n) for n in notes], key=lambda n: n["start"])
    res: list[dict] = []
    for n in ns:
        if res:
            p = res[-1]
            if abs(n["pitch"] - p["pitch"]) == 1 and n["start"] - p["end"] <= gap:
                m1, m2 = _seg_median(pm, p["start"], p["end"], hop), _seg_median(pm, n["start"], n["end"], hop)
                if m1 is not None and m2 is not None and abs(m1 - m2) < thresh:
                    allm = _seg_median(pm, p["start"], n["end"], hop, trim=0.1)
                    p["end"] = max(p["end"], n["end"])
                    if allm is not None:
                        p["pitch"] = int(round(allm))
                        p["cents"] = round((allm - round(allm)) * 100, 1)
                    continue
        res.append(n)
    return res


def absorb_glides(notes: list[dict], short: float = GLIDE_MAX, gap: float = MERGE_GAP) -> list[dict]:
    """滑音 / 装饰：短于 short、紧挨着一个更长且只差 1 半音的音 → 并进那个长音。"""
    ns = sorted([dict(n) for n in notes], key=lambda n: n["start"])
    changed = True
    while changed:
        changed = False
        for i, n in enumerate(ns):
            d = n["end"] - n["start"]
            if d >= short:
                continue
            best = None
            for j in (i - 1, i + 1):
                if 0 <= j < len(ns):
                    m = ns[j]
                    g = n["start"] - m["end"] if j < i else m["start"] - n["end"]
                    if g <= gap and abs(m["pitch"] - n["pitch"]) == 1 and m["end"] - m["start"] > d:
                        if best is None or (m["end"] - m["start"]) > (ns[best]["end"] - ns[best]["start"]):
                            best = j
            if best is not None:
                m = ns[best]
                m["start"], m["end"] = min(m["start"], n["start"]), max(m["end"], n["end"])
                del ns[i]
                changed = True
                break
    return ns


def split_rearticulation(notes: list[dict], rms_db: np.ndarray, pm: np.ndarray, hop: float,
                         dip_db: float = SPLIT_DIP_DB, min_piece: float = SPLIT_MIN) -> list[dict]:
    """长音内部的重新起音（换字、同音高连唱两个字）：某处响度比前后两侧峰值都低 ≥ dip_db → 切开，
    每段音高按该段 pYIN 中位数重新取整。"""
    res: list[dict] = []
    mp = max(1, int(min_piece / hop))
    for n in sorted([dict(x) for x in notes], key=lambda x: x["start"]):
        a, b = int(n["start"] / hop), min(len(rms_db), int(n["end"] / hop))
        if b - a < 2 * mp:
            res.append(n)
            continue
        r = rms_db[a:b]
        cuts, k = [], mp
        while k < len(r) - mp:
            left = r[max(0, k - 3 * mp):k].max()
            right = r[k + 1:min(len(r), k + 3 * mp)].max()
            if r[k] == r[max(0, k - 2):k + 3].min() and min(left, right) - r[k] >= dip_db:
                cuts.append(a + k)
                k += mp
            else:
                k += 1
        if not cuts:
            res.append(n)
            continue
        bounds = [n["start"]] + [c * hop for c in cuts] + [n["end"]]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e - s < 0.6 * min_piece:
                continue
            piece = dict(n, start=s, end=e)
            m = _seg_median(pm, s, e, hop, trim=0.15)
            if m is not None:
                piece["pitch"] = int(round(m))
                piece["cents"] = round((m - round(m)) * 100, 1)
            res.append(piece)
    return res


def trim_bias(notes: list[dict], ds: float = TRIM_START, de: float = TRIM_END, min_dur: float = 0.03) -> list[dict]:
    """统一修正起止点的系统偏差（vocadito：起点中位偏早 12 ms、终点中位偏晚 12 ms）。"""
    res = []
    for n in notes:
        s, e = n["start"] + ds, n["end"] - de
        if e - s < min_dur:
            s, e = n["start"], n["end"]
        res.append(dict(n, start=s, end=e))
    return res


def legato_fill(notes: list[dict], max_gap: float) -> list[dict]:
    """连奏填满：相邻两音间隙 ≤ max_gap 时，前一个音一直延到后一个音开始（MIDI“首尾相接”的写法）。
    这是记谱习惯而不是对错：CSD 的标注 68% 首尾相接，vocadito 只有 14%（见 docs/编辑量优化.md）。"""
    ns = sorted([dict(n) for n in notes], key=lambda n: n["start"])
    for a, b in zip(ns, ns[1:]):
        if 0 < b["start"] - a["end"] <= max_gap:
            a["end"] = b["start"]
    return ns


def transcribe(y: np.ndarray, out: dict, pyin: tuple, poly_params: dict | None = None,
               extend: bool = True) -> tuple[list[dict], dict, dict]:
    """返回 (音符, 统计, pYIN 音高曲线)。poly_params 里的 legato_gap > 0 时做“连奏填满”。"""
    sr = analysis.SR
    mono, curve = analysis.transcribe_mono(y, voiced_prob_min=VOICED_MIN, pyin=pyin)
    pn = poly.decode(out, poly.PolyParams.from_dict(poly_params))
    analysis.refine_cents(y, sr, pn)                     # 复音音符用频谱峰值测精确音分
    notes, st = poly.fuse_melody_poly_first(pn, [dict(n) for n in mono], curve)
    if extend:
        rms = librosa.amplitude_to_db(
            librosa.feature.rms(y=y, frame_length=analysis.FRAME, hop_length=analysis.HOP)[0], ref=1.0)
        notes = extend_notes(notes, pyin, rms, analysis.HOP / sr)
        hop = analysis.HOP / sr
        pm = np.where(pyin[1], librosa.hz_to_midi(pyin[0]), np.nan)
        before = len(notes)
        notes = merge_fragments(notes, rms, hop)          # 同音高碎片
        notes = merge_drift(notes, pm, hop)               # 音高漂移被取整成相邻两音
        notes = absorb_glides(notes)                      # 滑音 / 装饰小音
        st["merged"] = before - len(notes)
        before = len(notes)
        notes = split_rearticulation(notes, rms, pm, hop) # 长音内部重新起音
        st["split"] = len(notes) - before
        notes = trim_bias(notes)                          # 起止点系统偏差修正
        legato = float((poly_params or {}).get("legato_gap") or 0)
        if legato > 0:
            notes = legato_fill(notes, legato)            # 可选：连奏填满
    return notes, st, curve
