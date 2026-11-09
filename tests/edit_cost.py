r"""“编辑量”指标：把识别结果手工改成正确 MIDI，至少要做多少次编辑。

比学术的音符 F1 更贴近实际使用：用户在钢琴窗里删、补、合并、切开、改音高、拖动 / 拉长音符。
每 100 个正确音符需要的编辑次数越少越好。

编辑类型（每次计 1）：
  删除   识别出的音与任何真实音符都不重叠（或重叠太少）
  补音   某个真实音符完全没识别到
  合并   一个真实音符被识别成 k 段 → k−1 次
  切开   一个识别音符盖住了 m 个真实音符 → m−1 次（切开的分界点不再另计）
  改音高 音高误差 > 50 音分
  移动   起点误差 > 50 ms（若终点也偏、且偏移量接近，视为整体拖动，只算 1 次）
  拉伸   终点误差 > max(50 ms, 20% 时长)
用法：python -m tests.edit_cost      （在 vocadito 上评测各引擎与“人与人”）
"""
from __future__ import annotations

from collections import Counter

import numpy as np

ONSET_TOL = 0.05
PITCH_TOL = 50.0
OFFSET_RATIO = 0.2
ASSIGN_FRAC = 0.3       # 识别音符与真实音符重叠 ≥ 识别音符时长的 30%（或真实音符时长的 30%）才算“对应”
COVER_FRAC = 0.5        # 识别音符覆盖某真实音符 ≥ 50% 时长，才算“盖住”它（用于判断需要切开）


def _hz(n: dict) -> float:
    if "hz" in n:
        return n["hz"]
    return 440.0 * 2 ** ((n["pitch"] + (n.get("cents") or 0) / 100 - 69) / 12)


def edit_cost(ref: list[dict], est: list[dict]) -> Counter:
    ref = sorted(ref, key=lambda n: n["start"])
    est = sorted(est, key=lambda n: n["start"])
    c = Counter()
    if not ref:
        c["删除"] += len(est)
        return c
    rs = np.array([n["start"] for n in ref]); re_ = np.array([n["end"] for n in ref])
    rd = re_ - rs
    assigned: dict[int, list[int]] = {i: [] for i in range(len(ref))}
    span_of: dict[int, list[int]] = {}
    for j, e in enumerate(est):
        ov = np.clip(np.minimum(re_, e["end"]) - np.maximum(rs, e["start"]), 0, None)
        ed = max(1e-6, e["end"] - e["start"])
        cand = np.where((ov >= ASSIGN_FRAC * ed) | (ov >= ASSIGN_FRAC * rd))[0]
        if len(cand) == 0:
            c["删除"] += 1
            continue
        covers = [int(i) for i in np.where(ov >= COVER_FRAC * rd)[0]]
        if len(covers) >= 2:
            c["切开"] += len(covers) - 1
            span_of[j] = covers
            for i in covers:
                assigned[i].append(j)
        else:
            assigned[int(cand[np.argmax(ov[cand])])].append(j)
    for i, r in enumerate(ref):
        js = assigned[i]
        if not js:
            c["补音"] += 1
            continue
        c["合并"] += max(0, len(js) - 1)
        # 合并后的音：音高取与它重叠最多的那段
        ov = [min(r["end"], est[j]["end"]) - max(r["start"], est[j]["start"]) for j in js]
        main = est[js[int(np.argmax(ov))]]
        cents = abs(1200 * np.log2(_hz(main) / _hz(r)))
        edits = 0
        if cents > PITCH_TOL:
            c["改音高"] += 1
            edits += 1
        # 边界：切开产生的内部分界点不计；只看外侧边界
        s = min(est[j]["start"] for j in js)
        e_ = max(est[j]["end"] for j in js)
        for j in js:
            if j in span_of:
                sp = span_of[j]
                if sp[0] != i:
                    s = r["start"]           # 由“切开”得到的起点
                if sp[-1] != i:
                    e_ = r["end"]            # 由“切开”得到的终点
        ds, de = s - r["start"], e_ - r["end"]
        bad_s = abs(ds) > ONSET_TOL
        bad_e = abs(de) > max(ONSET_TOL, OFFSET_RATIO * (r["end"] - r["start"]))
        if bad_s and bad_e and abs(ds - de) <= ONSET_TOL:
            c["移动"] += 1
        else:
            c["移动"] += bad_s
            c["拉伸"] += bad_e
        if not (bad_s or bad_e) and edits == 0 and len(js) == 1 and js[0] not in span_of:
            c["_完全正确"] += 1
    c["_参考音符"] += len(ref)
    return c


def summarize(c: Counter) -> dict:
    n = c["_参考音符"]
    kinds = ["删除", "补音", "合并", "切开", "改音高", "移动", "拉伸"]
    total = sum(c[k] for k in kinds)
    return {"每100音编辑": 100 * total / n, "无需编辑": c["_完全正确"] / n,
            **{k: 100 * c[k] / n for k in kinds}}


def main():
    from optmusic import analysis, melody, poly
    from tests.vocal_bench import poly_output, tracks
    totals = {"复音引擎": Counter(), "单旋律v2": Counter(), "单旋律v3（当前）": Counter(), "人与人(A2→A1)": Counter()}
    for wav, a1, a2 in tracks():
        y = analysis.load_mono(str(wav)); out = poly_output(wav); pyin = analysis.run_pyin(y)
        pn = poly.decode(out); analysis.refine_cents(y, analysis.SR, pn)
        m02, c02 = analysis.transcribe_mono(y, voiced_prob_min=0.02, pyin=pyin)
        v2, _ = poly.fuse_melody_poly_first(pn, [dict(n) for n in m02], c02)
        v3, _, _ = melody.transcribe(y, out, pyin)
        for name, est in (("复音引擎", pn), ("单旋律v2", v2), ("单旋律v3（当前）", v3)):
            totals[name] += edit_cost(a1, est) + edit_cost(a2, est)
        totals["人与人(A2→A1)"] += edit_cost(a1, a2)
    print_table(totals)


def print_table(totals: dict):
    kinds = ["删除", "补音", "合并", "切开", "改音高", "移动", "拉伸"]
    print(f"{'方案':18s} {'每100音编辑':>10s} {'无需编辑':>8s} " + " ".join(f"{k:>5s}" for k in kinds))
    for name, c in totals.items():
        s = summarize(c)
        print(f"{name:18s} {s['每100音编辑']:10.1f} {s['无需编辑']:8.1%} " + " ".join(f"{s[k]:5.1f}" for k in kinds))


if __name__ == "__main__":
    main()
