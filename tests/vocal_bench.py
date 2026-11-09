r"""人声转写评测：vocadito 数据集（40 段独唱，7 种语言，两位标注者分别标注音符，CC BY 4.0）。

数据：https://zenodo.org/records/5557945 → 解压到 workspace/datasets/vocadito/
  （Audio/vocadito_N.wav，Annotations/Notes/vocadito_N_notesA1.csv / A2.csv：起点秒, 音高Hz, 时长秒）
运行：.venv\Scripts\python.exe -m tests.vocal_bench
评分（mir_eval 音符级 F1，参考标注 A1 与 A2 分别评分后取平均）：
  音高+起点  ：音高误差 ≤ 50 音分且起点误差 ≤ 50 ms
  含终点      ：再要求终点误差 ≤ max(50 ms, 20% 时长)
  仅起点      ：只看起点（不管音高）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from optmusic import analysis, melody, poly

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "workspace" / "datasets" / "vocadito"
CACHE = ROOT / "workspace" / "datasets" / "vocadito_cache"


def load_notes(path: Path) -> list[dict]:
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            s, hz, d = float(row[0]), float(row[1]), float(row[2])
            if hz > 0 and d > 0:
                out.append({"start": s, "end": s + d, "hz": hz})
    return out


def tracks():
    for wav in sorted((DATA / "Audio").glob("vocadito_*.wav"), key=lambda p: int(p.stem.split("_")[1])):
        a1 = DATA / "Annotations" / "Notes" / f"{wav.stem}_notesA1.csv"
        a2 = DATA / "Annotations" / "Notes" / f"{wav.stem}_notesA2.csv"
        if a1.exists() and a2.exists():
            yield wav, load_notes(a1), load_notes(a2)


def to_hz(n: dict) -> float:
    if "hz" in n:
        return n["hz"]
    return 440.0 * 2 ** ((n["pitch"] + (n.get("cents") or 0) / 100 - 69) / 12)


def scores(ref: list[dict], est: list[dict]) -> dict:
    import mir_eval

    def arr(ns):
        if not ns:
            return np.zeros((0, 2)), np.zeros(0)
        return (np.array([[n["start"], max(n["end"], n["start"] + 1e-3)] for n in ns]),
                np.array([to_hz(n) for n in ns]))
    ri, rp = arr(ref)
    ei, ep = arr(est)
    if len(ri) == 0 or len(ei) == 0:
        return {"F": 0.0, "F_off": 0.0, "F_on": 0.0, "P": 0.0, "R": 0.0}
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    _, _, f_off, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=0.2)
    _, _, f_on = mir_eval.transcription.onset_precision_recall_f1(ri, ei)
    return {"F": f, "F_off": f_off, "F_on": f_on, "P": p, "R": r}


def poly_output(wav: Path) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{wav.stem}.npz"
    out = poly.load_output(cache)
    if out is None:
        out = poly.infer(str(wav))
        poly.save_output(out, cache)
    return out


def engines(wav: Path, y: np.ndarray, out: dict) -> dict[str, list[dict]]:
    sr = analysis.SR
    pyin = analysis.run_pyin(y)
    mono25, curve25 = analysis.transcribe_mono(y, voiced_prob_min=0.25, pyin=pyin)
    mono02, curve02 = analysis.transcribe_mono(y, voiced_prob_min=0.02, pyin=pyin)
    poly_def = poly.decode(out)
    analysis.refine_cents(y, sr, poly_def)
    poly50 = poly.decode(out, poly.PolyParams.from_dict({"min_note_ms": 50}))
    analysis.refine_cents(y, sr, poly50)
    mel_v1, _ = poly.fuse_melody([dict(n) for n in mono02], poly50, curve02)
    mel_v2, _ = poly.fuse_melody_poly_first(poly_def, [dict(n) for n in mono02], curve02)
    mel_v3, _, _ = melody.transcribe(y, out, pyin)
    ratio = poly.monophony_ratio(out)
    auto = mel_v3 if ratio < 0.18 else poly_def
    return {"仅pYIN(置信>0.25)": mono25, "仅pYIN(置信>0.02)": mono02, "复音引擎": poly_def,
            "单旋律v1(pYIN主干)": mel_v1, "单旋律v2(复音主干+pYIN补漏)": mel_v2,
            "单旋律v3(v2+沿pYIN延伸)": mel_v3, "自动选择": auto,
            "_ratio": ratio}


def main(limit: int | None = None):
    rows: dict[str, list[dict]] = {}
    human = []
    ratios = []
    n_tracks = 0
    for wav, a1, a2 in tracks():
        if limit and n_tracks >= limit:
            break
        n_tracks += 1
        y = analysis.load_mono(str(wav))
        out = poly_output(wav)
        res = engines(wav, y, out)
        ratios.append(res.pop("_ratio"))
        for name, est in res.items():
            s1, s2 = scores(a1, est), scores(a2, est)
            rows.setdefault(name, []).append({k: (s1[k] + s2[k]) / 2 for k in s1})
        human.append(scores(a1, a2))
        print(f"  {wav.stem:14s} 参考音符 {len(a1):3d}/{len(a2):3d}  多音比例 {ratios[-1]:.0%}  "
              f"单旋律v3 F={rows['单旋律v3(v2+沿pYIN延伸)'][-1]['F']:.2f}", flush=True)

    def mean(lst, k):
        return float(np.mean([x[k] for x in lst]))
    print(f"\n{n_tracks} 段独唱（参考：两位标注者分别评分取平均）")
    print(f"自动选择：判为单旋律 {sum(r < 0.18 for r in ratios)}/{len(ratios)} 段，多音比例中位 {np.median(ratios):.1%}，最大 {max(ratios):.1%}")
    print(f"{'方案':24s} {'音高+起点F':>10s} {'P':>6s} {'R':>6s} {'含终点F':>8s} {'仅起点F':>8s}")
    for name, lst in rows.items():
        print(f"{name:24s} {mean(lst, 'F'):10.3f} {mean(lst, 'P'):6.3f} {mean(lst, 'R'):6.3f} "
              f"{mean(lst, 'F_off'):8.3f} {mean(lst, 'F_on'):8.3f}")
    print(f"{'人与人（A1 vs A2）':24s} {mean(human, 'F'):10.3f} {mean(human, 'P'):6.3f} {mean(human, 'R'):6.3f} "
          f"{mean(human, 'F_off'):8.3f} {mean(human, 'F_on'):8.3f}")
    return rows


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
