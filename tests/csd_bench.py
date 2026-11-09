r"""人声转写评测（第二个数据集）：CSD — Children's Song Dataset。

一位专业女歌手演唱 50 首英文 + 50 首韩文儿歌（每首两个调，共 200 段），附人工精修的 MIDI（含起止时间）。
许可：CC BY-NC-SA 4.0（仅限非商业研究）。数据：https://zenodo.org/records/4916302 → 解压到 workspace/datasets/csd/
运行：.venv\Scripts\python.exe -m tests.csd_bench [每种语言首数，默认 10]
与 vocadito 的区别：专业歌手、整首歌（1~2 分钟）、只有一位标注者（没有“人与人”参照）。
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from optmusic import analysis, melody, poly
from tests.edit_cost import edit_cost, print_table
from tests.vocal_bench import scores

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "workspace" / "datasets" / "csd"
CACHE = ROOT / "workspace" / "datasets" / "csd_cache"


def find_root() -> Path:
    for cand in (DATA / "CSD", DATA):
        if (cand / "english").is_dir() or (cand / "korean").is_dir():
            return cand
    raise SystemExit(f"找不到 CSD 数据：{DATA}")


def load_ref(csv_path: Path) -> list[dict]:
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for row in rows:
        try:
            s, e, p = float(row[0]), float(row[1]), float(row[2])
        except (ValueError, IndexError):
            continue                                    # 表头
        if e > s:
            out.append({"start": s, "end": e, "hz": 440.0 * 2 ** ((p - 69) / 12)})
    return out


def items(per_lang: int):
    root = find_root()
    for lang in ("english", "korean"):
        wavs = sorted((root / lang / "wav").glob("*a.wav"))[:per_lang]   # 每首只取 a 调
        for wav in wavs:
            ref = root / lang / "csv" / f"{wav.stem}.csv"
            if ref.exists():
                yield lang, wav, load_ref(ref)


def cached(name: str, fn):
    import pickle
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.write_bytes(pickle.dumps(v))
    return v


def main(per_lang: int = 10):
    rows: dict[str, list] = {}
    edits: dict[str, Counter] = {}
    n = 0
    for lang, wav, ref in items(per_lang):
        y = analysis.load_mono(str(wav))
        out = cached(wav.stem + "_bp", lambda: poly.infer(str(wav)))
        pyin = cached(wav.stem + "_pyin", lambda: analysis.run_pyin(y))
        pn = poly.decode(out)
        analysis.refine_cents(y, analysis.SR, pn)
        mono, _ = analysis.transcribe_mono(y, voiced_prob_min=0.25, pyin=pyin)
        mel, _, _ = melody.transcribe(y, out, pyin)
        ratio = poly.monophony_ratio(out)
        for name, est in (("仅pYIN", mono), ("复音引擎", pn), ("单旋律（当前）", mel)):
            rows.setdefault(name, []).append(scores(ref, est))
            edits.setdefault(name, Counter())
            edits[name] += edit_cost(ref, est)
        n += 1
        print(f"  {wav.stem:8s} {lang:7s} 参考 {len(ref):3d} 音  多音比例 {ratio:.0%}  "
              f"单旋律 F={rows['单旋律（当前）'][-1]['F']:.2f}", flush=True)
    print(f"\nCSD {n} 段（每段整首歌）")
    print(f"{'方案':14s} {'音高+起点F':>10s} {'P':>6s} {'R':>6s} {'含终点F':>8s} {'仅起点F':>8s}")
    for name, lst in rows.items():
        m = {k: float(np.mean([x[k] for x in lst])) for k in lst[0]}
        print(f"{name:14s} {m['F']:10.3f} {m['P']:6.3f} {m['R']:6.3f} {m['F_off']:8.3f} {m['F_on']:8.3f}")
    print()
    print_table(edits)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
