r"""VST 逐音校验的评测：误报率（正确音被判错）与检出率（故意改错的音被抓住并给出正确建议）。
用法：.venv\Scripts\python.exe -m tests.check_bench [VST路径，默认内置合成器]
"""
from __future__ import annotations

import json
import random
import sys

import librosa
import numpy as np

from optmusic import analysis, checker
from optmusic.host import RENDER_SR, PluginHost
from tests.poly_bench import BENCH, load_bench


def templates_for(host: PluginHost, pitches) -> tuple[dict, dict]:
    lin, logp = {}, {}
    for p, audio in host.render_batch(sorted(pitches)).items():
        mono = librosa.resample(audio.mean(axis=0), orig_sr=RENDER_SR, target_sr=analysis.SR)[int(0.03 * analysis.SR):]
        lin[p] = checker.semitone_spectrum(mono, analysis.SR)
        logp[p] = analysis.cqt_profile(mono)
    return lin, logp


def old_check(y, sr, n, logp):
    """旧方法：只拿单个音的模板和原声片段比对数谱相似度。"""
    p = int(n["pitch"])
    seg = y[int(n["start"] * sr): int(min(n["end"], n["start"] + 1.0) * sr)]
    prof = analysis.cqt_profile(seg)
    sims = {c: float(prof @ logp[c]) for c in checker.candidates(p) if c in logp}
    best = max(sims, key=sims.get)
    return best if best != p and sims[best] - sims[p] > 0.03 else None


def main(plugin=None, per_piece=50, seed=7):
    host = PluginHost()
    host.call = lambda fn, timeout=0: fn()
    host.load(plugin or "builtin")
    rnd = random.Random(seed)
    stats = {m: {"fa": 0, "n": 0, "caught": 0, "right": 0, "m": 0} for m in ("旧方法", "新方法")}
    for wav, truth in load_bench():
        y = analysis.load_mono(str(wav)); sr = analysis.SR
        notes = [dict(t, id=i + 1, cents=0.0) for i, t in enumerate(truth)]
        lin, logp = templates_for(host, checker.needed_pitches(notes) | {int(n["pitch"]) + d for n in notes for d in (-13, 13, -2, 2, -11, 11, -24, 24) if 21 <= int(n["pitch"]) + d <= 108})
        sample = rnd.sample(notes, min(per_piece, len(notes)))
        # 误报：原样校验
        new_res = {r["id"]: r for r in checker.check_notes(y, sr, sample, notes, lin)}
        for n in sample:
            stats["旧方法"]["n"] += 1; stats["新方法"]["n"] += 1
            stats["旧方法"]["fa"] += old_check(y, sr, n, logp) is not None
            stats["新方法"]["fa"] += new_res[n["id"]]["status"] == "suspect"
        # 检出：把一个音改错（±1 半音或 ±1 八度），其他音不变
        for n in sample:
            d = rnd.choice([12, -12, 1, -1])
            wrong = dict(n, pitch=int(n["pitch"]) + d)
            if not 21 <= wrong["pitch"] <= 108:
                continue
            mod = [wrong if m["id"] == n["id"] else m for m in notes]
            for name in ("旧方法", "新方法"):
                stats[name]["m"] += 1
                if name == "旧方法":
                    sug = old_check(y, sr, wrong, logp)
                else:
                    r = checker.check_notes(y, sr, [wrong], mod, lin)[0]
                    sug = r["suggest"] if r["status"] == "suspect" else None
                stats[name]["caught"] += sug is not None
                stats[name]["right"] += sug == int(n["pitch"])
        print(f"{wav.stem} 完成", flush=True)
    for name, s in stats.items():
        print(f"{name}：误报率 {s['fa']/s['n']:.0%}（{s['fa']}/{s['n']}）  "
              f"检出率 {s['caught']/s['m']:.0%}  建议正确率 {s['right']/s['m']:.0%}（{s['right']}/{s['m']}）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
