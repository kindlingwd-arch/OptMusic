r"""复音转写基准：用本机 VST 渲染“答案已知”的复音音乐，给转写打分。

  生成数据（只需一次，结果在 workspace/bench/）：
    .venv\Scripts\python.exe -m tests.poly_bench make
  评测（默认参数 vs. 不做后处理）：
    .venv\Scripts\python.exe -m tests.poly_bench eval

评分用 mir_eval 的音符级 F1：
  - onset F1：起点误差 ≤ 50 ms 且音高相同算命中；
  - full  F1：在此基础上终点误差 ≤ max(50 ms, 20% 时长)。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "workspace" / "bench"
SR = 44100
VST3 = r"C:\Program Files\Common Files\VST3"
BBC = VST3 + r"\BBC Symphony Orchestra (64 Bit).vst3"
PAPU = VST3 + r"\PAPU.vst3"

PROGRESSIONS = {
    "C": [[0, 4, 7], [9, 0, 4], [5, 9, 0], [7, 11, 2]],          # C Am F G
    "G": [[7, 11, 2], [4, 7, 11], [0, 4, 7], [2, 6, 9]],         # G Em C D
    "F": [[5, 9, 0], [2, 5, 9], [10, 2, 5], [0, 4, 7, 10]],      # F Dm Bb C7
}
SCALES = {"C": [0, 2, 4, 5, 7, 9, 11], "G": [7, 9, 11, 0, 2, 4, 6], "F": [5, 7, 9, 10, 0, 2, 4]}


def compose(key: str, bpm: float, bars: int, seed: int):
    """返回 (pad, bass, melody) 三层音符：[(pitch, vel, start, end)]。"""
    rnd = random.Random(seed)
    beat = 60 / bpm
    bar = 4 * beat
    prog = PROGRESSIONS[key]
    pad, bass, mel = [], [], []
    for b in range(bars):
        chord = prog[b % len(prog)]
        t0 = 0.5 + b * bar
        # 和弦垫音：每小节一个，约 1/3 的小节拆成两个半小节（重复弹奏同一和弦）
        halves = [(t0, t0 + bar)] if rnd.random() > 0.35 else [(t0, t0 + 2 * beat), (t0 + 2 * beat, t0 + bar)]
        for s, e in halves:
            for pc in chord[:3]:
                p = 60 + ((pc - 0) % 12)
                p = p if p <= 67 else p - 12          # C4~G4 附近的密集排列
                pad.append((p, 70, s, e - 0.03))
            if len(chord) > 3:
                pad.append((60 + chord[3] % 12 - (12 if 60 + chord[3] % 12 > 67 else 0), 60, s, e - 0.03))
        # 低音：根音，第 1、3 拍
        root = 36 + chord[0] % 12
        bass.append((root, 90, t0, t0 + 2 * beat - 0.05))
        bass.append((root if rnd.random() > 0.3 else root + 7, 85, t0 + 2 * beat, t0 + bar - 0.05))
        # 旋律：C5~C6，八分 / 四分 / 二分音符混合，含重复音
        t = t0
        last = 72 + rnd.choice(SCALES[key]) % 12
        while t < t0 + bar - 1e-6:
            d = rnd.choice([0.5, 0.5, 1, 1, 1, 2]) * beat
            d = min(d, t0 + bar - t)
            if rnd.random() < 0.15:
                t += d            # 休止
                continue
            if rnd.random() > 0.25:   # 25% 概率重复上一个音
                step = rnd.choice([-2, -1, 1, 2, 3, -3])
                deg = [72 + s % 12 for s in SCALES[key]]
                deg += [d2 + 12 for d2 in deg]
                cands = sorted(deg, key=lambda x: abs(x - (last + step)))
                last = cands[0]
            mel.append((last, 100, t, t + d - 0.04))
            t += d
    return pad, bass, mel


def render(host, plugin_path, events, duration):
    if plugin_path:
        host.load(plugin_path)
    else:
        host.load("builtin")
    return host._render_events(events, duration)


def make():
    from pedalboard import Pedalboard, Reverb
    from optmusic.host import PluginHost

    BENCH.mkdir(parents=True, exist_ok=True)
    host = PluginHost()
    host.call = lambda fn, timeout=0: fn()   # 脚本里直接在主线程执行
    pieces = [
        # 名称, 调, 速度, 垫音音色, 低音音色, 旋律音色
        ("strings_C", "C", 90, BBC, None, BBC),   # BBC 默认音色是小提琴（G3 以下无声），低音用内置音色
        ("mixed_G", "G", 104, BBC, None, PAPU),
        ("synth_F", "F", 80, None, None, None),
    ]
    for name, key, bpm, t_pad, t_bass, t_mel in pieces:
        pad, bass, mel = compose(key, bpm, bars=12, seed=sum(map(ord, name)))  # 固定种子，保证可复现
        dur = max(e for *_, e in pad + bass + mel) + 2.5
        mix = None
        for plug, ev, gain in ((t_pad, pad, 0.6), (t_bass, bass, 0.8), (t_mel, mel, 1.0)):
            a = render(host, plug, ev, dur)
            a = a / (np.abs(a).max() + 1e-9) * gain
            mix = a if mix is None else mix + a
        mix = Pedalboard([Reverb(room_size=0.45, wet_level=0.18, dry_level=0.9)])(mix, SR)
        mix = mix / np.abs(mix).max() * 0.85
        sf.write(BENCH / f"{name}.wav", mix.T, SR)
        # 弦乐有约 60 ms 的起弓时间，真值起点相应后移，保证评测公平
        truth = []
        for layer, plug, ev in (("pad", t_pad, pad), ("bass", t_bass, bass), ("melody", t_mel, mel)):
            lag = 0.06 if plug == BBC else 0.0
            truth += [{"pitch": p, "start": s + lag, "end": e, "layer": layer} for p, _, s, e in ev]
        json.dump(truth, open(BENCH / f"{name}.json", "w"))
        print(f"{name}: {len(truth)} 个音符（垫音 {len(pad)} / 低音 {len(bass)} / 旋律 {len(mel)}）", flush=True)


# ---------------------------------------------------------------- 评测
def load_bench():
    items = []
    for wav in sorted(BENCH.glob("*.wav")):
        truth = json.load(open(wav.with_suffix(".json")))
        items.append((wav, truth))
    return items


def model_output(wav: Path):
    from optmusic import poly
    cache = wav.with_suffix(".npz")
    out = poly.load_output(cache)
    if out is None:
        out = poly.infer(str(wav))
        poly.save_output(out, cache)
    return out


def score(notes, truth):
    truth = [t for t in truth]
    import mir_eval
    def arr(ns):
        iv = np.array([[n["start"], n["end"]] for n in ns]) if ns else np.zeros((0, 2))
        hz = np.array([440 * 2 ** ((n["pitch"] - 69) / 12) for n in ns]) if ns else np.zeros(0)
        return iv, hz
    ri, rp = arr(truth)
    ei, ep = arr(notes)
    on = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    full = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=0.2)
    return {"P": on[0], "R": on[1], "F": on[2], "F_full": full[2]}


def evaluate(params: dict | None = None, verbose=True):
    from optmusic import poly
    rows = []
    for wav, truth in load_bench():
        out = model_output(wav)
        st = {}
        notes = poly.decode(out, poly.PolyParams.from_dict(params), stats=st)
        s = score(notes, truth) | {"n": len(notes), "truth": len(truth)} | st
        rows.append((wav.stem, s))
        if verbose:
            print(f"  {wav.stem:10s} 真值 {len(truth):3d}  识别 {len(notes):3d}  "
                  f"P {s['P']:.2f}  R {s['R']:.2f}  F {s['F']:.3f}  含终点F {s['F_full']:.3f}  "
                  f"(原始 {st['raw']} → 合并后 {st['after_merge']} → 去泛音 {st['final']})")
    mean = {k: float(np.mean([r[1][k] for r in rows])) for k in ("P", "R", "F", "F_full")}
    if verbose:
        print(f"  平均  P {mean['P']:.3f}  R {mean['R']:.3f}  F {mean['F']:.3f}  含终点F {mean['F_full']:.3f}")
    return mean


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "make":
        make()
    else:
        print("【不做后处理（basic-pitch 默认解码）】")
        evaluate({"merge_gap_ms": 0, "ghost_ratio": 0, "reattack": 0, "backtrack_ms": 0})
        print("【当前参数（含合并碎片 + 去泛音）】")
        evaluate()
