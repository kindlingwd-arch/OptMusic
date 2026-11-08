r"""端到端算法自测：合成一段已知答案的音频，检查音符 / BPM / 下拍 / 调性 / 和弦。
运行：.venv\Scripts\python.exe -m tests.test_pipeline
"""
import numpy as np
import soundfile as sf

from optmusic import analysis, harmony
from optmusic.host import builtin_synth

BPM, SR = 100, 44100
BEAT = 60 / BPM
PICKUP = BEAT  # 一拍弱起：第 1 小节从 1 拍之后开始
# (小节内拍位置, 时长拍, 音高)  C - Am - F - G 进行的旋律 + 低音
MELODY = [(0, 1, 64), (1, 1, 67), (2, 1, 72), (3, 1, 67),
          (4, 1, 69), (5, 1, 72), (6, 2, 76),
          (8, 1, 65), (9, 1, 69), (10, 1, 72), (11, 1, 69),
          (12, 1, 67), (13, 1, 71), (14, 2, 74)]
BASS = [(0, 4, 48), (4, 4, 45), (8, 4, 41), (12, 4, 43)]
EXPECT_CHORDS = ["C", "Am", "F", "G"]


def make_audio(path, with_bass=False):
    ev = [(p, 100, PICKUP + b * BEAT, PICKUP + (b + d) * BEAT - 0.02) for b, d, p in MELODY]
    if with_bass:
        ev += [(p, 80, PICKUP + b * BEAT, PICKUP + (b + d) * BEAT - 0.02) for b, d, p in BASS]
    dur = PICKUP + 17 * BEAT
    y = builtin_synth(ev, dur, SR).mean(axis=0)
    # 底鼓：每小节第一拍
    for bar in range(4):
        t0 = int((PICKUP + bar * 4 * BEAT) * SR)
        t = np.arange(int(0.12 * SR)) / SR
        kick = 0.8 * np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 25)
        y[t0:t0 + len(kick)] += kick
    sf.write(path, y / np.abs(y).max() * 0.8, SR)
    return ev


def main():
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "optmusic_test.wav")
    truth = make_audio(path)
    y = analysis.load_mono(path)
    notes, _ = analysis.transcribe_mono(y)
    grid = analysis.estimate_grid(y, note_onsets=[n["start"] for n in notes])
    print(f"音符：识别 {len(notes)} / 真实 {len(truth)}")
    hits = 0
    for p, _, s, e in truth:
        m = [n for n in notes if n["pitch"] == p and abs(n["start"] - s) < 0.06]
        hits += bool(m)
    print(f"  音高+起点(±60ms) 命中 {hits}/{len(truth)}")
    print(f"BPM：{grid.bpm:.2f}（真实 {BPM}）  拍号 {grid.beats_per_bar}/4  稳定度 {grid.confidence:.2f}")
    first_db = grid.origin + grid.beats_per_bar * 60 / grid.bpm
    print(f"下拍：第 1 小节起点 {grid.origin:.3f}s，下一小节 {first_db:.3f}s（真实下拍 {PICKUP:.3f}s）")

    ev2 = make_audio(path, with_bass=True)
    mid = [{"pitch": p, "velocity": v, "start": s, "end": e} for p, v, s, e in ev2]
    key, chords = harmony.analyze_harmony(mid, BPM, 4, PICKUP - 4 * BEAT)
    print(f"调性：{key['name']}（{key['signature_text']}，r={key['confidence']}）")
    names = [c["name"] for c in chords if c["name"] != "N"]
    print(f"和弦：{names}（期望 {EXPECT_CHORDS}）")
    key2, chords2 = harmony.analyze_harmony(
        [{"pitch": n["pitch"], "velocity": n["velocity"], "start": n["start"], "end": n["end"]} for n in notes],
        grid.bpm, grid.beats_per_bar, grid.origin)
    print(f"仅旋律(识别结果)：调性 {key2['name']}，和弦 {[c['name'] for c in chords2 if c['name'] != 'N']}")

    ok = (hits >= len(truth) - 1 and abs(grid.bpm - BPM) < 1.5 and grid.beats_per_bar == 4
          and abs((first_db - PICKUP + 1e-9)) < 0.08 and key["name"] == "C 大调" and names == EXPECT_CHORDS)
    print("结果：", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
