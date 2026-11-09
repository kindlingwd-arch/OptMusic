r"""人声转写“分数为什么低”的分析：换不同口径评分 + 误差分类。
用法：.venv\Scripts\python.exe -m tests.vocal_report
"""
import csv
from collections import Counter
import numpy as np
import mir_eval
from optmusic import analysis, melody, poly
from tests.vocal_bench import DATA, tracks, poly_output, to_hz

def f0_ref(stem):
    t, f = [], []
    with open(DATA / "Annotations" / "F0" / f"{stem}_f0.csv") as fh:
        for row in csv.reader(fh):
            t.append(float(row[0])); f.append(float(row[1]))
    return np.array(t), np.array(f)

def notes_to_f0(notes, times):
    f = np.zeros_like(times)
    for n in notes:
        k = (times >= n["start"]) & (times < n["end"])
        f[k] = to_hz(n)
    return f

def arr(ns):
    return (np.array([[n["start"], max(n["end"], n["start"] + 1e-3)] for n in ns]), np.array([to_hz(n) for n in ns]))

def note_f(ref, est, onset_tol=0.05, pitch_tol=50.0):
    ri, rp = arr(ref); ei, ep = arr(est)
    return mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=onset_tol,
                                                               pitch_tolerance=pitch_tol, offset_ratio=None)[2]

def classify(ref, est):
    """参考音符没被匹配上的原因 / 识别音符多出来的原因（音高+起点 50ms 口径）。"""
    ri, rp = arr(ref); ei, ep = arr(est)
    match = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    mr, me = {i for i, _ in match}, {j for _, j in match}
    miss, extra = Counter(), Counter()
    cents = lambda a, b: 1200 * np.log2(a / b)
    for i, r in enumerate(ref):
        if i in mr: continue
        ov = [j for j, e in enumerate(est) if min(r["end"], e["end"]) > max(r["start"], e["start"])]
        if not ov: miss["完全没识别到（静音/漏音）"] += 1; continue
        d = [abs(cents(to_hz(est[j]), to_hz(r))) for j in ov]
        j = ov[int(np.argmin(d))]; dc = min(d)
        if dc <= 50:
            miss["音高对，但起点差 >50ms（切分/时间）"] += 1
        elif abs(dc - 1200) <= 60 or abs(dc - 2400) <= 60:
            miss["八度错"] += 1
        elif dc <= 150:
            miss["差约一个半音（含音准偏差>50音分）"] += 1
        else:
            miss["音高错（其他）"] += 1
    for j, e in enumerate(est):
        if j in me: continue
        ov = [i for i, r in enumerate(ref) if min(r["end"], e["end"]) > max(r["start"], e["start"])]
        if not ov: extra["多出来的音（参考里是静音）"] += 1; continue
        d = min(abs(cents(to_hz(e), to_hz(ref[i]))) for i in ov)
        extra["同音高的碎片/重复切分" if d <= 50 else "音高与参考不同"] += 1
    return miss, extra

def main():
    rows = {"系统(单旋律v3)": [], "人与人(A1 vs A2)": []}
    miss_all, extra_all, nref, nest = Counter(), Counter(), 0, 0
    for wav, a1, a2 in tracks():
        y = analysis.load_mono(str(wav)); out = poly_output(wav)
        est, _, _ = melody.transcribe(y, out, analysis.run_pyin(y))
        t, f = f0_ref(wav.stem)
        for name, cand, refs in (("系统(单旋律v3)", est, (a1, a2)), ("人与人(A1 vs A2)", a2, (a1,))):
            r = {}
            for tol in (0.05, 0.1, 0.15):
                r[f"on{int(tol*1000)}"] = np.mean([note_f(rf, cand, tol) for rf in refs])
            r["on50_100c"] = np.mean([note_f(rf, cand, 0.05, 100) for rf in refs])
            r["onset_only"] = np.mean([mir_eval.transcription.onset_precision_recall_f1(arr(rf)[0], arr(cand)[0])[2] for rf in refs])
            est_f0 = notes_to_f0(cand, t)
            m = mir_eval.melody.evaluate(t, f, t, est_f0)
            r["RPA"] = m["Raw Pitch Accuracy"]; r["RCA"] = m["Raw Chroma Accuracy"]
            r["VR"] = m["Voicing Recall"]; r["OA"] = m["Overall Accuracy"]
            # 有声帧里：系统给出了音高的那些帧，音高对的比例
            v = (f > 0) & (est_f0 > 0)
            r["voiced_pitch_ok"] = float(np.mean(np.abs(1200 * np.log2(est_f0[v] / f[v])) <= 50)) if v.any() else 0
            rows[name].append(r)
        for rf in (a1, a2):
            mi, ex = classify(rf, est); miss_all += mi; extra_all += ex; nref += len(rf); nest += len(est)

    mean = lambda name, k: np.mean([r[k] for r in rows[name]])
    print("口径 （40 段独唱，系统 vs 两位标注者取平均；人与人 = A2 对 A1）")
    labels = [("on50", "音符：音高±50音分 + 起点±50ms（标准，之前报的 0.635）"),
              ("on100", "音符：音高±50音分 + 起点±100ms"), ("on150", "音符：音高±50音分 + 起点±150ms"),
              ("on50_100c", "音符：音高±100音分 + 起点±50ms"), ("onset_only", "音符：只看起点±50ms（不管音高）"),
              ("voiced_pitch_ok", "逐帧：两边都有音的帧里，音高对（±50音分）的比例"),
              ("RPA", "逐帧：参考有声帧里音高对的比例（RPA）"), ("RCA", "逐帧：同上但忽略八度（RCA）"),
              ("VR", "逐帧：参考有声帧里我们也判有声（VR）"), ("OA", "逐帧：整体准确率（OA，含无声帧）")]
    print(f"  {'':52s} {'系统':>6s} {'人与人':>6s}")
    for k, lab in labels:
        print(f"  {lab:52s} {mean('系统(单旋律v3)', k):6.3f} {mean('人与人(A1 vs A2)', k):6.3f}")
    print(f"\n参考音符 {nref} 个（两位标注者合计），没被匹配上的原因：")
    for k, v in miss_all.most_common(): print(f"  {v:5d} ({v/nref:5.1%})  {k}")
    print(f"识别出的音符 {nest} 个（计两次），多出来的原因：")
    for k, v in extra_all.most_common(): print(f"  {v:5d} ({v/nest:5.1%})  {k}")


if __name__ == "__main__":
    main()
