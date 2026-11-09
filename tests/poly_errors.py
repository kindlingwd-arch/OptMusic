"""复音转写误差分析：把误报 / 漏报按类型归类。用法：python -m tests.poly_errors [参数JSON]"""
import json, sys
from collections import Counter
import numpy as np
from tests.poly_bench import load_bench, model_output
from optmusic import poly

def analyze(params=None):
    fp_types, fn_layers, tp_layers, onset_err = Counter(), Counter(), Counter(), []
    for wav, truth in load_bench():
        notes = poly.decode(model_output(wav), poly.PolyParams.from_dict(params))
        used = set()
        matched_est = set()
        for i, t in enumerate(truth):   # 贪心匹配：同音高、起点 ±50ms
            cands = [j for j, n in enumerate(notes) if j not in matched_est and n["pitch"] == t["pitch"] and abs(n["start"] - t["start"]) <= 0.05]
            if cands:
                j = min(cands, key=lambda j: abs(notes[j]["start"] - t["start"]))
                matched_est.add(j); used.add(i); tp_layers[t["layer"]] += 1
            else:
                fn_layers[t["layer"]] += 1
                near = [n for n in notes if n["pitch"] == t["pitch"] and abs(n["start"] - t["start"]) <= 0.3]
                if near:
                    onset_err.append(min((n["start"] - t["start"] for n in near), key=abs))
        for j, n in enumerate(notes):
            if j in matched_est:
                continue
            # 误报归类
            sounding = [t for t in truth if t["start"] - 0.05 <= n["start"] <= t["end"]]
            same = [t for t in sounding if t["pitch"] == n["pitch"]]
            if same:
                fp_types["同音高·在真实音符中间（碎片/重复触发）" if n["start"] - same[0]["start"] > 0.05 else "同音高·起点偏差>50ms"] += 1
                continue
            iv = {n["pitch"] - t["pitch"] for t in sounding}
            if iv & {12, 19, 24, 28}:
                fp_types["泛音（高八度/十二度/两八度）"] += 1
            elif iv & {-12, -24}:
                fp_types["低八度"] += 1
            elif iv & {1, -1}:
                fp_types["相邻半音"] += 1
            elif any(abs(t["end"] - n["start"]) < 0.3 and t["pitch"] == n["pitch"] for t in truth):
                fp_types["前一个音的尾音/混响"] += 1
            else:
                fp_types["其他"] += 1
    tot_fp = sum(fp_types.values())
    print(f"误报 {tot_fp} 个：")
    for k, v in fp_types.most_common():
        print(f"   {v:4d}  {v/tot_fp:4.0%}  {k}")
    print("各声部 召回：", {l: f"{tp_layers[l]}/{tp_layers[l]+fn_layers[l]} = {tp_layers[l]/(tp_layers[l]+fn_layers[l]):.0%}" for l in ("pad", "bass", "melody")})
    if onset_err:
        e = np.array(onset_err)
        print(f"漏报中有 {len(e)} 个其实找到了同音高音符但起点偏差 >50ms：中位偏差 {np.median(e)*1000:+.0f}ms，偏晚的占 {(e>0).mean():.0%}")

if __name__ == "__main__":
    analyze(json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)
