"""画图：模型热度图 + 真值（绿框）+ 识别结果（红框）。python -m tests.poly_plot <name> <t0> <t1> <out.png> [参数JSON]"""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tests.poly_bench import BENCH, model_output
from optmusic import poly
name, t0, t1, outp = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
params = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
truth = json.load(open(BENCH / f"{name}.json")); out = model_output(BENCH / f"{name}.wav")
notes = poly.decode(out, poly.PolyParams.from_dict(params))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
for ax, key, title in ((axes[0], "note", "“正在响”概率 (note)"), (axes[1], "onset", "“刚弹下”概率 (onset)")):
    a = out[key]
    ax.imshow(a.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
              extent=[0, len(a) / poly.FPS, 20.5, 108.5])
    for t in truth:
        if t["end"] >= t0 and t["start"] <= t1:
            ax.add_patch(Rectangle((t["start"], t["pitch"] - 0.5), t["end"] - t["start"], 1, fill=False, ec="#52c41a", lw=2))
    for n in notes:
        if n["end"] >= t0 and n["start"] <= t1:
            ax.add_patch(Rectangle((n["start"], n["pitch"] - 0.35), n["end"] - n["start"], 0.7, fill=False, ec="#ff4d4f", lw=1.3, ls="--"))
    ps = [t["pitch"] for t in truth]
    ax.set_xlim(t0, t1); ax.set_ylim(min(ps) - 4, max(ps) + 16); ax.set_title(f"{name}：{title}   绿框=真值  红虚线=识别")
plt.tight_layout(); plt.savefig(outp, dpi=80); print("saved", outp)
