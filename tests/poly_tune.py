"""复音后处理参数自动搜索（坐标上升）：每轮依次优化每个参数，目标 = 平均 onset F1 + 0.5 × 含终点 F1。
用法：python -m tests.poly_tune"""
import json, time
from tests.poly_bench import evaluate

GRID = {
    "onset_thresh":    [0.3, 0.4, 0.5, 0.6, 0.7],
    "frame_thresh":    [0.2, 0.25, 0.3, 0.4],
    "min_note_ms":     [50, 80, 120],
    "reattack":        [0.0, 0.1, 0.2, 0.3, 0.45],
    "merge_gap_ms":    [0, 30, 60, 120],
    "ghost_ratio":     [0.0, 0.4, 0.6, 0.8, 1.0],
    "ghost_overlap":   [0.5, 0.7, 0.9],
    "backtrack_ms":    [0, 40, 80, 150],
    "backtrack_level": [0.2, 0.35, 0.5],
}

def objective(m): return m["F"] + 0.5 * m["F_full"]

def tune(rounds=3):
    params = {}
    best = evaluate(params, verbose=False); best_o = objective(best)
    print(f"起点：F {best['F']:.3f}  含终点F {best['F_full']:.3f}", flush=True)
    for r in range(rounds):
        improved = False
        for k, vals in GRID.items():
            for v in vals:
                trial = dict(params, **{k: v})
                m = evaluate(trial, verbose=False); o = objective(m)
                if o > best_o + 1e-4:
                    best, best_o, params, improved = m, o, trial, True
            print(f"  第{r+1}轮 {k:15s} → {params.get(k, '默认')}   F {best['F']:.3f}  含终点F {best['F_full']:.3f}", flush=True)
        if not improved:
            break
    print("最优参数：", json.dumps(params, ensure_ascii=False))
    return params

if __name__ == "__main__":
    t = time.time(); tune(); print(f"用时 {time.time()-t:.0f}s")
