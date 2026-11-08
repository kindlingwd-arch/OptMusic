"""复音转写引擎：basic-pitch（ONNX）+ 后处理。

两步分离：
1. infer()  ：运行神经网络，得到三张后验图（每帧 × 88 个琴键）
               note    —— 这个音“正在响”的概率
               onset   —— 这个音“刚被弹下”的概率
               contour —— 1/3 半音精度的音高显著度（用于算音分偏差）
             结果按工程缓存，调参数不用重跑模型。
2. decode() ：阈值 + 峰值检测 → 音符，再做本模块的后处理：
               a. 合并被切碎的长音（同音高、间隙很小、且没有新的起音）
               b. 去掉泛音“幽灵音”（与更强的低八度/十二度/两个八度音同时响、且明显更弱）
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FPS = 22050 / 256          # basic-pitch 输出帧率（约 86.13 帧/秒）
MIDI_OFFSET = 21           # 第 0 个频带 = A0

def frame_times(n_frames: int) -> np.ndarray:
    """模型帧号 → 秒。basic-pitch 按 2 秒窗口拼接输出，每个窗口有约 10 ms 的偏移，
    帧号与时间并非严格线性（200 秒累计约 1 秒），必须用它自己的换算。"""
    from basic_pitch.note_creation import model_frames_to_time
    return model_frames_to_time(n_frames)


def time_to_frame(times: np.ndarray, t: float) -> int:
    return int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))


_model = None
_model_lock = threading.Lock()


def available() -> bool:
    try:
        import basic_pitch  # noqa: F401
        import onnxruntime  # noqa: F401
        return _model_path().exists()
    except Exception:
        return False


def _model_path() -> Path:
    import basic_pitch
    return Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from basic_pitch.inference import Model
            _model = Model(_model_path())
        return _model


def infer(audio_path: str) -> dict:
    from basic_pitch.inference import run_inference
    out = run_inference(audio_path, _get_model())
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def save_output(out: dict, path: Path):
    np.savez_compressed(path, **out)


def load_output(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@dataclass
class PolyParams:
    # 默认值由 tests/poly_tune.py 在合成基准上自动搜索得到（min_note 从 120 放宽到 90，兼顾十六分音符）
    onset_thresh: float = 0.6      # 起音灵敏度：越低越容易认出新音（也越多误报）
    frame_thresh: float = 0.4      # 持续灵敏度：越低音符越长、弱音越多
    min_note_ms: float = 90        # 最短音符
    merge_gap_ms: float = 60       # 合并同音高碎片的最大间隙
    reattack: float = 0.0          # 间隙处起音概率低于此值才合并（高于则认为是真的重复弹奏）
    ghost_ratio: float = 0.8       # 泛音判定：比低音弱到此比例以下才删除
    ghost_overlap: float = 0.5     # 泛音判定：时间上至少有这么大比例被低音覆盖
    backtrack_ms: float = 40       # 起点最多前移多少
    backtrack_level: float = 0.35  # 前移到“正在响”的概率低于此值为止
    melodia: bool = True           # basic-pitch 的补漏步骤（找没有明显起音的持续音）

    @classmethod
    def from_dict(cls, d: dict | None) -> "PolyParams":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, type(getattr(p, k))(v))
        return p

    def to_dict(self):
        return asdict(self)


def decode(out: dict, params: PolyParams | None = None, stats: dict | None = None) -> list[dict]:
    p = params or PolyParams()
    events = notes_polyphonic(out["note"], out["onset"], p.onset_thresh, p.frame_thresh,
                              min_note_len=max(1, int(round(p.min_note_ms / 1000 * FPS))),
                              melodia=p.melodia)
    contour = out["contour"]
    T = frame_times(len(out["note"]))
    notes = []
    for s, e, pitch, amp in events:
        notes.append({"start": float(T[s]), "end": float(T[min(e, len(T) - 1)]), "pitch": int(pitch),
                      "amp": float(amp), "cents": note_cents(contour, s, e, pitch)})
    n_raw = len(notes)
    notes = merge_fragments(notes, out["note"], out["onset"], p, T)
    n_merged = len(notes)
    notes = remove_ghosts(notes, p)
    refine_onsets(notes, out["note"], p, T)
    if stats is not None:
        stats.update(raw=n_raw, after_merge=n_merged, final=len(notes))

    for n in notes:
        n["velocity"] = int(np.clip(30 + n["amp"] * 160, 20, 127))
        n["confidence"] = round(n.pop("amp"), 3)
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def notes_polyphonic(frames: np.ndarray, onsets: np.ndarray, onset_thresh: float, frame_thresh: float,
                     min_note_len: int, melodia: bool = True, energy_tol: int = 11):
    """basic-pitch 的 output_to_notes_polyphonic 的等价改写（结果相同，快约 20 倍）。

    第一步：在起音图的局部峰值（> onset_thresh）处开始一个音，沿时间延伸到“正在响”概率
            连续 energy_tol 帧低于 frame_thresh 为止，并把用掉的能量清零。
    第二步（melodia）：剩余能量里从最大值开始往前后延伸，补上没有明显起音的持续音。
            原实现每次都对整张图求 argmax；因为能量只会被清零、不会增加，
            按初始值从大到小排序后依次处理（跳过已清零的格子）与原实现等价。"""
    import scipy.signal
    from basic_pitch.note_creation import get_infered_onsets

    n_frames, n_bins = frames.shape
    max_bin = n_bins - 1
    onsets = get_infered_onsets(onsets, frames)
    peak = np.zeros(onsets.shape)
    idx = scipy.signal.argrelmax(onsets, axis=0)
    peak[idx] = onsets[idx]
    on_t, on_f = np.where(peak >= onset_thresh)
    order = np.argsort(on_t, kind="stable")[::-1]          # 从后往前处理（与原实现一致）
    remaining = frames.copy()
    events = []

    def clear(i0, i1, f):
        remaining[i0:i1, f] = 0
        if f < max_bin:
            remaining[i0:i1, f + 1] = 0
        if f > 0:
            remaining[i0:i1, f - 1] = 0

    for t0, f in zip(on_t[order], on_f[order]):
        if t0 >= n_frames - 1:
            continue
        col = remaining[:, f]
        i, k = t0 + 1, 0
        while i < n_frames - 1 and k < energy_tol:
            k = k + 1 if col[i] < frame_thresh else 0
            i += 1
        i -= k
        if i - t0 <= min_note_len:
            continue
        clear(t0, i, f)
        events.append((int(t0), int(i), int(f) + MIDI_OFFSET, float(frames[t0:i, f].mean())))

    if melodia:
        cand_t, cand_f = np.where(remaining > frame_thresh)
        vals = remaining[cand_t, cand_f]
        flat = cand_t * n_bins + cand_f
        order = np.lexsort((flat, -vals))                  # 值降序，同值按扁平下标升序（同 argmax）
        for t_mid, f in zip(cand_t[order], cand_f[order]):
            if remaining[t_mid, f] <= frame_thresh:
                continue
            remaining[t_mid, f] = 0
            col = remaining[:, f]
            i, k = t_mid + 1, 0
            while i < n_frames - 1 and k < energy_tol:
                k = k + 1 if col[i] < frame_thresh else 0
                clear(i, i + 1, f)
                i += 1
            i_end = i - 1 - k
            i, k = t_mid - 1, 0
            while i > 0 and k < energy_tol:
                k = k + 1 if col[i] < frame_thresh else 0
                clear(i, i + 1, f)
                i -= 1
            i_start = i + 1 + k
            if i_end - i_start <= min_note_len:
                continue
            events.append((int(i_start), int(i_end), int(f) + MIDI_OFFSET, float(frames[i_start:i_end, f].mean())))
    return events


def note_cents(contour: np.ndarray, s: int, e: int, pitch: int) -> float:
    """用 1/3 半音精度的音高显著度图估计音分偏差：每帧在 ±2 个子频带内取最大，再取中位数。"""
    c = (pitch - MIDI_OFFSET) * 3
    lo, hi = max(0, c - 2), min(contour.shape[1], c + 3)
    seg = contour[s:e, lo:hi]
    if seg.size == 0:
        return 0.0
    off = np.argmax(seg, axis=1) + lo - c
    return round(float(np.clip(np.median(off) * 100 / 3, -50, 50)), 1)


def merge_fragments(notes: list[dict], frame: np.ndarray, onset: np.ndarray, p: PolyParams,
                    T: np.ndarray) -> list[dict]:
    """把被切碎的长音合并回来。同一音高的相邻两段，满足以下条件就合并：
    - 间隙 ≤ merge_gap，或间隙里“正在响”的概率一直不低（声音其实没断）；
    - 且后一段开头没有明显的新起音（起音概率 < reattack）——否则是真的重复弹奏。"""
    gap = p.merge_gap_ms / 1000
    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(n["pitch"], []).append(n)
    out = []
    for pitch, ns in by_pitch.items():
        ns.sort(key=lambda n: n["start"])
        col = pitch - MIDI_OFFSET
        cur = ns[0]
        for n in ns[1:]:
            f = time_to_frame(T, n["start"])
            lo, hi = max(0, f - 2), min(len(onset), f + 3)
            attack = float(onset[lo:hi, col].max()) if 0 <= col < onset.shape[1] and hi > lo else 1.0
            g0, g1 = time_to_frame(T, cur["end"]), f
            sustained = (g1 <= g0) or (0 <= col < frame.shape[1]
                                       and float(frame[g0:g1 + 1, col].min()) > 0.5 * p.frame_thresh)
            if (n["start"] - cur["end"] <= gap or sustained) and attack < p.reattack:
                w1, w2 = cur["end"] - cur["start"], n["end"] - n["start"]
                cur["amp"] = (cur["amp"] * w1 + n["amp"] * w2) / max(1e-6, w1 + w2)
                cur["end"] = max(cur["end"], n["end"])
            else:
                out.append(cur)
                cur = n
        out.append(cur)
    return out


GHOST_INTERVALS = (12, 19, 24)   # 第 2、3、4 泛音分别比基频高 12、19、24 个半音


def remove_ghosts(notes: list[dict], p: PolyParams) -> list[dict]:
    """删除“泛音幽灵音”：它的时间范围基本被一个更强的、低 12/19/24 半音的音覆盖。"""
    if p.ghost_ratio <= 0:
        return notes
    keep = []
    for n in notes:
        dur = n["end"] - n["start"]
        ghost = False
        for m in notes:
            if m is n or (n["pitch"] - m["pitch"]) not in GHOST_INTERVALS:
                continue
            overlap = min(n["end"], m["end"]) - max(n["start"], m["start"])
            if (overlap >= p.ghost_overlap * dur and n["start"] >= m["start"] - 0.05
                    and n["amp"] < p.ghost_ratio * m["amp"]):
                ghost = True
                break
        if not ghost:
            keep.append(n)
    return keep


def refine_onsets(notes: list[dict], frame: np.ndarray, p: PolyParams, T: np.ndarray):
    """起点前移：模型在“起音概率峰值”处开始一个音，而缓慢起音（弦乐、垫音）的声音其实更早出现。
    沿该音高往前找，直到“正在响”的概率降到 backtrack_level 以下（最多前移 backtrack_ms），
    且不越过同音高前一个音符的结尾。"""
    if p.backtrack_ms <= 0:
        return
    max_back = int(p.backtrack_ms / 1000 * FPS)
    last_end: dict[int, float] = {}
    for n in sorted(notes, key=lambda n: n["start"]):
        col = n["pitch"] - MIDI_OFFSET
        if not 0 <= col < frame.shape[1]:
            continue
        f = time_to_frame(T, n["start"])
        limit = max(0, f - max_back, time_to_frame(T, last_end.get(n["pitch"], 0.0)))
        k = f
        while k - 1 >= limit and frame[k - 1, col] >= p.backtrack_level:
            k -= 1
        n["start"] = float(T[k])
        last_end[n["pitch"]] = n["end"]


def estimate_polyphony(y: np.ndarray, sr: int) -> float:
    """粗估同时发声的音级数（色度图中显著音级个数的中位数），用于自动选择引擎。"""
    import librosa
    H = librosa.effects.harmonic(y[: sr * 60])
    C = librosa.feature.chroma_cqt(y=H, sr=sr, hop_length=2048)
    loud = librosa.feature.rms(y=H, hop_length=2048)[0][: C.shape[1]]
    frames = C[:, loud > 0.1 * loud.max()] if loud.max() > 0 else C
    if frames.shape[1] == 0:
        return 0.0
    return float(np.median((frames > 0.5).sum(axis=0)))
