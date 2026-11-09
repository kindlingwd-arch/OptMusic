"""FastAPI 服务：上传 / 分析 / 插件 / 渲染 / 校验 / 导出。"""

import io
import os
import sys
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, checker, harmony, melody, poly
from .host import BUILTIN, RENDER_SR, EditorOpen, PluginHost, scan_vst3
from .midi_export import export_midi

if getattr(sys, "frozen", False):
    # PyInstaller 打包后：界面文件在解包目录，工作区放到用户目录（程序目录可能不可写）
    WEB = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "web"
    WORK = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "OptMusic" / "workspace"
else:
    ROOT = Path(__file__).resolve().parent.parent
    WEB = ROOT / "web"
    WORK = ROOT / "workspace"
WORK.mkdir(parents=True, exist_ok=True)


def create_app(host: PluginHost) -> FastAPI:
    app = FastAPI(title="OptMusic")
    audio_cache: dict[str, np.ndarray] = {}
    ref_cache: dict[tuple, np.ndarray] = {}
    chroma_cache: dict[str, np.ndarray] = {}

    def audio_chroma(pid: str) -> np.ndarray:
        if pid not in chroma_cache:
            y = get_audio(pid)
            chroma_cache[pid] = librosa.feature.chroma_cqt(y=y, sr=analysis.SR, hop_length=1024).mean(axis=1)
        return chroma_cache[pid]

    def audio_path(pid: str) -> Path:
        d = WORK / pid
        files = [p for p in d.glob("source.*")] if d.is_dir() else []
        if not files:
            raise HTTPException(404, "找不到音频，请重新上传")
        return files[0]

    def get_audio(pid: str) -> np.ndarray:
        if pid not in audio_cache:
            audio_cache.clear()
            audio_cache[pid] = analysis.load_mono(str(audio_path(pid)))
        return audio_cache[pid]

    def wav_response(audio: np.ndarray, sr: int = RENDER_SR) -> Response:
        buf = io.BytesIO()
        sf.write(buf, np.clip(audio.T, -1, 1), sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav",
                        headers={"X-Plugin-Version": str(host.state_version)})

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except EditorOpen as e:
            raise HTTPException(409, str(e))
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ 音频
    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        pid = uuid.uuid4().hex[:12]
        d = WORK / pid
        d.mkdir(parents=True)
        ext = Path(file.filename or "a.wav").suffix.lower() or ".wav"
        dst = d / f"source{ext}"
        dst.write_bytes(await file.read())
        try:
            info = sf.info(str(dst))
            duration = info.duration
        except Exception:
            duration = float(librosa.get_duration(path=str(dst)))
        return {"id": pid, "name": file.filename, "duration": duration}

    @app.get("/api/audio/{pid}")
    def get_audio_file(pid: str):
        return FileResponse(audio_path(pid))

    class AnalyzeReq(BaseModel):
        id: str
        mode: str = "auto"             # auto | mono | poly
        beats_per_bar: int | None = None
        fmin: float = 65.0
        fmax: float = 1400.0
        min_note: float = 0.07
        poly_params: dict | None = None

    poly_cache: dict[str, dict] = {}
    tuning_cache: dict[str, float] = {}

    MELODY_RATIO = 0.18      # 多音比例低于此值判为单旋律（实测：独唱 0.06，合成旋律+鼓 0.15，和弦伴奏 0.25，复音合成 0.52+）
    mono_cache: dict[tuple, tuple] = {}

    # 单旋律引擎：复音音符为主干 + pYIN 补漏 / 去矛盾（见算法文档 1.5）
    MELODY_VOICED_MIN = 0.02     # 补漏用的 pYIN 置信度下限（仅 pYIN 引擎为 0.25）
    MELODY_POLY_DEFAULTS: dict = {"legato_gap": 0.0}   # 复音部分沿用复音引擎默认值；连奏填满默认关闭

    def engine_defaults(engine: str) -> dict:
        d = poly.PolyParams().to_dict()
        if engine == "melody":
            d.update(MELODY_POLY_DEFAULTS)
        return d

    pyin_cache: dict[tuple, tuple] = {}

    def pyin_result(pid: str, fmin=65.0, fmax=1400.0):
        """pYIN 音高跟踪较慢（3.7 分钟约 30 秒），按工程缓存，供单音 / 单旋律引擎复用。"""
        key = (pid, fmin, fmax)
        if key not in pyin_cache:
            if len(pyin_cache) > 2:
                pyin_cache.clear()
            pyin_cache[key] = analysis.run_pyin(get_audio(pid), fmin=fmin, fmax=fmax)
        return pyin_cache[key]

    def mono_result(pid: str, fmin=65.0, fmax=1400.0, min_note=0.07, voiced_min=0.25):
        key = (pid, fmin, fmax, min_note, voiced_min)
        if key not in mono_cache:
            if len(mono_cache) > 4:
                mono_cache.clear()
            mono_cache[key] = analysis.transcribe_mono(get_audio(pid), fmin=fmin, fmax=fmax, min_note=min_note,
                                                       voiced_prob_min=voiced_min, pyin=pyin_result(pid, fmin, fmax))
        return mono_cache[key]

    def melody_notes(pid: str, poly_params: dict | None):
        """单旋律（人声 / 独奏）：复音模型主干 + pYIN 补漏 / 去矛盾 + 沿 pYIN 延伸（melody.py）。"""
        params = engine_defaults("melody") | (poly_params or {})
        notes, st, _ = melody.transcribe(get_audio(pid), poly_output(pid), pyin_result(pid), params)
        return notes, {"raw": st["poly_total"], "after_merge": st["poly_kept"], "final": len(notes),
                       "poly_kept": st["poly_kept"], "poly_dropped": st["poly_dropped"],
                       "pyin_added": st["pyin_added"]}

    def tuning(pid: str) -> float:
        if pid not in tuning_cache:
            tuning_cache[pid] = analysis.estimate_tuning(get_audio(pid))
        return tuning_cache[pid]

    def poly_output(pid: str) -> dict:
        """复音模型输出：内存 → 磁盘缓存 → 运行模型。"""
        if pid in poly_cache:
            return poly_cache[pid]
        npz = WORK / pid / "basicpitch.npz"
        out = poly.load_output(npz)
        if out is None:
            out = poly.infer(str(audio_path(pid)))
            poly.save_output(out, npz)
        poly_cache.clear()
        poly_cache[pid] = out
        return out

    def activation_payload(out: dict) -> dict:
        """把“正在响”的概率图压缩成 uint8 + base64，前端画成热度图（每 2 帧取最大值）。"""
        import base64
        a = out["note"]
        if len(a) % 2:
            a = np.vstack([a, a[-1:]])
        a = a.reshape(-1, 2, a.shape[1]).max(axis=1)
        q = np.clip(a * 255, 0, 255).astype(np.uint8)
        T = poly.frame_times(len(out["note"]))
        fps_eff = (len(T) - 1) / max(1e-6, float(T[-1] - T[0]))   # 线性近似（误差 < 10 ms）
        return {"fps": fps_eff / 2, "t0": float(T[0]), "lo": poly.MIDI_OFFSET, "bins": int(q.shape[1]),
                "frames": int(q.shape[0]), "data": base64.b64encode(q.tobytes()).decode()}

    def number(notes):
        for i, n in enumerate(notes):
            n["id"] = i + 1
            n["start"] = round(n["start"], 4)
            n["end"] = round(n["end"], 4)
        return notes

    @app.post("/api/analyze")
    def analyze(req: AnalyzeReq):
        def run():
            t0 = time.time()
            y = get_audio(req.id)
            curve, activation, polyphony, stats = None, None, None, {}
            mode = req.mode
            if mode in ("poly", "melody") and not poly.available():
                raise HTTPException(400, "复音模型不可用（缺少 basic-pitch / onnxruntime）")
            if mode == "auto":
                if poly.available():
                    # 先跑复音模型，用“同时有两个非泛音关系的音在响”的比例判断是不是单旋律
                    polyphony = round(poly.monophony_ratio(poly_output(req.id)), 3)
                    mode = "melody" if polyphony < MELODY_RATIO else "poly"
                else:
                    mode = "mono"
            if mode in ("poly", "melody"):
                out = poly_output(req.id)
                activation = activation_payload(out)
            if mode == "mono":
                mono_notes, curve = mono_result(req.id, req.fmin, req.fmax, req.min_note)
            elif mode == "melody":
                _, curve = mono_result(req.id, voiced_min=MELODY_VOICED_MIN)
            if mode == "poly":
                notes = poly.decode(out, poly.PolyParams.from_dict(engine_defaults("poly") | (req.poly_params or {})), stats)
                analysis.refine_cents(y, analysis.SR, notes)   # 复音：用频谱峰值测精确音分
            elif mode == "melody":
                notes, stats = melody_notes(req.id, req.poly_params)
            else:
                notes = [dict(n) for n in mono_notes]
            grid = analysis.estimate_grid(y, beats_per_bar=req.beats_per_bar,
                                          note_onsets=[n["start"] for n in notes])
            number(notes)
            key, chords = harmony.analyze_harmony(notes, grid.bpm, grid.beats_per_bar, grid.origin,
                                                  audio_chroma(req.id))
            return {"notes": notes, "grid": grid.to_dict(), "pitch_curve": curve,
                    "activation": activation, "engine": mode, "polyphony": polyphony, "stats": stats,
                    "engine_label": {"poly": "复音引擎", "melody": "单旋律引擎（复音模型 + pYIN 补漏）",
                                     "mono": "单音引擎（pYIN）"}[mode],
                    "poly_params": engine_defaults(mode) | (req.poly_params or {}),
                    "poly_defaults": engine_defaults(mode),
                    "key": key, "chords": chords, "tuning_cents": round(tuning(req.id), 1),
                    "duration": len(y) / analysis.SR, "elapsed": round(time.time() - t0, 2)}
        return guard(run)

    class RetranscribeReq(BaseModel):
        id: str
        poly_params: dict | None = None
        engine: str = "poly"

    @app.post("/api/retranscribe")
    def retranscribe(req: RetranscribeReq):
        """只用新参数重新解码复音模型输出（模型不重跑，通常 < 1 秒）。"""
        def run():
            t0 = time.time()
            stats = {}
            if req.engine == "melody":
                notes, stats = melody_notes(req.id, req.poly_params)
            else:
                notes = poly.decode(poly_output(req.id), poly.PolyParams.from_dict(req.poly_params), stats)
                analysis.refine_cents(get_audio(req.id), analysis.SR, notes)
            return {"notes": number(notes), "stats": stats, "elapsed": round(time.time() - t0, 2)}
        return guard(run)

    class HarmonyReq(BaseModel):
        notes: list[dict]
        bpm: float
        beats_per_bar: int = 4
        origin: float = 0.0
        id: str | None = None

    @app.post("/api/harmony")
    def harmony_api(req: HarmonyReq):
        def run():
            chroma = audio_chroma(req.id) if req.id else None
            key, chords = harmony.analyze_harmony(req.notes, req.bpm, req.beats_per_bar, req.origin, chroma)
            return {"key": key, "chords": chords}
        return guard(run)

    @app.get("/api/capabilities")
    def capabilities():
        return {"poly": poly.available(), "poly_defaults": poly.PolyParams().to_dict()}

    # ------------------------------------------------------------------ 插件
    @app.get("/api/plugins")
    def plugins():
        return {"plugins": [{"name": "内置合成器", "path": BUILTIN}] + scan_vst3(),
                "current": host.info()}

    class LoadReq(BaseModel):
        path: str
        plugin_name: str | None = None

    @app.post("/api/plugins/load")
    def load(req: LoadReq):
        ref_cache.clear()
        return guard(host.load, req.path, req.plugin_name)

    @app.post("/api/plugins/editor")
    def editor():
        guard(host.open_editor)
        return host.info()

    @app.get("/api/plugins/presets")
    def plugin_presets():
        return {"presets": guard(host.presets), "current": host.info().get("preset")}

    class PresetReq(BaseModel):
        name: str | None = None

    @app.post("/api/plugins/preset")
    def plugin_preset(req: PresetReq):
        ref_cache.clear()
        return guard(host.set_preset, req.name)

    @app.get("/api/plugins/current")
    def current():
        return host.info()

    # ------------------------------------------------------------------ 渲染
    class NoteReq(BaseModel):
        pitch: int
        velocity: int = 100
        duration: float = 0.8

    @app.post("/api/render/note")
    def render_note(req: NoteReq):
        dur = float(np.clip(req.duration, 0.05, 8.0))
        audio = guard(host.render, [(req.pitch, req.velocity, 0.0, dur)], dur + 1.2)
        return wav_response(audio)

    class SeqReq(BaseModel):
        notes: list[dict]
        tail: float = 2.5

    @app.post("/api/render/sequence")
    def render_sequence(req: SeqReq):
        if not req.notes:
            raise HTTPException(400, "没有音符")
        events = [(n["pitch"], n.get("velocity", 100), n["start"], n["end"]) for n in req.notes]
        duration = max(e for *_, e in events) + req.tail
        audio = guard(host.render, events, duration)
        return wav_response(audio)

    class BankReq(BaseModel):
        pitches: list[int]
        note: float = 1.5          # 每个音按住的时长
        spacing: float = 2.2       # 相邻两个音的间隔（含释音）
        velocity: int = 100

    @app.post("/api/render/bank")
    def render_bank(req: BankReq):
        """音色库：把多个音高排在一条轨上一次渲染（单击试听用）。
        第 i 个音高从 i × spacing 秒开始，前端按此切片缓存，之后单击音符即时发声。"""
        ps = [int(p) for p in req.pitches if 21 <= int(p) <= 108][:120]
        if not ps:
            raise HTTPException(400, "没有音高")
        events = [(p, req.velocity, i * req.spacing, i * req.spacing + req.note) for i, p in enumerate(ps)]
        audio = guard(host.render, events, len(ps) * req.spacing + 0.3)
        r = wav_response(audio)
        r.headers["X-Bank-Pitches"] = ",".join(map(str, ps))
        r.headers["X-Bank-Spacing"] = str(req.spacing)
        return r

    # ------------------------------------------------------------------ 校验
    class CheckReq(BaseModel):
        id: str
        notes: list[dict]                    # 本次要校验的音符
        context: list[dict] | None = None    # 工程里全部音符（找“同时发声的其他音”用）
        cents_tolerance: float = 30.0

    def ensure_refs(pitches) -> None:
        """把还没缓存的参考音一次性批量渲染（命令行渲染器每次调用都有固定开销）。"""
        missing = sorted({p for p in pitches if (host.state_version, p) not in ref_cache})
        for p, audio in host.render_batch(missing).items():
            mono = librosa.resample(audio.mean(axis=0), orig_sr=RENDER_SR, target_sr=analysis.SR)
            ref_cache[(host.state_version, p)] = checker.semitone_spectrum(mono[int(0.03 * analysis.SR):], analysis.SR)

    @app.post("/api/check")
    def check(req: CheckReq):
        def run():
            y = get_audio(req.id)
            context = req.context or req.notes
            # 与待校验音符同时发声的其他音也需要模板
            span = [(n["start"], n["end"]) for n in req.notes]
            near = [m for m in context if any(min(e, m["end"]) > max(s, m["start"]) for s, e in span)]
            ensure_refs(checker.needed_pitches(req.notes) | {int(m["pitch"]) for m in near})
            templates = {p: v for (ver, p), v in ref_cache.items() if ver == host.state_version}
            tun = tuning(req.id)
            # 音准按“相对整体调音”判断：整首歌按 A4=442Hz 调的话，每个音 +8 音分不算跑调
            rel = [dict(n, cents=float(n.get("cents") or 0.0) - tun) for n in req.notes]
            results = checker.check_notes(y, analysis.SR, rel, near, templates,
                                          cents_tolerance=req.cents_tolerance)
            return {"results": results, "plugin": host.plugin_name}
        return guard(run)

    # ------------------------------------------------------------------ 导出
    class ExportReq(BaseModel):
        notes: list[dict]
        bpm: float
        beats_per_bar: int = 4
        origin: float = 0.0
        align: str = "bar"
        name: str = "OptMusic"

    @app.post("/api/export/midi")
    def export(req: ExportReq):
        data = export_midi(req.notes, req.bpm, req.beats_per_bar, req.origin, req.align, req.name)
        return Response(data, media_type="audio/midi")

    @app.exception_handler(HTTPException)
    async def http_err(_, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.middleware("http")
    async def no_cache(request, call_next):
        # 界面文件不缓存：程序更新后 Electron / 浏览器不会继续用旧的 app.js
        resp = await call_next(request)
        if not request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
    return app
