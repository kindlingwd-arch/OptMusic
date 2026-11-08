"""FastAPI 服务：上传 / 分析 / 插件 / 渲染 / 校验 / 导出。"""

import io
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

from . import analysis, harmony
from .host import BUILTIN, RENDER_SR, EditorOpen, PluginHost, scan_vst3
from .midi_export import export_midi

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
WORK = ROOT / "workspace"
WORK.mkdir(exist_ok=True)


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
        mode: str = "mono"             # mono | poly
        beats_per_bar: int | None = None
        fmin: float = 65.0
        fmax: float = 1400.0
        min_note: float = 0.07

    @app.post("/api/analyze")
    def analyze(req: AnalyzeReq):
        def run():
            t0 = time.time()
            y = get_audio(req.id)
            curve = None
            if req.mode == "poly":
                if not analysis.poly_available():
                    raise HTTPException(400, "复音模式需要安装 basic-pitch")
                notes = analysis.transcribe_poly(str(audio_path(req.id)))
            else:
                notes, curve = analysis.transcribe_mono(
                    y, fmin=req.fmin, fmax=req.fmax, min_note=req.min_note)
            grid = analysis.estimate_grid(y, beats_per_bar=req.beats_per_bar,
                                          note_onsets=[n["start"] for n in notes])
            for i, n in enumerate(notes):
                n["id"] = i + 1
                n["start"] = round(n["start"], 4)
                n["end"] = round(n["end"], 4)
            key, chords = harmony.analyze_harmony(notes, grid.bpm, grid.beats_per_bar, grid.origin,
                                                  audio_chroma(req.id))
            return {"notes": notes, "grid": grid.to_dict(), "pitch_curve": curve,
                    "key": key, "chords": chords,
                    "duration": len(y) / analysis.SR, "elapsed": round(time.time() - t0, 2)}
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
        return {"poly": analysis.poly_available()}

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

    # ------------------------------------------------------------------ 校验
    class CheckReq(BaseModel):
        id: str
        notes: list[dict]
        cents_tolerance: float = 30.0

    def ref_profile(pitch: int) -> np.ndarray:
        key = (host.state_version, pitch)
        if key not in ref_cache:
            audio = host.render([(pitch, 100, 0.0, 0.8)], 0.9)
            mono = librosa.resample(audio.mean(axis=0), orig_sr=RENDER_SR, target_sr=analysis.SR)
            ref_cache[key] = analysis.cqt_profile(mono[int(0.03 * analysis.SR):])
        return ref_cache[key]

    @app.post("/api/check")
    def check(req: CheckReq):
        def run():
            y = get_audio(req.id)
            sr = analysis.SR
            results = []
            for n in req.notes:
                p = int(n["pitch"])
                s = int(n["start"] * sr)
                e = int(min(n["end"], n["start"] + 1.0) * sr)
                seg = y[s:e]
                if len(seg) < 512 or np.max(np.abs(seg)) < 1e-4:
                    results.append({"id": n["id"], "status": "silent", "sims": {}})
                    continue
                prof = analysis.cqt_profile(seg)
                cands = [c for c in (p, p - 12, p + 12, p - 1, p + 1) if 21 <= c <= 108]
                sims = {c: analysis.similarity(prof, ref_profile(c)) for c in cands}
                best = max(sims, key=sims.get)
                cents = float(n.get("cents") or 0.0)
                if best != p and sims[best] - sims[p] > 0.03:
                    status = "suspect"
                elif abs(cents) > req.cents_tolerance:
                    status = "offpitch"
                else:
                    status = "ok"
                results.append({"id": n["id"], "status": status, "match": round(sims[p], 3),
                                "suggest": best if status == "suspect" else None,
                                "sims": {str(k): round(v, 3) for k, v in sims.items()}})
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

    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
    return app
