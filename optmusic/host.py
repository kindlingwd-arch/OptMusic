"""VST3 插件宿主。

所有插件操作（加载 / 渲染 / 打开插件界面）都在主线程执行：
JUCE 与很多插件的 GUI 要求在主线程上运行。Web 服务跑在后台线程，
通过任务队列把工作交给主线程。
"""
from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np

RENDER_SR = 44100
BUILTIN = "builtin"

VST3_DIRS = [
    r"C:\Program Files\Common Files\VST3",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Common\VST3"),
]


def scan_vst3(extra_dirs: list[str] | None = None) -> list[dict]:
    found = []
    for d in VST3_DIRS + (extra_dirs or []):
        root = Path(d)
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for name in list(dirnames) + filenames:
                if name.lower().endswith(".vst3"):
                    p = Path(dirpath) / name
                    found.append({"name": p.stem, "path": str(p)})
            # 不要深入 .vst3 包内部
            dirnames[:] = [n for n in dirnames if not n.lower().endswith(".vst3")]
    found.sort(key=lambda x: x["name"].lower())
    return found


def inner_binaries(path: str) -> list[str]:
    p = Path(path)
    if not p.is_dir():
        return []
    return [str(x) for x in (p / "Contents" / "x86_64-win").glob("*.vst3")]


class EditorOpen(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 独立程序渲染（Pianoteq）
# ---------------------------------------------------------------------------
# Pianoteq 的 VST3 版本在 pedalboard 里收不到 MIDI，离线渲染完全无声（实测：换缓冲大小、采样率、
# 实时节奏、关闭多核 / CPU 过载检测、空白配置都无效）。但 Pianoteq 独立程序支持命令行导出：
#   Pianoteq.exe --headless --midi in.mid --wav out.wav [--preset 名称]
# 实测时间对齐误差 2 ms，每次调用约 2.6 秒（主要是引擎启动）。

def find_standalone(vst_path: str) -> str | None:
    """若是 Pianoteq，返回对应独立程序 exe 的路径。"""
    stem = Path(vst_path).stem                      # 例：Pianoteq 6 STAGE
    if "pianoteq" not in stem.lower():
        return None
    for base in (r"C:\Program Files\Modartt", r"C:\Program Files (x86)\Modartt"):
        exact = Path(base) / stem / f"{stem}.exe"
        if exact.exists():
            return str(exact)
    for base in (r"C:\Program Files\Modartt", r"C:\Program Files (x86)\Modartt"):
        hits = sorted(Path(base).glob("*/Pianoteq*.exe")) if Path(base).is_dir() else []
        if hits:
            return str(hits[0])
    return None


class StandaloneRenderer:
    NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW

    def __init__(self, exe: str):
        self.exe = exe
        self.name = Path(exe).stem
        self.preset: str | None = None              # None = 使用 Pianoteq 当前（上次保存）的音色
        self._presets: list[str] | None = None
        self.workdir = Path(os.environ.get("TEMP", ".")) / "OptMusic-render"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def _user_prefs(self) -> Path | None:
        """用户自己的 Pianoteq 配置（含授权与当前音色），例：%APPDATA%/Modartt/Pianoteq62 STAGE.prefs。"""
        d = Path(os.environ.get("APPDATA", "")) / "Modartt"
        edition = self.name.split()[-1] if len(self.name.split()) > 2 else ""   # STAGE / Standard / Pro
        cands = [p for p in d.glob("Pianoteq*.prefs") if not edition or p.stem.endswith(edition)]
        return max(cands, key=lambda p: p.stat().st_mtime) if cands else None

    def _run(self, args: list[str], timeout: float):
        """命令行运行 Pianoteq，并保证**不改动用户自己的配置文件**。

        无界面运行会重写 %APPDATA%/Modartt/*.prefs（实测文件从 125 KB 变成 241 KB，音频设备等设置可能被改写）。
        Pianoteq 6.2 的 --prefs 参数在无界面模式下会崩溃（Internal Error frrbmp.cc:294），不能用副本；
        所以运行前备份、运行后若有变化就逐字节还原（连同修改时间）。
        插件界面（独立程序窗口）打开期间渲染会被拒绝，不会和用户自己的修改冲突。"""
        import subprocess
        prefs = self._user_prefs()
        backup = prefs.read_bytes() if prefs and prefs.exists() else None
        st = prefs.stat() if backup is not None else None
        try:
            return subprocess.run([self.exe, *args], capture_output=True, timeout=timeout,
                                  creationflags=self.NO_WINDOW)
        finally:
            if backup is not None:
                try:
                    if prefs.read_bytes() != backup:
                        prefs.write_bytes(backup)
                    if prefs.stat().st_mtime_ns != st.st_mtime_ns:
                        os.utime(prefs, ns=(st.st_atime_ns, st.st_mtime_ns))
                except OSError:
                    pass

    def presets(self) -> list[str]:
        if self._presets is None:
            r = self._run(["--list-presets", "--headless"], timeout=120)
            text = r.stdout.decode("utf-8", "replace")
            self._presets = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return self._presets

    def render(self, events, duration: float) -> np.ndarray:
        import mido
        import soundfile as sf

        n = int(duration * RENDER_SR)
        if not events:
            return np.zeros((2, n), dtype=np.float32)
        # 120 BPM、960 tick/拍 → 1 秒 = 1920 tick
        mid = mido.MidiFile(ticks_per_beat=960)
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=500000))
        ev = []
        for p, v, s, e in events:
            ev.append((round(max(0.0, s) * 1920), 1, mido.Message("note_on", note=int(p) & 0x7F,
                                                                   velocity=int(np.clip(v, 1, 127)))))
            ev.append((round(max(s, e) * 1920), 0, mido.Message("note_off", note=int(p) & 0x7F, velocity=0)))
        ev.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for t, _, m in ev:
            tr.append(m.copy(time=t - last))
            last = t
        # 固定文件名：Pianoteq 的“最近打开的 MIDI”列表里最多只会多出这一项
        mpath, wpath = self.workdir / "OptMusic-render.mid", self.workdir / "OptMusic-render.wav"
        mid.save(str(mpath))
        wpath.unlink(missing_ok=True)
        cmd = [self.exe, "--headless", "--quiet", "--midi", str(mpath), "--wav", str(wpath),
               "--rate", str(RENDER_SR), "--bit-depth", "24"]
        if self.preset:
            cmd += ["--preset", self.preset]
        r = self._run(cmd[1:], timeout=max(120, duration * 3))
        if not wpath.exists():
            raise RuntimeError(f"{self.name} 命令行渲染失败：{r.stderr.decode('utf-8', 'replace')[-300:]}")
        y, sr = sf.read(str(wpath), dtype="float32", always_2d=True)
        y = y.T
        if y.shape[0] == 1:
            y = np.vstack([y, y])
        if sr != RENDER_SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=RENDER_SR)
        out = np.zeros((2, n), dtype=np.float32)
        out[:, : min(n, y.shape[1])] = y[:2, :n]
        return out


class PluginHost:
    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self.plugin = None
        self.cli: StandaloneRenderer | None = None   # Pianoteq 等走独立程序命令行渲染
        self._gui_proc = None
        self.plugin_path: str | None = None
        self.plugin_name: str = BUILTIN
        self.editor_open = False
        self.state_version = 0         # 插件或音色变化时递增，用于前端/缓存失效
        self._editor_close = threading.Event()

    def shutdown(self):
        """退出程序时关闭仍然打开的插件窗口。"""
        self._editor_close.set()

    # ---------------- 宿主线程循环（主线程或专用线程，所有插件操作都在这一个线程上） ----------------
    def run_forever(self, stop: threading.Event):
        while not stop.is_set():
            try:
                fn, fut = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as e:  # noqa: BLE001
                    fut.set_exception(e)

    def call(self, fn, timeout: float = 300):
        if self.editor_open:
            raise EditorOpen("插件界面已打开，请先关闭插件窗口再渲染")
        fut: Future = Future()
        self._jobs.put((fn, fut))
        return fut.result(timeout=timeout)

    # ---------------- 插件管理 ----------------
    def load(self, path: str, plugin_name: str | None = None) -> dict:
        if path == BUILTIN:
            def _b():
                self.plugin, self.cli, self.plugin_path, self.plugin_name = None, None, None, BUILTIN
                self.state_version += 1
                return self.info()
            return self.call(_b)

        exe = find_standalone(path)
        if exe:
            def _load_cli():
                self.plugin, self.cli = None, StandaloneRenderer(exe)
                self.plugin_path, self.plugin_name = path, self.cli.name
                self.state_version += 1
                peak = float(np.abs(self._render_events([(60, 100, 0.0, 0.5)], 1.0)).max())
                warning = None if peak > 1e-4 else f"{self.cli.name} 命令行渲染没有声音，请打开它检查音色/授权"
                return self.info() | {"warning": warning}
            return self.call(_load_cli, timeout=300)

        def _load():
            from pedalboard import VST3Plugin, load_plugin
            last_err = None
            # 很多 .vst3 是“包”目录，pedalboard 直接扫描包会失败，这时改用包内的二进制
            for cand in [path] + inner_binaries(path):
                try:
                    names = VST3Plugin.get_plugin_names_for_file(cand)
                    name = plugin_name if plugin_name in names else (names[0] if names else None)
                    plugin = load_plugin(cand, plugin_name=name) if len(names) > 1 else load_plugin(cand)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
            else:
                raise RuntimeError(f"无法加载插件：{last_err}")
            if not plugin.is_instrument:
                raise RuntimeError(f"{plugin.name} 不是乐器插件（是效果器），无法用 MIDI 发声")
            self.plugin, self.cli, self.plugin_path, self.plugin_name = plugin, None, path, plugin.name
            self.state_version += 1
            # 预热并检测：采样型音源第一次渲染常常较慢；部分插件离线渲染无声
            self._render_events([(60, 100, 0.0, 0.3)], 0.6)
            peak = float(np.abs(self._render_events([(60, 100, 0.0, 0.4)], 0.8)).max())
            warning = None
            if peak < 1e-4:
                warning = (f"{plugin.name} 离线渲染没有声音（可能需要激活授权，或该插件不支持离线渲染/"
                           "需要先在插件界面里加载音色）。可点“插件界面”检查，或换用其他音源。")
            return self.info() | {"names": names, "warning": warning}
        return self.call(_load, timeout=600)

    def info(self) -> dict:
        renderer = "standalone" if self.cli else ("vst" if self.plugin else "builtin")
        return {"name": self.plugin_name, "path": self.plugin_path or BUILTIN,
                "is_builtin": renderer == "builtin", "renderer": renderer,
                "preset": self.cli.preset if self.cli else None,
                "editor_open": self.editor_open, "version": self.state_version}

    def presets(self) -> list[str]:
        return self.call(lambda: self.cli.presets()) if self.cli else []

    def set_preset(self, name: str | None) -> dict:
        if not self.cli:
            raise RuntimeError("当前音源不支持选择预设")
        def _set():
            self.cli.preset = name or None
            self.state_version += 1
            return self.info()
        return self.call(_set)

    def open_editor(self):
        if self.editor_open:
            return
        if self.cli:
            # 打开独立程序界面；用户在里面换琴/调音色，关闭后其设置会被保存，命令行渲染随之生效
            import subprocess
            self.editor_open = True
            self._gui_proc = subprocess.Popen([self.cli.exe])

            def _wait(proc=self._gui_proc):
                proc.wait()
                self.editor_open = False
                self.state_version += 1
            threading.Thread(target=_wait, daemon=True).start()
            return
        if self.plugin is None:
            raise RuntimeError("内置合成器没有插件界面")

        def _show():
            self.editor_open = True
            self._editor_close.clear()
            try:
                self.plugin.show_editor(self._editor_close)  # 阻塞直到用户关闭窗口
            finally:
                self.editor_open = False
                self.state_version += 1     # 音色可能被修改
        fut: Future = Future()
        self._jobs.put((_show, fut))

    # ---------------- 渲染 ----------------
    def _render_events(self, events, duration: float) -> np.ndarray:
        """events: [(pitch, velocity, start_sec, end_sec)] → (2, N) float32 @ RENDER_SR"""
        if self.cli:
            return self.cli.render(events, duration)
        if self.plugin is None:
            return builtin_synth(events, duration)
        msgs = []
        for p, v, s, e in events:
            msgs.append((max(0.0, s), 1, bytes([0x90, int(p) & 0x7F, int(np.clip(v, 1, 127))])))
            msgs.append((max(0.0, e), 0, bytes([0x80, int(p) & 0x7F, 0])))
        msgs.sort(key=lambda m: (m[0], m[1]))  # 同一时刻先 note_off 再 note_on
        # 不使用 reset=True：部分插件（如 Pianoteq）每次 reset 要重新初始化 10 秒以上，且之后输出静音。
        # 改为：发送 延音踏板抬起 + All Sound Off + All Notes Off，再渲染一小段静音把残响冲掉。
        panic = [(bytes([0xB0, 64, 0]), 0.0), (bytes([0xB0, 120, 0]), 0.0), (bytes([0xB0, 123, 0]), 0.0)]
        self.plugin(panic, duration=0.5, sample_rate=RENDER_SR, num_channels=2, reset=False)
        audio = self.plugin([(b, t) for t, _, b in msgs], duration=duration,
                            sample_rate=RENDER_SR, num_channels=2, reset=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        return audio

    def render(self, events, duration: float) -> np.ndarray:
        return self.call(lambda: self._render_events(events, duration))

    def render_batch(self, pitches: list[int], dur: float = 0.6, spacing: float = 1.1) -> dict[int, np.ndarray]:
        """一次渲染多个参考音（逐音校验用）：把音按 spacing 间隔排在一条轨上，渲染一次再切开。
        对命令行渲染（每次调用约 2.6 秒）尤其重要。"""
        if not pitches:
            return {}
        events = [(p, 100, i * spacing, i * spacing + dur) for i, p in enumerate(pitches)]
        audio = self.render(events, len(pitches) * spacing + 0.5)
        seg = int((dur + 0.1) * RENDER_SR)
        return {p: audio[:, int(i * spacing * RENDER_SR): int(i * spacing * RENDER_SR) + seg]
                for i, p in enumerate(pitches)}


def builtin_synth(events, duration: float, sr: int = RENDER_SR) -> np.ndarray:
    """简单的加法合成（无 VST 时的兜底音色）。"""
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float32)
    for p, v, s, e in events:
        f = 440.0 * 2 ** ((p - 69) / 12)
        rel = 0.15
        i0 = int(s * sr)
        i1 = min(n, int((e + rel) * sr))
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0) / sr
        wave = sum((0.6 ** (h - 1)) * np.sin(2 * np.pi * f * h * t)
                   for h in range(1, 7) if f * h < sr / 2)
        env = np.minimum(1.0, t / 0.01) * np.exp(-t * 1.5)
        hold = e - s
        env = np.where(t > hold, env * np.clip(1 - (t - hold) / rel, 0, 1), env)
        out[i0:i1] += (v / 127) * 0.25 * wave * env
    return np.stack([out, out])
