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


class PluginHost:
    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self.plugin = None
        self.plugin_path: str | None = None
        self.plugin_name: str = BUILTIN
        self.editor_open = False
        self._lock = threading.Lock()  # 保证同一时间只有一个调用方在排队等结果
        self.state_version = 0         # 插件或音色变化时递增，用于前端/缓存失效

    # ---------------- 主线程循环 ----------------
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
                self.plugin, self.plugin_path, self.plugin_name = None, None, BUILTIN
                self.state_version += 1
                return self.info()
            return self.call(_b)

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
            self.plugin, self.plugin_path, self.plugin_name = plugin, path, plugin.name
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
        return {"name": self.plugin_name, "path": self.plugin_path or BUILTIN,
                "is_builtin": self.plugin is None, "editor_open": self.editor_open,
                "version": self.state_version}

    def open_editor(self):
        if self.plugin is None:
            raise RuntimeError("内置合成器没有插件界面")
        if self.editor_open:
            return

        def _show():
            self.editor_open = True
            try:
                self.plugin.show_editor()  # 阻塞直到用户关闭窗口
            finally:
                self.editor_open = False
                self.state_version += 1     # 音色可能被修改
        fut: Future = Future()
        self._jobs.put((_show, fut))

    # ---------------- 渲染 ----------------
    def _render_events(self, events, duration: float) -> np.ndarray:
        """events: [(pitch, velocity, start_sec, end_sec)] → (2, N) float32 @ RENDER_SR"""
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
