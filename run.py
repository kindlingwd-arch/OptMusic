"""OptMusic 后端入口（Python：音频分析 + VST 宿主 + HTTP API）。

- 桌面版由 Electron 外壳（desktop/）启动本进程：
    optmusic-backend --port 0 --watch-stdin
  启动完成后在 stdout 打印一行  OPTMUSIC_READY <port>，Electron 收到后加载界面。
  --watch-stdin：stdin 被关闭（Electron 退出或崩溃）时自动退出，不留残留进程。
- 开发时也可单独运行并用浏览器打开：
    python run.py --open
VST 宿主始终占用主线程（JUCE / 插件界面要求），Web 服务在后台线程。
"""
import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

READY_TAG = "OPTMUSIC_READY"


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("找不到可用端口")


def start_server(host, port: int):
    import uvicorn
    from optmusic.server import create_app

    app = create_app(host)
    # log_config=None：打包后无控制台时默认日志配置会出错
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", log_config=None))
    threading.Thread(target=server.run, daemon=True, name="web-server").start()
    for _ in range(600):
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("Web 服务启动超时")


def warmup():
    """后台预热：触发 numba JIT 编译（首次运行写入缓存），让用户第一次分析更快。"""
    try:
        import numpy as np
        from optmusic import analysis
        sr = analysis.SR
        t = np.arange(int(sr * 1.5)) / sr
        y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        notes, _ = analysis.transcribe_mono(y)
        analysis.estimate_grid(y, note_onsets=[n["start"] for n in notes])
        analysis.cqt_profile(y)
    except Exception as e:  # noqa: BLE001
        print(f"warmup failed: {e}", file=sys.stderr, flush=True)


def watch_stdin(stop: threading.Event):
    """父进程（Electron）退出时 stdin 会到达 EOF。"""
    try:
        while sys.stdin.readline():
            pass
    except Exception:  # noqa: BLE001
        pass
    stop.set()


def main():
    ap = argparse.ArgumentParser(description="OptMusic backend")
    ap.add_argument("--port", type=int, default=8765, help="0 = 自动选择空闲端口")
    ap.add_argument("--open", action="store_true", help="启动后用系统浏览器打开界面")
    ap.add_argument("--watch-stdin", action="store_true", help="stdin 关闭时退出（供 Electron 使用）")
    args = ap.parse_args()

    from optmusic.host import PluginHost

    port = free_port(args.port) if args.port else free_port(0)
    host = PluginHost()
    stop = threading.Event()
    server = start_server(host, port)
    print(f"{READY_TAG} {port}", flush=True)

    threading.Thread(target=warmup, daemon=True, name="warmup").start()
    if args.watch_stdin:
        threading.Thread(target=watch_stdin, args=(stop,), daemon=True).start()
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        host.run_forever(stop)  # 主线程：VST 宿主
    except KeyboardInterrupt:
        pass
    finally:
        host.shutdown()
        server.should_exit = True
    os._exit(0)  # 确保插件残留线程不阻止退出


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
