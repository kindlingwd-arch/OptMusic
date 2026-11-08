"""OptMusic 启动入口：Web 服务在后台线程，VST 宿主占用主线程。"""
import argparse
import threading
import webbrowser

import uvicorn

from optmusic.host import PluginHost
from optmusic.server import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    host = PluginHost()
    app = create_app(host)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning"))
    stop = threading.Event()

    def serve():
        server.run()
        stop.set()

    threading.Thread(target=serve, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"OptMusic 已启动：{url}  （Ctrl+C 退出）")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        host.run_forever(stop)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True


if __name__ == "__main__":
    main()
