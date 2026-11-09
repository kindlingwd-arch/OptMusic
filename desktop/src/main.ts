/**
 * OptMusic 桌面外壳（Electron 主进程）。
 *
 * 1. 立即显示窗口和“启动中”页面；
 * 2. 启动 Python 后端（音频分析 + VST 宿主 + HTTP API），等待 stdout 打印 `OPTMUSIC_READY <port>`；
 * 3. 窗口加载 http://127.0.0.1:<port>；
 * 4. 退出时关闭后端（关闭 stdin → 后端自行退出，超时则强制结束）。
 */
import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from "electron";
import { ChildProcess, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

const READY_TAG = "OPTMUSIC_READY";
const START_TIMEOUT_MS = 180_000;

let win: BrowserWindow | null = null;
let backend: ChildProcess | null = null;
let backendUrl: string | null = null;
let quitting = false;

// ------------------------------------------------------------------ 后端进程
interface BackendCommand { cmd: string; args: string[]; cwd: string; }

function backendCommand(): BackendCommand {
  const args = ["--port", "0", "--watch-stdin"];
  if (app.isPackaged) {
    // 打包后：resources/backend/optmusic-backend.exe（PyInstaller 产物）
    const dir = path.join(process.resourcesPath, "backend");
    return { cmd: path.join(dir, "optmusic-backend.exe"), args, cwd: dir };
  }
  // 开发：使用仓库根目录的虚拟环境运行 run.py
  const root = path.resolve(__dirname, "..", "..");
  const py = path.join(root, ".venv", "Scripts", "python.exe");
  return { cmd: py, args: [path.join(root, "run.py"), ...args], cwd: root };
}

function logFile(): string {
  const dir = path.join(app.getPath("userData"), "logs");
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, "backend.log");
}

function startBackend(): Promise<string> {
  return new Promise((resolve, reject) => {
    const { cmd, args, cwd } = backendCommand();
    if (!fs.existsSync(cmd)) {
      reject(new Error(`找不到后端程序：${cmd}`));
      return;
    }
    const log = fs.createWriteStream(logFile(), { flags: "w" });
    log.write(`[${new Date().toISOString()}] ${cmd} ${args.join(" ")}\n`);

    const numbaCache = path.join(app.getPath("userData"), "numba-cache");
    const child = spawn(cmd, args, {
      cwd,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1", NUMBA_CACHE_DIR: numbaCache },
    });
    backend = child;

    let settled = false;
    let tail = "";
    const timer = setTimeout(() => {
      if (!settled) { settled = true; reject(new Error("后端启动超时（3 分钟）")); }
    }, START_TIMEOUT_MS);

    let buf = "";
    child.stdout!.setEncoding("utf8");
    child.stdout!.on("data", (chunk: string) => {
      log.write(chunk);
      buf += chunk;
      let i: number;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!settled && line.startsWith(READY_TAG)) {
          settled = true;
          clearTimeout(timer);
          resolve(`http://127.0.0.1:${line.split(/\s+/)[1]}`);
        }
      }
    });
    child.stderr!.setEncoding("utf8");
    child.stderr!.on("data", (chunk: string) => {
      log.write(chunk);
      tail = (tail + chunk).slice(-3000);
    });
    child.on("exit", (code) => {
      log.write(`\n[exit ${code}]\n`);
      backend = null;
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`后端异常退出（代码 ${code}）\n${tail}`));
      } else if (!quitting) {
        showError(`后端进程意外退出（代码 ${code}）。\n日志：${logFile()}\n\n${tail}`);
      }
    });
    child.on("error", (err) => {
      if (!settled) { settled = true; clearTimeout(timer); reject(err); }
    });
  });
}

/** 关闭 stdin 让后端自行退出；3 秒内没退出则强制结束。返回的 Promise 在后端结束后完成。 */
function stopBackend(): Promise<void> {
  const child = backend;
  if (!child || child.exitCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(() => { try { child.kill(); } catch { /* ignore */ } resolve(); }, 3000);
    child.once("exit", () => { clearTimeout(timer); resolve(); });
    try { child.stdin?.end(); } catch { /* ignore */ }
  });
}

// ------------------------------------------------------------------ 窗口
function showError(message: string): void {
  if (!win || win.isDestroyed()) return;
  const js = `window.showError && window.showError(${JSON.stringify(message)})`;
  const loadingUrl = win.webContents.getURL();
  if (loadingUrl.startsWith("file:")) {
    win.webContents.executeJavaScript(js).catch(() => undefined);
  } else {
    win.loadFile(path.join(__dirname, "..", "static", "loading.html")).then(() =>
      win?.webContents.executeJavaScript(js)).catch(() => undefined);
  }
}

function buildMenu(): void {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {
      label: "文件",
      submenu: [
        { label: "打开日志目录", click: () => shell.openPath(path.dirname(logFile())) },
        { type: "separator" },
        { label: "退出", role: "quit" },
      ],
    },
    {
      label: "视图",
      submenu: [
        { label: "重新加载界面", accelerator: "Ctrl+R", click: () => { if (backendUrl) win?.loadURL(backendUrl); } },
        { label: "开发者工具", accelerator: "F12", click: () => win?.webContents.toggleDevTools() },
        { type: "separator" },
        { label: "实际大小", role: "resetZoom" },
        { label: "放大", role: "zoomIn" },
        { label: "缩小", role: "zoomOut" },
        { type: "separator" },
        { label: "全屏", role: "togglefullscreen" },
      ],
    },
  ]));
}

async function createWindow(): Promise<void> {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1000,
    minHeight: 640,
    title: "OptMusic 音乐工作台",
    backgroundColor: "#16181d",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.once("ready-to-show", () => win?.show());
  win.on("closed", () => { win = null; });

  // 只允许加载本地后端页面；外部链接用系统浏览器打开
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e, url) => {
    if (backendUrl && !url.startsWith(backendUrl)) e.preventDefault();
  });

  await win.loadFile(path.join(__dirname, "..", "static", "loading.html"));

  const t0 = Date.now();
  const ticker = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    win?.webContents.executeJavaScript(`window.setProgress && window.setProgress(${s})`).catch(() => undefined);
  }, 1000);
  try {
    backendUrl = await startBackend();
    clearInterval(ticker);
    await win?.loadURL(backendUrl);
  } catch (err) {
    clearInterval(ticker);
    showError(`${(err as Error).message}\n\n日志：${logFile()}`);
  }
}

// ------------------------------------------------------------------ IPC：原生功能
ipcMain.handle("save-file", async (_e, name: string, data: Uint8Array) => {
  if (!win) return null;
  const ext = path.extname(name).slice(1) || "mid";
  const { canceled, filePath } = await dialog.showSaveDialog(win, {
    defaultPath: path.join(app.getPath("documents"), name),
    filters: [{ name: ext.toUpperCase() === "MID" ? "MIDI 文件" : ext.toUpperCase(), extensions: [ext] }],
  });
  if (canceled || !filePath) return null;
  await fs.promises.writeFile(filePath, Buffer.from(data));
  return filePath;
});

// ------------------------------------------------------------------ 生命周期
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) { if (win.isMinimized()) win.restore(); win.focus(); }
  });
  app.whenReady().then(() => { buildMenu(); return createWindow(); });
  // 退出：先等后端退出（最多 3 秒，超时强制结束），再用 app.exit 强制结束主进程。
  // 只用 app.quit() 时，后端子进程的管道句柄会让 Electron 主进程在 quit 事件之后仍然残留。
  let exiting = false;
  const shutdown = (): void => {
    if (exiting) return;
    exiting = true;
    quitting = true;
    stopBackend().finally(() => {
      app.exit(0);
      // 兜底：实测偶尔 app.exit 之后主进程仍残留（无窗口），3 秒后强制结束
      setTimeout(() => process.exit(0), 3000).unref();
    });
    // 再兜底：后端迟迟不退出时，最多 8 秒后整体强制结束
    setTimeout(() => { try { backend?.kill(); } catch { /* ignore */ } process.exit(0); }, 8000).unref();
  };
  app.on("before-quit", (e) => { if (!exiting) { e.preventDefault(); shutdown(); } });
  app.on("window-all-closed", shutdown);
}
