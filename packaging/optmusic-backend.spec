# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 配置：把 Python 后端（run.py）打包成 optmusic-backend.exe（onedir）。
# 构建：desktop> npm run build:backend   （或 pyinstaller packaging/optmusic-backend.spec）
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

datas = [(str(ROOT / "web"), "web")]
# librosa 通过 lazy_loader 读取 __init__.pyi 存根来延迟导入子模块，必须带上 .pyi 和数据文件
datas += collect_data_files("librosa", include_py_files=False)
datas += collect_data_files("lazy_loader")

hiddenimports = []
hiddenimports += collect_submodules("librosa")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("pedalboard")   # 渲染 MIDI 时动态导入 pedalboard.midi_utils
hiddenimports += collect_submodules("mido")
hiddenimports += ["optmusic.server", "optmusic.analysis", "optmusic.harmony",
                  "optmusic.host", "optmusic.midi_export", "multipart", "python_multipart"]

excludes = ["tkinter", "matplotlib", "IPython", "jupyter", "pytest", "webview", "clr",
            "pythonnet", "PyInstaller", "notebook", "pandas"]

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
    # librosa 里的 @jit(cache=True) 需要能找到 .py 源文件才能写 numba 编译缓存，
    # 否则打包后每次分析都要重新编译（约 60 秒）。
    module_collection_mode={"librosa": "py"},
)
# VC++ 运行库：某些依赖会把旧版 MSVCP140.dll（14.29）带进来，而新版 VST 插件
# 用新版 VC 编译，加载到旧运行库里会直接访问冲突崩溃（0xC0000005）。
# 因此去掉依赖带来的版本，统一使用本机 System32 中的最新版（应用本地部署）。
import os
VC_DLLS = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
           "msvcp140_codecvt_ids.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"]
a.binaries = [b for b in a.binaries if os.path.basename(b[0]).lower() not in VC_DLLS]
sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
for dll in VC_DLLS:
    src = os.path.join(sys32, dll)
    if os.path.exists(src):
        a.binaries.append((dll, src, "BINARY"))

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="optmusic-backend",
    console=True,          # 需要 stdout 与 Electron 通信；Electron 以隐藏窗口方式启动
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="optmusic-backend", upx=False)
