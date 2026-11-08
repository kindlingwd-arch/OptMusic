# OptMusic 音乐工作台

第一版：类似 Melodyne 的「音频 → MIDI」工作台。

- 把 WAV（也支持 FLAC / MP3 / OGG）识别成 MIDI 音符，在钢琴窗中显示音符、实测音高曲线和音准偏差；
- 自动识别 **BPM、拍号、下拍、小节**，以及 **调性 / 调号** 和 **和弦**；
- 调用本机 **VST3 音源** 逐音试听（原声 / VST / A-B / 叠加）并自动校验每个音是否正确（八度错、半音错、音准偏）；
- 可编辑音符（移动、改长度、增删、量化、撤销），导出标准 MIDI 文件。

算法原理见 **[docs/算法原理.md](docs/算法原理.md)**。

## 运行

需要 Python 3.12（推荐用 [uv](https://docs.astral.sh/uv/)）：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

启动（会自动打开浏览器 http://127.0.0.1:8765）：

```bash
.venv\Scripts\python.exe run.py
```

或双击 `start.bat`。

算法自测：

```bash
.venv\Scripts\python.exe -m tests.test_pipeline
```

## 使用流程

1. **打开音频** → 自动识别（单音旋律模式）。
2. 检查 **节奏网格**：标尺上的橙色三角是检测到的原始拍点，可打开 **节拍器** 播放核对；速度不对用 `÷2 / ×2`，小节线不对用 `◀ 一拍 / 一拍 ▶` 或“把选中音符设为小节起点”。
3. 右侧 **音源** 选择 VST3 插件 → 加载（可打开插件界面选音色，关闭窗口后生效）。
4. **校验全部音符**：绿 = 正确，橙 = 音准偏，红 = 疑似错音（可“采纳建议音高”）；用“下一个问题”逐个检查，`1/2/3/4` 键试听原声 / VST / A-B / 叠加。
5. 编辑后 **导出 MIDI**（按小节对齐，或按音频起点对齐）。

### 快捷键

| 操作 | 按键 |
|---|---|
| 播放 / 停止 | 空格 |
| 删除 | Delete |
| 移半音 / 八度 | ↑ ↓ / Shift + ↑ ↓ |
| 按吸附移动 | ← → |
| 量化 | Q |
| 撤销 / 重做 | Ctrl+Z / Ctrl+Y |
| 全选 | Ctrl+A |
| 缩放 / 横向滚动 / 行高 | Ctrl+滚轮 / Shift+滚轮 / Alt+滚轮 |
| 拖动时不吸附 | 按住 Alt |

## 结构

```
run.py                 启动入口（Web 服务在后台线程，VST 宿主占用主线程）
optmusic/analysis.py   音高追踪、音符切分、BPM/下拍/拍号、频谱轮廓
optmusic/harmony.py    调性 / 调号、和弦识别
optmusic/host.py       VST3 扫描、加载、渲染（pedalboard），内置兜底合成器
optmusic/midi_export.py MIDI 导出
optmusic/server.py     HTTP API（FastAPI）
web/                   前端（钢琴窗、播放、校验界面）
docs/算法原理.md        算法原理文档
tests/                 合成音频端到端自测
```

## 已知限制（v0.1）

- 单音模式适合人声、独奏乐器；和声音频需复音模式（basic-pitch，需 Python 3.11 环境）。
- 恒定速度假设，变速歌曲网格会逐渐偏离。
- 只支持 VST3（pedalboard 不支持 VST2）。本机实测：Surge XT、PAPU、BBC Symphony Orchestra 可用；Pianoteq 6 STAGE 离线渲染无声。
