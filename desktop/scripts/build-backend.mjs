// 用仓库的 .venv 运行 PyInstaller，输出到 <repo>/build/backend/optmusic-backend
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const py = path.join(root, ".venv", "Scripts", "python.exe");
if (!existsSync(py)) {
  console.error(`找不到 ${py}，请先按 README 创建 Python 虚拟环境`);
  process.exit(1);
}
const r = spawnSync(py, [
  "-m", "PyInstaller", "--noconfirm", "--clean",
  "--distpath", path.join(root, "build", "backend"),
  "--workpath", path.join(root, "build", "pyinstaller"),
  path.join(root, "packaging", "optmusic-backend.spec"),
], { cwd: root, stdio: "inherit" });
process.exit(r.status ?? 1);
