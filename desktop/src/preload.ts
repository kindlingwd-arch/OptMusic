/** 预加载脚本：向网页暴露受限的原生功能（window.optmusic）。 */
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("optmusic", {
  isDesktop: true,
  /** 弹出“另存为”对话框并写入文件；取消时返回 null，否则返回保存路径。 */
  saveFile: (name: string, data: Uint8Array): Promise<string | null> =>
    ipcRenderer.invoke("save-file", name, data),
});
