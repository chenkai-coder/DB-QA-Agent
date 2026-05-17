# 程序入口：启动桌面 GUI 应用
import os
import sys
import multiprocessing

def _app_base_dir() -> str:
    """获取程序基础目录，兼容开发环境和 PyInstaller 打包路径。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = _app_base_dir()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def main() -> int:
    """启动桌面 GUI 应用，返回退出状态码。"""
    multiprocessing.freeze_support()
    try:
        from ui.desktop import BaseAppUI
        app = BaseAppUI()
        app.mainloop()
        return 0
    except BaseException as exc:
        try:
            import tkinter.messagebox
            tkinter.messagebox.showerror("启动失败", f"程序启动发生错误：\n{exc}")
        except:
            pass
        return 1

if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
