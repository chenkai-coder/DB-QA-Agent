import os
import sqlite3
import sys
from typing import Optional


def get_base_path() -> str:
    """
    获取程序基础路径。
    兼容开发环境和 PyInstaller 打包后的运行环境。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_db_path(db_name: str = "app.db") -> str:
    """
    返回数据库文件路径。
    数据库统一放在项目根目录下的 data 文件夹中。
    """
    base_path = get_base_path()
    data_dir = os.path.join(base_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, db_name)


class DatabaseConnection:
    """
    数据库连接管理类。
    每次需要时获取连接，避免长连接带来的复杂问题。
    """

    def __init__(self, db_name: str = "app.db"):
        self.db_path = get_db_path(db_name)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn