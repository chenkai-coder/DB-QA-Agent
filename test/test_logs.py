"""测试日志查询功能"""
from db.storage_service import StorageService


def test_logs():
    """测试查询日志"""
    storage = StorageService()

    print("=== 查询日志 ===")
    logs = storage.list_logs()
    assert isinstance(logs, list)
    for log in logs:
        print(log)
