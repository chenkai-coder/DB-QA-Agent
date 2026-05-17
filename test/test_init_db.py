"""测试数据库初始化"""
from db.storage_service import StorageService


def test_init_db():
    """测试数据库初始化"""
    storage = StorageService()
    records = storage.list_records()
    assert isinstance(records, list)
    print(f"数据库初始化成功，当前知识记录数：{len(records)}")
