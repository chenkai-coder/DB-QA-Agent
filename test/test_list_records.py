"""测试列出所有知识记录"""
from db.storage_service import StorageService


def test_list_records():
    """测试列出所有记录"""
    storage = StorageService()
    records = storage.list_records()
    assert isinstance(records, list)
    print(f"共 {len(records)} 条知识记录：")
    for record in records:
        print(record)
