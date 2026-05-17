"""测试删除记录功能"""
from db.storage_service import StorageService


def test_delete_record():
    """测试删除记录"""
    storage = StorageService()
    records = storage.list_records()

    if not records:
        print("当前没有记录，无法删除。")
        return

    record_id = records[0]["id"]
    ok = storage.delete_record(record_id)

    assert ok
    print("删除结果：", ok)
    print("删除后剩余记录：")
    for record in storage.list_records():
        print(record)
