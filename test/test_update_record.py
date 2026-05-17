"""测试更新记录功能"""
from db.storage_service import StorageService


def test_update_record():
    """测试更新记录"""
    storage = StorageService()
    records = storage.list_records()

    if not records:
        print("当前没有记录，无法更新。")
        return

    record_id = records[0]["id"]
    ok = storage.update_record(
        record_id,
        {
            "summary": "已更新的通用知识摘要内容",
            "tags": "数据库,事务,多模型,Agent,已更新"
        }
    )

    assert ok
    print("更新结果：", ok)
    updated = storage.get_record_by_id(record_id)
    print("更新后记录：")
    print(updated)
    assert updated is not None
