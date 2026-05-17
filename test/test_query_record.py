"""测试查询记录功能"""
import pytest
from db.storage_service import StorageService


@pytest.fixture
def storage():
    return StorageService()


def print_result(result):
    if not result:
        print("未查询到结果。")
        return
    for item in result:
        print(item)


def test_query_by_author(storage):
    """按作者查询"""
    result = storage.query_record({"author": "Yanfeng Zhang"})
    print_result(result)
    # 至少应该有一些记录
    assert isinstance(result, list)


def test_query_by_author_xin_ai(storage):
    """按作者查询：Xin Ai"""
    result = storage.query_record({"author": "Xin Ai"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_author_ge_yu(storage):
    """按作者查询：Ge Yu"""
    result = storage.query_record({"author": "Ge Yu"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_data_type(storage):
    """按数据类型查询"""
    result = storage.query_record({"data_type": "paper"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_source_icde(storage):
    """按来源查询：ICDE"""
    result = storage.query_record({"source": "ICDE"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_source_tkde(storage):
    """按来源查询：TKDE"""
    result = storage.query_record({"source": "TKDE"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_source_sigmod(storage):
    """按来源查询：SIGMOD"""
    result = storage.query_record({"source": "SIGMOD"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_year_2025(storage):
    """按年份查询：2025"""
    result = storage.query_record({"created_date": "2025"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_year_2026(storage):
    """按年份查询：2026"""
    result = storage.query_record({"created_date": "2026"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_category(storage):
    """按分类查询"""
    result = storage.query_record({"category": "academic"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_title_graph(storage):
    """按标题查询：Graph"""
    result = storage.query_record({"title": "Graph"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_title_gnn(storage):
    """按标题查询：GNN"""
    result = storage.query_record({"title": "GNN"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_keyword_gnn(storage):
    """按关键词模糊查询：GNN"""
    result = storage.query_record({"keyword": "GNN"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_keyword_graphrag(storage):
    """按关键词模糊查询：GraphRAG"""
    result = storage.query_record({"keyword": "GraphRAG"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_keyword_transaction(storage):
    """按关键词模糊查询：transaction"""
    result = storage.query_record({"keyword": "transaction"})
    print_result(result)
    assert isinstance(result, list)


def test_query_by_keyword_blockchain(storage):
    """按关键词模糊查询：blockchain"""
    result = storage.query_record({"keyword": "blockchain"})
    print_result(result)
    assert isinstance(result, list)
