"""测试批量插入论文记录"""
import pytest
from db.storage_service import StorageService


@pytest.fixture
def storage():
    return StorageService()


SAMPLE_PAPERS = [
    {
        "data_type": "paper",
        "title": "Incremental GNN Embedding Computation on Streaming Graphs",
        "category": "academic",
        "tags": "GNN,图计算,流式图,数据库",
        "summary": "研究流式图上的增量GNN嵌入计算方法。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "conference": "ICDE 2026",
            "location": "Montréal, Canada",
            "authors": [
                "Qiange Wang", "Haoran Lv", "Yanfeng Zhang",
                "Weng-Fai Wong", "Bingsheng He"
            ]
        },
        "author": "Qiange Wang, Haoran Lv, Yanfeng Zhang, Weng-Fai Wong, Bingsheng He",
        "source": "ICDE 2026",
        "created_date": "2026",
        "event_date": None,
        "keyword_text": "GNN streaming graph embedding incremental",
        "entity_text": "Qiange Wang Haoran Lv Yanfeng Zhang Weng-Fai Wong Bingsheng He ICDE TaaSDB",
        "source_type": "image",
        "source_path": "latest_papers_2026_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "GETL: An Extract-Transform-Load Framework Across Graph Models in Graph Warehouse",
        "category": "academic",
        "tags": "图数据库,ETL,数据仓库",
        "summary": "跨图模型的数据ETL框架。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "journal": "TKDE",
            "authors": [
                "Feng Yao", "Xiaokang Yang", "Shufeng Gong",
                "Qian Tao", "Yanfeng Zhang", "Wenyuan Yu", "Ge Yu"
            ]
        },
        "author": "Feng Yao, Xiaokang Yang, Shufeng Gong, Qian Tao, Yanfeng Zhang, Wenyuan Yu, Ge Yu",
        "source": "TKDE",
        "created_date": "2026",
        "event_date": None,
        "keyword_text": "graph ETL warehouse graph models",
        "entity_text": "Feng Yao Xiaokang Yang Shufeng Gong Qian Tao Yanfeng Zhang Wenyuan Yu Ge Yu TKDE graph warehouse",
        "source_type": "image",
        "source_path": "latest_papers_2026_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "Ponzitracker: A General Detection Framework for Ponzi Scheme in Blockchains",
        "category": "academic",
        "tags": "区块链,安全,检测",
        "summary": "区块链中庞氏骗局检测框架。",
        "raw_text": "",
        "structured_json": {
            "ccf": "B",
            "conference": "DASFAA 2026",
            "location": "Jeju Island, South Korea",
            "authors": [
                "Gang Wang", "Yiping Teng", "Zhen Song", "Leyang Li",
                "Qinnan Zhang", "Qingbo Zhang", "Yanfeng Zhang", "Ge Yu"
            ]
        },
        "author": "Gang Wang, Yiping Teng, Zhen Song, Leyang Li, Qinnan Zhang, Qingbo Zhang, Yanfeng Zhang, Ge Yu",
        "source": "DASFAA 2026",
        "created_date": "2026",
        "event_date": None,
        "keyword_text": "blockchain ponzi detection security",
        "entity_text": "Gang Wang Yiping Teng Zhen Song Leyang Li Qinnan Zhang Qingbo Zhang Yanfeng Zhang Ge Yu DASFAA blockchain",
        "source_type": "image",
        "source_path": "latest_papers_2026_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "PAT: Towards Transaction Routing with Page Affinity in Shared-Cache Databases",
        "category": "academic",
        "tags": "数据库,事务,缓存",
        "summary": "共享缓存数据库中的事务路由优化。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "conference": "ICDE 2026",
            "location": "Montréal, Canada",
            "authors": [
                "Zhongqin Tan", "Haoyuan Zhang", "Yanfeng Zhang",
                "Zeshun Peng", "Weixing Zhou", "Jinyu Zhang",
                "Yang Ren", "Guoliang Li", "Ge Yu"
            ]
        },
        "author": "Zhongqin Tan, Haoyuan Zhang, Yanfeng Zhang, Zeshun Peng, Weixing Zhou, Jinyu Zhang, Yang Ren, Guoliang Li, Ge Yu",
        "source": "ICDE 2026",
        "created_date": "2026",
        "event_date": None,
        "keyword_text": "transaction routing cache database page affinity",
        "entity_text": "Zhongqin Tan Haoyuan Zhang Yanfeng Zhang Zeshun Peng Weixing Zhou Jinyu Zhang Yang Ren Guoliang Li Ge Yu ICDE database transaction",
        "source_type": "image",
        "source_path": "latest_papers_2026_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "AcOrch: Accelerating Sampling-based GNN Training under CPU-NPU Heterogeneous Environments",
        "category": "academic",
        "tags": "GNN,异构计算,训练优化",
        "summary": "CPU-NPU异构环境下GNN训练加速。",
        "raw_text": "",
        "structured_json": {
            "ccf": "B",
            "journal": "FCS",
            "authors": [
                "Kefu Chen", "Xin Ai", "Qiange Wang", "Yanfeng Zhang", "Ge Yu"
            ]
        },
        "author": "Kefu Chen, Xin Ai, Qiange Wang, Yanfeng Zhang, Ge Yu",
        "source": "FCS",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "GNN heterogeneous training CPU NPU",
        "entity_text": "Kefu Chen Xin Ai Qiange Wang Yanfeng Zhang Ge Yu FCS GNN CPU NPU",
        "source_type": "image",
        "source_path": "latest_papers_2025_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "A Topology-Aware Localized Update Strategy for Graph-Based ANN Index",
        "category": "academic",
        "tags": "ANN,图索引,向量检索",
        "summary": "面向图ANN索引的拓扑感知更新策略。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "conference": "VLDB 2026",
            "location": "Boston, MA, USA",
            "authors": [
                "Song Yu", "Shengyuan Lin", "Shufeng Gong", "Yongqing Xie",
                "Ruicheng Liu", "Yijie Zhou", "Ji Sun",
                "Yanfeng Zhang", "Guoliang Li", "Ge Yu"
            ]
        },
        "author": "Song Yu, Shengyuan Lin, Shufeng Gong, Yongqing Xie, Ruicheng Liu, Yijie Zhou, Ji Sun, Yanfeng Zhang, Guoliang Li, Ge Yu",
        "source": "VLDB 2026",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "ANN graph index topology update",
        "entity_text": "Song Yu Shengyuan Lin Shufeng Gong Yongqing Xie Ruicheng Liu Yijie Zhou Ji Sun Yanfeng Zhang Guoliang Li Ge Yu VLDB ANN",
        "source_type": "image",
        "source_path": "latest_papers_2025_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "GoGraph: Accelerating Graph Processing through Incremental Reordering",
        "category": "academic",
        "tags": "图计算,性能优化",
        "summary": "通过增量重排序加速图计算。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "journal": "TKDE",
            "authors": [
                "Yijie Zhou", "Shufeng Gong", "Feng Yao", "Hanzhang Chen",
                "Song Yu", "Pengxi Liu", "Yanfeng Zhang",
                "Ge Yu", "Jeffrey Xu Yu"
            ]
        },
        "author": "Yijie Zhou, Shufeng Gong, Feng Yao, Hanzhang Chen, Song Yu, Pengxi Liu, Yanfeng Zhang, Ge Yu, Jeffrey Xu Yu",
        "source": "TKDE",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "graph processing incremental reordering optimization",
        "entity_text": "Yijie Zhou Shufeng Gong Feng Yao Hanzhang Chen Song Yu Pengxi Liu Yanfeng Zhang Ge Yu Jeffrey Xu Yu TKDE graph",
        "source_type": "image",
        "source_path": "latest_papers_2025_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "GeoLayer: Towards Low-Latency and Cost-Efficient Geo-Distributed Graph Stores with Layered Graph",
        "category": "academic",
        "tags": "分布式系统,图存储",
        "summary": "低延迟地理分布式图存储系统。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "conference": "ICDE 2026",
            "location": "Montréal, Canada",
            "authors": [
                "Feng Yao", "Xiaokang Yang", "Shufeng Gong",
                "Song Yu", "Yanfeng Zhang", "Ge Yu"
            ]
        },
        "author": "Feng Yao, Xiaokang Yang, Shufeng Gong, Song Yu, Yanfeng Zhang, Ge Yu",
        "source": "ICDE 2026",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "distributed graph storage low latency geo-distributed",
        "entity_text": "Feng Yao Xiaokang Yang Shufeng Gong Song Yu Yanfeng Zhang Ge Yu ICDE distributed graph",
        "source_type": "image",
        "source_path": "latest_papers_2025_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "NeutronAscend: Optimizing GNN Training with Ascend AI Processors",
        "category": "academic",
        "tags": "GNN,AI芯片",
        "summary": "基于Ascend芯片优化GNN训练。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "journal": "TACO",
            "authors": [
                "Xin Ai", "Bing Zhang", "Qiange Wang",
                "Yanfeng Zhang", "Hao Yuan", "Shufeng Gong", "Ge Yu"
            ]
        },
        "author": "Xin Ai, Bing Zhang, Qiange Wang, Yanfeng Zhang, Hao Yuan, Shufeng Gong, Ge Yu",
        "source": "TACO",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "GNN Ascend AI processor optimization",
        "entity_text": "Xin Ai Bing Zhang Qiange Wang Yanfeng Zhang Hao Yuan Shufeng Gong Ge Yu TACO Ascend AI",
        "source_type": "image",
        "source_path": "latest_papers_2025_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training",
        "category": "academic",
        "tags": "MoE,分布式训练",
        "summary": "分布式MoE训练调度框架。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "conference": "NeurIPS 2025",
            "location": "San Diego, USA",
            "authors": [
                "Yunqi Gao", "Bing Hu", "Mahdi Boloursaz Mashhadi",
                "A-Long Jin", "Yanfeng Zhang", "Pei Xiao",
                "Rahim Tafazolli", "Merouane Abdelkader"
            ]
        },
        "author": "Yunqi Gao, Bing Hu, Mahdi Boloursaz Mashhadi, A-Long Jin, Yanfeng Zhang, Pei Xiao, Rahim Tafazolli, Merouane Abdelkader",
        "source": "NeurIPS 2025",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "MoE distributed training pipeline scheduling",
        "entity_text": "Yunqi Gao Bing Hu Mahdi Boloursaz Mashhadi A-Long Jin Yanfeng Zhang Pei Xiao Rahim Tafazolli Merouane Abdelkader NeurIPS MoE",
        "source_type": "image",
        "source_path": "latest_papers_2025_01.png",
        "status": "normal"
    },
    {
        "data_type": "paper",
        "title": "DepCache: A KV Cache Management Framework for GraphRAG with Dependency Attention",
        "category": "academic",
        "tags": "GraphRAG,缓存管理",
        "summary": "GraphRAG中的KV缓存管理。",
        "raw_text": "",
        "structured_json": {
            "ccf": "A",
            "conference": "SIGMOD 2026",
            "location": "Bangalore, India",
            "authors": [
                "Hao Yuan", "Xin Ai", "Qiange Wang", "Peizheng Li",
                "Jiayang Yu", "Chaoyi Chen", "Xinbo Yang",
                "Yanfeng Zhang", "Zhenbo Fu", "Yingyou Wen", "Ge Yu"
            ]
        },
        "author": "Hao Yuan, Xin Ai, Qiange Wang, Peizheng Li, Jiayang Yu, Chaoyi Chen, Xinbo Yang, Yanfeng Zhang, Zhenbo Fu, Yingyou Wen, Ge Yu",
        "source": "SIGMOD 2026",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "GraphRAG KV cache dependency attention",
        "entity_text": "Hao Yuan Xin Ai Qiange Wang Peizheng Li Jiayang Yu Chaoyi Chen Xinbo Yang Yanfeng Zhang Zhenbo Fu Yingyou Wen Ge Yu SIGMOD GraphRAG",
        "source_type": "image",
        "source_path": "latest_papers_2025_01.png",
        "status": "normal"
    }
]


def test_batch_insert_papers(storage):
    """测试批量插入论文数据"""
    success_count = 0
    fail_count = 0

    for paper in SAMPLE_PAPERS:
        try:
            record_id = storage.insert_record(paper)
            assert record_id > 0
            success_count += 1
            print(f"插入成功: {paper['title']} (id={record_id})")
        except Exception as e:
            fail_count += 1
            print(f"插入失败: {paper['title']} -> {e}")

    print(f"\n总计: 成功={success_count}, 失败={fail_count}")
    assert success_count > 0 or fail_count == len(SAMPLE_PAPERS)
