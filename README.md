# DB-QA-Agent

基于私有数据的智能问答 Agent，通过桌面 GUI 界面实现对本地知识库的自然语言问答。

## 功能特性

- **文档导入**：支持图片（PNG/JPG/JPEG/BMP/WebP）、文本（TXT/MD）等格式文件上传
- **向量检索**：使用 ChromaDB 构建本地向量索引，实现语义级文档检索
- **智能问答**：基于 Qwen-Agent 框架，调用通义千问大模型进行自然语言问答
- **对话管理**：支持多轮会话，上下文连续
- **知识统计**：按作者、来源等维度生成数据可视化图表（matplotlib）
- **图片解析**：视觉大模型 OCR 提取图片内容，自动结构化入库
- **独立运行**：可通过 PyInstaller 打包为独立 exe，无需安装 Python 环境

## 快速开始

### 环境要求

- Python 3.10+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动程序

```bash
python app_main.py
```

### 运行测试

```bash
pytest test/
```

### 打包为 exe

```powershell
powershell -ExecutionPolicy Bypass -File build/build_app.ps1
```

## 项目结构

```
DB-QA-Agent/
├── app_main.py                    # 程序入口，启动 GUI
├── requirements.txt               # Python 依赖
├── README.md                      # 项目说明文档
├── .gitignore                     # Git 忽略规则
├── .gitattributes                 # Git 属性配置
│
├── agent_core/                    # 智能问答核心引擎
│   ├── smart_agent.py             # Agent 主逻辑（意图识别、工具调度、ReAct 循环）
│   └── agent_helpers.py           # 提示词清洗与格式化辅助
├── agent/                         # Agent 控制器
│   ├── __init__.py
│   └── agent_controller.py        # 统一 Agent 分发接口
├── tools/                         # 工具集（供 Agent 调用）
│   └── agent_tools.py             # 图片解析、批量导入、知识检索、语义搜索、统计分析等（13 个工具）
│
├── db/                            # 数据库层（SQLite）
│   ├── __init__.py
│   ├── connection.py              # 数据库连接管理
│   ├── schema.py                  # 数据表定义（knowledge_records / sessions / ingest_logs）
│   ├── knowledge_records_repository.py  # 知识记录增删改查
│   ├── log_repository.py          # 日志管理
│   ├── session_repository.py      # 会话管理
│   └── storage_service.py         # 统一业务层接口（组合三个仓库）
├── database/                      # 向量数据库
│   ├── vector_index.py            # ChromaDB 向量索引服务
│   └── vector_memory.py           # 历史问答向量记忆
│
├── ui/                            # 桌面 GUI 界面
│   └── desktop.py                 # CustomTkinter 主窗口（聊天、预览、进度展示）
├── test/                          # 单元测试（pytest，共 18 个测试）
│   ├── __init__.py
│   ├── test_agent.py              # Agent 控制器基本功能测试
│   ├── test_agent_delete.py       # 删除意图测试
│   ├── test_agent_detail.py       # 详情查询测试
│   ├── test_agent_insert.py       # 插入意图测试
│   ├── test_agent_invalid_intent.py # 非法意图测试
│   ├── test_agent_mock_input.py   # 模拟输入测试
│   ├── test_agent_query.py        # 查询意图测试
│   ├── test_agent_summarize.py    # 总结意图测试
│   ├── test_agent_update.py       # 更新意图测试
│   ├── test_db.py                 # 数据库基础测试
│   ├── test_delete_record.py      # 记录删除测试
│   ├── test_init_db.py            # 数据库初始化测试
│   ├── test_insert_record.py      # 记录插入测试
│   ├── test_list_records.py       # 记录列表测试
│   ├── test_logs.py               # 日志功能测试
│   ├── test_query_record.py       # 记录查询测试
│   ├── test_session.py            # 会话管理测试
│   └── test_update_record.py      # 记录更新测试
│
├── build/                         # PyInstaller 构建脚本
│   ├── build_app.ps1              # exe 打包脚本
│   ├── build_setup.ps1            # 安装包制作脚本
│   ├── DB_QA_Agent.spec           # PyInstaller 配置文件
│   ├── installer_install.cmd      # 安装脚本（CMD）
│   └── installer_install.ps1      # 安装脚本（PowerShell）
├── dist/                          # 构建产物
│   └── DB-QA-Agent/               # 打包后的 exe 程序，双击即可运行，无需 Python 环境
├── data/                          # 数据目录（运行时自动生成）
│   ├── app.db                     # SQLite 数据库文件
│   ├── chroma_index/              # ChromaDB 向量索引数据
│   └── analysis_charts/           # 统计分析图表输出
├── assets/                        # 资源文件
│   └── app.ico                    # 应用图标
├── setup/                         # Windows 安装包
│   └── DB-QA-Agent-Setup.exe      # 安装程序
└── IMPROVEMENTS.md                # 项目改进记录
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 大模型 | 通义千问（Qwen） |
| Agent 框架 | Qwen-Agent |
| GUI | CustomTkinter |
| 向量数据库 | ChromaDB |
| 关系数据库 | SQLite |
| 数据可视化 | Matplotlib |
| 文档解析 | Pillow + qwen-agent DocParser |
| 测试框架 | pytest |
| 打包工具 | PyInstaller |

## Agent 工具列表

| 工具名 | 功能 |
|--------|------|
| ParseAndInsertImage | 单张图片 OCR 解析并入库 |
| BatchImportFiles | 批量扫描文件夹导入图文数据 |
| ParseAndInsertTextFile | 长文本文件解析入库 |
| CompareImageWithRecords | 图片与数据库记录对照对比 |
| SearchKnowledgeRecords | 关键词模糊检索知识库 |
| SemanticSearchKnowledge | 向量语义相似度检索 |
| AnalyzeKnowledgeChart | 按维度统计并生成图表 |
| QueryByAuthorOrTitle | 按作者/标题精确查询 |
| ListAllRecords | 列出所有记录概要 |
| UpdatePaperRecord | 更新记录字段 |
| DeletePaperRecord | 删除单条记录 |
| DeleteRecordsByKeyword | 按关键词批量删除 |
| CountAndGroupStatistics | 按年份/作者分组统计 |
