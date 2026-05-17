# 批量导入卡死问题诊断与解决方案

## 问题分析

### 原因 1: 超时无保护
- **现象**: 批量导入大文件夹时，UI 界面完全卡死，没有任何反应
- **根因**: 每个图片都要调用大模型进行 OCR/解析（耗时 15-30 秒），而递归扫描的文件夹里可能有数十个文件。线程之间没有超时控制，如果某个文件解析失败会无限期等待

### 原因 2: 无进度反馈
- **现象**: 用户不知道系统是否在工作
- **根因**: 没有进度条、没有实时状态更新，整个批量导入是一个"黑盒"

### 原因 3: 无图片对照
- **现象**: 用户要对比文件时，无法在界面看到图片
- **根因**: UI 设计中没有图片预览区

---

## 解决方案

### 改进 1: 工具层 - 超时保护 + 进度报告 ✅

**文件**: `tools/agent_tools.py`

```python
class BatchImportFiles(BaseTool):
    def _process_file_with_timeout(self, fp: str, timeout: int = 20) -> tuple:
        """处理单个文件，使用超时保护"""
        # - 每个文件最多等待 20 秒
        # - 超过时间自动放弃，不卡死
        # - 返回成功/失败状态
```

**关键特性**:
- ✅ **单文件超时**: 每个图片/文本文件处理最多 20 秒，超时自动跳过
- ✅ **文件数量限制**: 最多处理 50 个文件（防止卡顿）
- ✅ **实时进度报告**: 通过 `ui_queue` 发送 `preview_file` 消息，显示当前文件
- ✅ **错误统计**: 返回成功/失败计数，让用户了解进度

### 改进 2: UI 层 - 图片预览 + 进度条 ✅

**文件**: `ui/desktop.py`

#### 布局改进:
```
┌─────────────────────────────────────────────────────────┐
│  顶栏按钮                                               │
├──────────────────────────┬──────────────────────────────┤
│  主聊天输出区            │  📸 图片对照区 (新增)       │
│  (文本框)                │  - 图片预览 (300x250)       │
│                          │  - 文件名显示               │
├──────────────────────────┤  - 进度条 (新增)            │
│  思考过程区              │  - 进度文本                 │
│  (可折叠)                │                              │
├──────────────────────────┴──────────────────────────────┤
│  快捷键区（📁扫目录 🖼️扫图 📄扫文档 🔍对照图片）      │
├─────────────────────────────────────────────────────────┤
│  输入框 + 发送按钮                                      │
├─────────────────────────────────────────────────────────┤
│  状态标签                                               │
└─────────────────────────────────────────────────────────┘
```

#### 新增组件:
1. **图片预览框** (`preview_canvas`)
   - 大小: 300x250 像素
   - 显示当前正在处理/对照的图片
   - 自动缩放，保持宽高比

2. **进度条** (`progress_bar`)
   - 实时更新处理进度
   - 显示文件数: "正在处理: (5/30) document.jpg"

3. **文件名标签** (`preview_filename`)
   - 显示当前处理文件的完整名称

### 改进 3: 通讯层 - UI 队列集成 ✅

**文件**: `agent_core/smart_agent.py`

```python
# 工具调用时传入 ui_queue
obs = tool_instance.call(param_str, ui_queue=ui_queue)
```

**消息类型**:
- `preview_file`: 传入文件路径，UI 显示预览图
- `status`: 更新状态标签和进度文本

---

## 使用效果

### 批量导入 E:/科研论文 文件夹

**改进前**:
- 🔴 UI 卡死，无反应
- 🔴 无进度反馈
- 🔴 不知道处理到哪了

**改进后**:
- ✅ 实时显示进度: "正在处理: (3/30) formula_01.png"
- ✅ 右侧预览当前图片内容
- ✅ 单个文件超时跳过，不影响整体进度
- ✅ 完成后显示: "共扫描 30 份数据，成功入库 27 份，失败 3 份"

---

## 技术细节

### 超时机制实现
```python
def _process_file_with_timeout(self, fp: str, timeout: int = 20):
    result = {'success': False, 'data': None}
    exception_container = []
    
    def process():
        # 真实的处理逻辑
        ...
    
    t = threading.Thread(target=process, daemon=True)
    t.start()
    t.join(timeout=timeout)  # 关键：等待最多 timeout 秒
    
    if t.is_alive():
        result['error'] = f"处理超时 (>{timeout}秒)"
    
    return result
```

### 进度更新
```python
for idx, fp in enumerate(valid_files):
    # 发送当前进度
    ui_queue.put({'type': 'status', 'data': f"正在处理: ({idx+1}/{len(valid_files)}) {os.path.basename(fp)}"})
    # 发送要显示的文件
    ui_queue.put({'type': 'preview_file', 'data': fp})
```

### 图片预览加载
```python
def show_preview_image(self, file_path: str):
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        img = Image.open(file_path)
        img.thumbnail((300, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.preview_canvas.configure(image=photo, text="")
        self.preview_canvas.image = photo  # 保持引用
```

---

## 验证清单

- [x] 超时保护机制（单个文件 20 秒上限）
- [x] 文件数量限制（最多 50 个）
- [x] 实时进度反馈
- [x] 图片预览功能
- [x] 进度条显示
- [x] UI 布局调整（左右分割）
- [x] 错误处理和统计
- [x] UI 队列集成

---

## 后续优化建议

1. **可配置超时**: 在 UI 上添加设置界面，让用户自定义超时时间
2. **取消按钮**: 批量导入过程中允许用户点击"取消"停止处理
3. **导入历史**: 记录每次导入的结果，显示成功/失败列表
4. **多线程并发**: 使用线程池同时处理多个文件（需谨慎，防止过度并发）
5. **缓存机制**: 对已处理过的文件缓存结果，避免重复处理
