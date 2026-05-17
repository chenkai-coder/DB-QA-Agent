# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

PROJECT_ROOT = Path(SPEC).resolve().parent.parent

datas = collect_data_files("qwen_agent")
binaries = []
hiddenimports = []

for package in ["customtkinter", "ui", "agent_core", "tools", "db", "database"]:
    try:
        d, b, h = collect_all(package)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# matplotlib data files (fonts for Chinese chart rendering)
import matplotlib as _mpl
_mpl_data_path = Path(_mpl.__file__).parent / "mpl-data" / "fonts"
if _mpl_data_path.exists():
    datas.append((str(_mpl_data_path), "matplotlib/mpl-data/fonts"))

hiddenimports += [
    "qwen_agent.agents",
    "qwen_agent.tools.base",
    "soundfile",
    "tkinter",
    "tkinter.filedialog",
    "PIL._tkinter_finder",
    "PIL.Image",
    "PIL.ImageStat",
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "matplotlib.figure",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.font_manager",
    "ui.desktop",
    "agent_core.smart_agent",
    # chromadb embedding and runtime dependencies
    "chromadb.api.rust",
    "chromadb.api.segment",
    "chromadb.telemetry.product.posthog",
    "chromadb.telemetry.product",
    "chromadb.telemetry",
    "chromadb.utils.embedding_functions",
    # embedding backends used by chromadb DefaultEmbeddingFunction
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._ld_preload",
    "tokenizers",
    "numpy",
    # opentelemetry (used by chromadb telemetry)
    "opentelemetry.sdk",
    "opentelemetry.sdk.trace",
    "opentelemetry.exporter.otlp.proto.grpc",
    # dashscope / qwen-agent API
    "dashscope",
    "dashscope.common",
    "dashscope.audio",
    "dashscope.image",
    "dashscope.text",
    # openai compatibility layer
    "openai",
]

# Collect ALL chromadb submodules (heavy but guarantees no missing imports)
try:
    _chroma_all = collect_submodules("chromadb")
    hiddenimports += _chroma_all
except Exception:
    pass

# Collect ALL onnxruntime submodules
try:
    _onnx_all = collect_submodules("onnxruntime")
    hiddenimports += _onnx_all
    _onnx_datas, _onnx_bins, _ = collect_all("onnxruntime")
    datas += _onnx_datas
    binaries += _onnx_bins
except Exception:
    pass

# Collect ALL tokenizers submodules and data files
try:
    _tok_all = collect_submodules("tokenizers")
    hiddenimports += _tok_all
    _tok_datas, _tok_bins, _ = collect_all("tokenizers")
    datas += _tok_datas
    binaries += _tok_bins
except Exception:
    pass

a = Analysis(
    [str(PROJECT_ROOT / "app_main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "qwen_agent.gui",
        "IPython",
        "pytest",
        "sphinx",
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "keras",
        "jax",
        "scipy",
        "sklearn",
        "numba",
        "cv2",
        "plotly",
        "bokeh",
        "selenium",
        "django",
        "skimage",
        "matplotlib.backends.backend_qt",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_qtcairo",
        "matplotlib.backends.qt_compat",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DB-QA-Agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(PROJECT_ROOT / "assets" / "app.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DB-QA-Agent",
)
