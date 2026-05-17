import threading
import queue
import customtkinter as ctk
import tkinter.filedialog as filedialog
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_core.smart_agent import QA_Agent_System
from tools.agent_tools import set_pending_path


class BaseAppUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("基于私有数据的智能问答Agent")
        self.geometry("1200x900")

        self.agent = QA_Agent_System()
        self.ui_queue = queue.Queue()
        self.show_thinking = True
        self.current_preview_image = None

        self.after(50, self.pump_events)

        # ================= 顶栏 =================
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=5)

        self.btn_new_chat = ctk.CTkButton(
            top_frame,
            text="✨ 新建对话",
            width=100,
            command=self.helper_new_chat
        )
        self.btn_new_chat.pack(side="left", padx=5)

        self.btn_toggle_think = ctk.CTkButton(
            top_frame,
            text="👁️ 隐藏思考过程",
            width=120,
            command=self.helper_toggle_think
        )
        self.btn_toggle_think.pack(side="right", padx=5)

        # ================= 主内容区 =================
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        # 左侧：聊天区 + 思考区
        self.left_panel = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.textbox = ctk.CTkTextbox(
            self.left_panel,
            wrap="word",
            font=("Microsoft YaHei", 13)
        )
        self.textbox.pack(fill="both", expand=True, padx=0, pady=0)

        self.think_box = ctk.CTkTextbox(
            self.left_panel,
            wrap="word",
            font=("Consolas", 11),
            fg_color="#1E1E1E",
            text_color="#A9B7C6",
            height=150
        )
        self.think_box.pack(fill="both", expand=False, padx=0, pady=(10, 0))

        # 右侧：预览区，默认隐藏
        self.right_panel = ctk.CTkFrame(
            self.main_container,
            fg_color="#2b2b2b",
            border_width=2,
            border_color="#404040"
        )
        self.right_panel.pack_propagate(False)
        self.right_panel.configure(width=340)

        self.preview_title = ctk.CTkLabel(
            self.right_panel,
            text="📸 图片预览区",
            font=("Microsoft YaHei", 12, "bold")
        )
        self.preview_title.pack(pady=(5, 10))

        # 图片预览控件
        self.preview_canvas = ctk.CTkLabel(
            self.right_panel,
            text="(待加载图片)",
            font=("Microsoft YaHei", 11),
            fg_color="#1E1E1E",
            text_color="#CCCCCC",
            wraplength=300,
            justify="center"
        )
        self.preview_canvas.pack(fill="both", expand=True, pady=(0, 10))

        # 文档预览控件，默认不显示
        self.preview_textbox = ctk.CTkTextbox(
            self.right_panel,
            wrap="word",
            font=("Microsoft YaHei", 12),
            fg_color="#F7F7F7",
            text_color="#222222",
            height=520
        )

        self.preview_filename = ctk.CTkLabel(
            self.right_panel,
            text="",
            font=("Microsoft YaHei", 10),
            text_color="#AAAAAA",
            wraplength=300
        )
        self.preview_filename.pack(pady=(0, 5))

        self.progress_bar = ctk.CTkProgressBar(self.right_panel, height=8)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 5))

        self.progress_text = ctk.CTkLabel(
            self.right_panel,
            text="",
            font=("Microsoft YaHei", 9),
            text_color="#CCCCCC"
        )
        self.progress_text.pack()

        self.hide_right_preview_panel()

        # ================= 快捷按钮区 =================
        self.cmd_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cmd_frame.pack(fill="x", padx=10, pady=(0, 5))

        self.btn_insert_dir = ctk.CTkButton(
            self.cmd_frame,
            text="📁 扫目录",
            fg_color="#345B50",
            hover_color="#244238",
            command=self.helper_insert_folder,
            width=100
        )
        self.btn_insert_dir.pack(side="left", padx=5)

        self.btn_insert_img = ctk.CTkButton(
            self.cmd_frame,
            text="🖼️ 扫图",
            command=self.helper_insert_image,
            width=100
        )
        self.btn_insert_img.pack(side="left", padx=5)

        self.btn_insert_txt = ctk.CTkButton(
            self.cmd_frame,
            text="📄 扫文档",
            fg_color="#A24A40",
            hover_color="#7A3932",
            command=self.helper_insert_text,
            width=100
        )
        self.btn_insert_txt.pack(side="left", padx=5)

        self.btn_compare_img = ctk.CTkButton(
            self.cmd_frame,
            text="🔍 对照图片",
            fg_color="#5A4A8A",
            hover_color="#46396B",
            command=self.helper_compare_image,
            width=100
        )
        self.btn_compare_img.pack(side="left", padx=5)

        # ================= 多行输入区 =================
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.entry = ctk.CTkTextbox(
            self.input_frame,
            wrap="word",
            height=90,
            font=("Microsoft YaHei", 13)
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.input_placeholder = "例如：列出知识库中所有发表在 ICDE 上的论文标题和作者……"
        self.entry.insert("1.0", self.input_placeholder)
        self.entry.bind("<FocusIn>", self.clear_input_placeholder)
        self.entry.bind("<Control-Return>", self.send_req)

        self.btn_send = ctk.CTkButton(
            self.input_frame,
            text="发送给大脑\nCtrl+Enter",
            command=self.send_req,
            width=100,
            height=90
        )
        self.btn_send.pack(side="right", fill="y")

        # ================= 状态栏 =================
        self.status = ctk.CTkLabel(
            self,
            text="当前 AI 状态: 闲置就绪 (指令等待中...)"
        )
        self.status.pack(side="left", padx=10, pady=5)

    # ================= 右侧预览区控制 =================

    def show_right_preview_panel(self):
        if not self.right_panel.winfo_ismapped():
            self.right_panel.pack(
                side="right",
                fill="y",
                padx=0,
                pady=0,
                ipadx=10,
                ipady=10
            )

    def hide_right_preview_panel(self):
        if self.right_panel.winfo_ismapped():
            self.right_panel.pack_forget()

    # ================= 输入框辅助 =================

    def clear_input_placeholder(self, event=None):
        current_text = self.entry.get("1.0", "end").strip()
        if current_text == self.input_placeholder:
            self.entry.delete("1.0", "end")

    def set_input_text(self, text: str):
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", text)

    # ================= 顶栏按钮逻辑 =================

    def helper_new_chat(self):
        self.agent.global_history = []

        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.insert("end", "✨ [已建立全新记忆空白的回话]\n")
        self.textbox.configure(state="disabled")

        self.think_box.configure(state="normal")
        self.think_box.delete("1.0", "end")
        self.think_box.configure(state="disabled")

        self.current_preview_image = None
        self.preview_canvas.configure(text="(待加载图片)", image=None)
        self.preview_filename.configure(text="")
        self.preview_textbox.configure(state="normal")
        self.preview_textbox.delete("1.0", "end")
        self.preview_textbox.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_text.configure(text="")
        self.hide_right_preview_panel()

        self.set_input_text(self.input_placeholder)

    def helper_toggle_think(self):
        if self.show_thinking:
            self.think_box.pack_forget()
            self.btn_toggle_think.configure(text="👁️ 显示思考过程")
            self.show_thinking = False
        else:
            self.think_box.pack(
                fill="both",
                expand=False,
                padx=0,
                pady=(10, 0),
                after=self.textbox
            )
            self.btn_toggle_think.configure(text="👁️ 隐藏思考过程")
            self.show_thinking = True

    def repack_bottom_frames(self):
        pass

    # ================= 快捷按钮逻辑 =================

    def helper_insert_folder(self):
        path = filedialog.askdirectory(title="选择你想让系统扫描入库的文件夹")
        if path:
            path = path.replace("\\", "/")
            set_pending_path("folder_path", path)
            self.set_input_text("请帮我批量扫描导入当前选中文件夹里的图文数据，完成后只汇报成功数、失败数和关键结果。")

    def helper_insert_image(self):
        filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        path = filedialog.askopenfilename(
            title="指派一张需要入库分析的图像",
            filetypes=filetypes
        )
        if path:
            path = path.replace("\\", "/")
            set_pending_path("image_path", path)
            self.show_preview_image(path)
            self.set_input_text("帮我解析当前选中的图片并录入到库中，入库后只基于图像内容告诉我提取出的标题、作者和摘要。")

    def helper_insert_text(self):
        filetypes = [("Text/Markdown", "*.txt *.md")]
        path = filedialog.askopenfilename(
            title="指派一份需要入库的长文档文本",
            filetypes=filetypes
        )
        if path:
            path = path.replace("\\", "/")
            set_pending_path("file_path", path)
            self.show_preview_text(path)
            self.set_input_text("请读取当前选中的本地文本存入知识库，并只根据文本内容给我一个极简总结。")

    def helper_compare_image(self):
        filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        path = filedialog.askopenfilename(
            title="选择一张需要对照分析的图片",
            filetypes=filetypes
        )
        if path:
            path = path.replace("\\", "/")
            set_pending_path("image_path", path)
            self.show_preview_image(path)
            self.set_input_text("请帮我解析并对照当前选中的图片内容，如果数据库里有相近记录，请只基于内容对比相同点和不同点，用自然语言汇报给我。")

    # ================= 发送与事件循环 =================

    def send_req(self, ev=None):
        txt = self.entry.get("1.0", "end").strip()

        if not txt or txt == self.input_placeholder:
            return "break"

        self.entry.delete("1.0", "end")

        self.textbox.configure(state="normal")
        self.textbox.insert(
            "end",
            f"\n\n=========================\n"
            f"[User 发出复杂指令]: {txt}\n"
            f"=========================\n"
        )
        self.textbox.configure(state="disabled")

        t = threading.Thread(
            target=self.agent.stream_chat_query,
            args=(txt, self.ui_queue)
        )
        t.daemon = True
        t.start()

        return "break"

    def pump_events(self):
        self.textbox.configure(state="normal")
        self.think_box.configure(state="normal")

        while not self.ui_queue.empty():
            msg = self.ui_queue.get()

            if msg["type"] in ("stream_chunk", "think_chunk"):
                self.think_box.insert("end", msg["data"])
                self.think_box.see("end")
                self.think_box.update_idletasks()

            elif msg["type"] == "final_answer":
                self.textbox.insert("end", f"\n[Agent 答复]: {msg['data']}\n")
                self.textbox.see("end")

            elif msg["type"] == "status":
                self.status.configure(text=msg["data"])
                self.status.update_idletasks()

            elif msg["type"] == "preview_file":
                path = msg["data"]
                if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    self.show_preview_image(path)
                elif path.lower().endswith((".txt", ".md")):
                    self.show_preview_text(path)
                self.right_panel.update_idletasks()

            elif msg["type"] == "error":
                self.textbox.insert("end", f"\n[致命异常保护]: {msg['data']}\n")
                self.textbox.see("end")

        self.textbox.configure(state="disabled")
        self.think_box.configure(state="disabled")
        self.after(50, self.pump_events)

    # ================= 右侧预览逻辑 =================

    def show_preview_text(self, file_path: str):
        """在右侧预览框中显示 txt/md 文档内容。"""
        try:
            self.show_right_preview_panel()

            # 文档预览和图片预览互斥
            self.preview_canvas.pack_forget()
            self.preview_textbox.pack(fill="both", expand=True, pady=(0, 10))

            self.current_preview_image = None
            self.preview_title.configure(text="📄 文档预览区")

            if not os.path.exists(file_path):
                self.preview_filename.configure(text="")
                self.preview_textbox.configure(state="normal")
                self.preview_textbox.delete("1.0", "end")
                self.preview_textbox.insert("end", "文件不存在")
                self.preview_textbox.configure(state="disabled")
                return

            filename = os.path.basename(file_path)
            self.preview_filename.configure(text=filename)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if len(content) > 5000:
                content = content[:5000] + "\n\n……文档内容较长，预览仅显示前 5000 字。"

            self.preview_textbox.configure(state="normal")
            self.preview_textbox.delete("1.0", "end")
            self.preview_textbox.insert(
                "end",
                content if content.strip() else "文档为空"
            )
            self.preview_textbox.configure(state="disabled")

        except Exception as e:
            self.show_right_preview_panel()
            self.preview_canvas.pack_forget()
            self.preview_textbox.pack(fill="both", expand=True, pady=(0, 10))

            self.preview_title.configure(text="📄 文档预览区")
            self.preview_filename.configure(text="")
            self.preview_textbox.configure(state="normal")
            self.preview_textbox.delete("1.0", "end")
            self.preview_textbox.insert("end", f"文档预览失败: {str(e)}")
            self.preview_textbox.configure(state="disabled")

    def show_preview_image(self, file_path: str):
        """在右侧预览框中显示图片。"""
        try:
            self.show_right_preview_panel()

            # 图片预览和文档预览互斥
            self.preview_textbox.pack_forget()
            self.preview_canvas.pack(fill="both", expand=True, pady=(0, 10))

            self.preview_title.configure(text="📸 图片预览区")

            if not os.path.exists(file_path):
                self.preview_canvas.configure(text="文件不存在", image=None)
                self.preview_filename.configure(text="")
                self.current_preview_image = None
                return

            filename = os.path.basename(file_path)
            self.preview_filename.configure(text=filename)

            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                img = Image.open(file_path).convert("RGB")
                photo = ctk.CTkImage(
                    light_image=img,
                    dark_image=img,
                    size=(300, 250)
                )
                self.preview_canvas.configure(image=photo, text="")
                self.current_preview_image = photo
                self.preview_canvas.update()
            else:
                self.show_preview_text(file_path)

        except Exception as e:
            self.preview_canvas.configure(text=f"图片预览失败: {str(e)}", image=None)
            self.current_preview_image = None


if __name__ == "__main__":
    app = BaseAppUI()
    app.mainloop()
