# handwrite_grid_ui.py
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 与训练一致的模型定义 ConvNet
# ---------------------------
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ---------------------------
# 设备 & 模型加载（尝试多个文件）
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ConvNet().to(device)

loaded_file = None
for candidate in ("mnist_cnn.ckpt", "mnist_cnn50.ckpt", "mnist_cnn.pth"):
    if os.path.exists(candidate):
        try:
            state = torch.load(candidate, map_location=device)
            # state may be a state_dict or full model
            if isinstance(state, dict) and all(k.startswith("layer") or k.startswith("fc") or k.startswith("bn") for k in state.keys()):
                model.load_state_dict(state)
            else:
                # try to load full model object (rare)
                model = state
            loaded_file = candidate
            print("Loaded model from", candidate)
            break
        except Exception as e:
            print(f"Failed loading {candidate}: {e}")
if loaded_file is None:
    print("No checkpoint found or failed to load. Using randomly initialized model. Place mnist_cnn.ckpt in the script folder to load.")

model.eval()

# ---------------------------
# UI & Drawing app
# ---------------------------
GRID_NUM = 28            # logical grid: 28x28
CELL_PIXELS = 10         # each visual cell size (pixels)
CANVAS_PIX = GRID_NUM * CELL_PIXELS  # 280

class HandwriteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FUCK FUCK FUCK")
        # fixed size to keep layout stable
        self.root.geometry(f"{CANVAS_PIX + 360}x{CANVAS_PIX + 40}")  # canvas + sidebar width, some margin
        self.root.resizable(True, True)
        self.bg_color = "#121221"
        self.sidebar_bg = "#1f1f33"
        self.root.configure(bg=self.bg_color)

        # Left canvas frame
        self.canvas_frame = tk.Frame(root, bg=self.bg_color)
        self.canvas_frame.grid(row=0, column=0, padx=12, pady=12)

        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_PIX, height=CANVAS_PIX, bg="black", highlightthickness=2, highlightbackground="#00c2a8")
        self.canvas.pack()

        # Create grid rectangles and keep refs
        self.rects = {}
        for r in range(GRID_NUM):
            for c in range(GRID_NUM):
                x1 = c * CELL_PIXELS
                y1 = r * CELL_PIXELS
                x2 = x1 + CELL_PIXELS
                y2 = y1 + CELL_PIXELS
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#222232", fill="black")
                self.rects[(r, c)] = rect

        # Logical image: 28x28 L, black background (0), white stroke (255)
        self.img = Image.new("L", (GRID_NUM, GRID_NUM), 0)
        self.draw = ImageDraw.Draw(self.img)

        # Bind events: left to paint (fill whole cell), right to erase
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<Button-1>", self.on_paint)
        self.canvas.bind("<Button-3>", self.on_erase)

        # Sidebar frame
        self.sidebar = tk.Frame(root, width=340, height=CANVAS_PIX, bg=self.sidebar_bg)
        self.sidebar.grid(row=0, column=1, padx=(0,12), pady=12, sticky="n")

        # Preview label and image
        lbl_preview = tk.Label(self.sidebar, text="灰度预览 (模型输入)", fg="#a8fff0", bg=self.sidebar_bg, font=("Arial", 11, "bold"))
        lbl_preview.pack(pady=(8,4))
        self.preview_holder = tk.Label(self.sidebar, bg="#333344")
        self.preview_holder.pack(pady=4)

        # Prediction display
        lbl_guess = tk.Label(self.sidebar, text="老子猜这是：", fg="white", bg=self.sidebar_bg, font=("Arial", 12))
        lbl_guess.pack(pady=(14,0))
        self.pred_label = tk.Label(self.sidebar, text="?", fg="red", bg=self.sidebar_bg, font=("Arial", 48, "bold"))
        self.pred_label.pack(pady=4)

        # Probability chart area using matplotlib
        self.fig, self.ax = plt.subplots(figsize=(3.2,2.3), dpi=80)
        self.fig.patch.set_facecolor(self.sidebar_bg)
        self.ax.set_facecolor(self.sidebar_bg)
        self.ax.tick_params(colors="white")
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=self.sidebar)
        self.canvas_chart.get_tk_widget().pack(pady=6)

        # Entry + confirm button
        entry_frame = tk.Frame(self.sidebar, bg=self.sidebar_bg)
        entry_frame.pack(pady=(8,4))
        self.entry = tk.Entry(entry_frame, font=("Arial", 12))
        self.entry.grid(row=0, column=0, padx=(0,6))
        btn_confirm = tk.Button(entry_frame, text="确定", command=self.check_answer, bg="#00c2a8", fg="black", font=("Arial", 11, "bold"))
        btn_confirm.grid(row=0, column=1)

        # Buttons at bottom
        btn_frame = tk.Frame(self.sidebar, bg=self.sidebar_bg)
        btn_frame.pack(pady=10)
        btn_predict = tk.Button(btn_frame, text="预测", command=self.predict, bg="#00a6ff", fg="white", width=12, font=("Arial", 11, "bold"))
        btn_predict.grid(row=0, column=0, padx=6)
        btn_clear = tk.Button(btn_frame, text="清空画布", command=self.clear_canvas, bg="#ff4d6d", fg="white", width=12, font=("Arial", 11, "bold"))
        btn_clear.grid(row=0, column=1, padx=6)

        # bottom info
        self.status = tk.Label(self.sidebar, text=f"模型: {'已加载' if 'mnist_cnn.ckpt' in os.listdir('.') or 'mnist_cnn50.ckpt' in os.listdir('.') else '未加载'}", fg="#cfe", bg=self.sidebar_bg)
        self.status.pack(pady=(8,0))

        # prepare transformation (no normalization since trained with ToTensor only)
        self.transform = transforms.ToTensor()

        # initial empty plot
        self.plot_probs(np.zeros(10))

    def on_paint(self, event):
        c = int(event.x // CELL_PIXELS)
        r = int(event.y // CELL_PIXELS)
        if 0 <= r < GRID_NUM and 0 <= c < GRID_NUM:
            # visually fill cell
            self.canvas.itemconfig(self.rects[(r,c)], fill="white")
            # logically set pixel to white (255)
            self.draw.point((c, r), 255)

    def on_erase(self, event):
        c = int(event.x // CELL_PIXELS)
        r = int(event.y // CELL_PIXELS)
        if 0 <= r < GRID_NUM and 0 <= c < GRID_NUM:
            self.canvas.itemconfig(self.rects[(r,c)], fill="black")
            self.draw.point((c, r), 0)

    def clear_canvas(self):
        # reset visual grid and image
        for (r,c), rect in self.rects.items():
            self.canvas.itemconfig(rect, fill="black")
        self.img = Image.new("L", (GRID_NUM, GRID_NUM), 0)
        self.draw = ImageDraw.Draw(self.img)
        # clear preview & prediction & chart
        self.preview_holder.config(image="", text="")
        self.pred_label.config(text="?")
        self.entry.delete(0, tk.END)
        self.plot_probs(np.zeros(10))

    def predict(self):
        # self.img is already logical 28x28 (one pixel per grid cell)
        # Convert to PIL Image sized 28x28, and (optionally) invert if needed.
        img_28 = self.img.copy().resize((28,28), Image.NEAREST)  # already 28x28, but safe
        # show preview scaled up
        preview = img_28.resize((140,140), Image.NEAREST)
        tk_preview = ImageTk.PhotoImage(preview)
        self.preview_holder.config(image=tk_preview)
        self.preview_holder.image = tk_preview

        # prepare tensor: [1,1,28,28], values in [0,1]
        tensor = self.transform(img_28).unsqueeze(0).to(device)  # ToTensor: 0..1 where white = 1

        # predict
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
            pred = int(probs.argmax())

        # show result
        self.pred_label.config(text=str(pred))
        self.prediction = pred

        # plot probabilities and mark top as red
        self.plot_probs(probs)

    def plot_probs(self, probs):
        self.ax.clear()
        colors = ["#00d6c6"] * 10
        top_idx = int(np.argmax(probs)) if probs is not None and len(probs)>0 else -1
        if top_idx >= 0:
            colors[top_idx] = "#ff4444"  # highlight top in red
        bars = self.ax.bar(range(10), probs, color=colors)
        # show numeric on top of bars
        for i, b in enumerate(bars):
            h = b.get_height()
            self.ax.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}", ha='center', color='white', fontsize=8)
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 1)
        self.ax.set_facecolor(self.sidebar_bg)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.tick_params(colors='white')
        self.ax.set_title("预测概率分布", color='white', fontsize=10)
        self.fig.tight_layout()
        self.canvas_chart.draw()

    def check_answer(self):
        user_in = self.entry.get().strip()
        if not user_in.isdigit():
            messagebox.showwarning("提示", "请输入 0-9 的数字再确认。")
            return
        if int(user_in) == getattr(self, "prediction", -1):
            messagebox.showinfo("牛逼", "我真他妈牛逼! 你也是 😎")
        else:
            messagebox.showwarning("菜！",
                                   "我操，你怎么训练的我")

# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwriteApp(root)
    root.mainloop()
