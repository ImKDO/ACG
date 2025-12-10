from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np

# Интеграция matplotlib в Tkinter
import matplotlib
matplotlib.use("TkAgg")

APP_TITLE = "Анализатор RGB-состава изображения"


class RgbAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x700")
        self.minsize(800, 500)

        # Исходное и текущее (с фильтрами) изображения PIL
        self.original_pil_img = None
        self.current_pil_img = None
        self.image_path = ""
        self.image_tk = None  # Ссылка на PhotoImage, чтобы избежать сборки мусора

        self._build_ui()
        self._update_ui_state()

        # Привязка события изменения размера окна для масштабирования картинки
        self.image_panel.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        """Создает и размещает все виджеты интерфейса."""
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # --- Основная структура ---
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # --- 1. Панель управления (сверху) ---
        control_panel = self._create_control_panel(main_frame)
        control_panel.grid(row=0, column=0, columnspan=2,
                           sticky="ew", pady=(0, 10))

        # --- 2. Панель с изображением (слева) ---
        self.image_panel = self._create_image_panel(main_frame)
        self.image_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 1))

        # --- 3. Панель с графиком (справа) ---
        chart_panel = self._create_chart_panel(main_frame)
        chart_panel.grid(row=1, column=1, sticky="nsew", padx=(1, 0))

        # --- 4. Строка состояния (снизу) ---
        self.status_var = tk.StringVar(
            value="Откройте изображение для начала работы.")
        status_bar = ttk.Label(
            self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=1)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_control_panel(self, parent):
        """Создает верхнюю панель с кнопками."""
        frame = ttk.Frame(parent)
        self.open_btn = ttk.Button(
            frame, text="Открыть изображение...", command=self._open_image)
        self.open_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.filter_grayscale_btn = ttk.Button(
            frame, text="Оттенки серого", command=self._apply_grayscale)
        self.filter_grayscale_btn.pack(side=tk.LEFT, padx=5)

        self.filter_invert_btn = ttk.Button(
            frame, text="Инвертировать цвета", command=self._apply_invert)
        self.filter_invert_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(
            frame, text="Сбросить", command=self._reset_image)
        self.reset_btn.pack(side=tk.LEFT, padx=(10, 0))
        return frame

    def _create_image_panel(self, parent):
        """Создает левую панель для отображения картинки."""
        frame = ttk.Labelframe(parent, text="Изображение", padding=1)
        self.image_label = ttk.Label(
            frame, text="Нет открытого изображения", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        return frame

    def _create_chart_panel(self, parent):
        """Создает правую панель для графика Matplotlib."""
        frame = ttk.Labelframe(parent, text="Анализ RGB", padding=1)
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        fig.tight_layout()
        return frame

    # --- Логика приложения ---
    def _open_image(self):
        """Открывает диалог выбора файла и загружает изображение."""
        path = filedialog.askopenfilename(
            title="Выбрать изображение",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.original_pil_img = Image.open(path).convert("RGB")
            self.current_pil_img = self.original_pil_img.copy()
            self.image_path = path
            self._update_all_views()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            self.original_pil_img = self.current_pil_img = None
        self._update_ui_state()

    def _reset_image(self):
        """Сбрасывает все примененные фильтры."""
        if self.original_pil_img:
            self.current_pil_img = self.original_pil_img.copy()
            self._update_all_views()

    def _apply_grayscale(self):
        if self.original_pil_img:
            # Применяем фильтр к оригиналу, чтобы избежать накопления эффектов
            self.current_pil_img = ImageOps.grayscale(
                self.original_pil_img).convert("RGB")
            self._update_all_views()

    def _apply_invert(self):
        if self.original_pil_img:
            self.current_pil_img = ImageOps.invert(self.original_pil_img)
            self._update_all_views()

    def _update_all_views(self):
        """Обновляет изображение на экране и перерисовывает график."""
        if self.current_pil_img:
            self._update_image_preview()
            self._draw_rgb_bars()
            w, h = self.original_pil_img.size
            self.status_var.set(
                f"Файл: {self.image_path}  |  Размеры: {w}x{h}")

    def _update_image_preview(self):
        """Масштабирует и отображает текущее изображение."""
        if not self.current_pil_img:
            return

        panel_w = self.image_panel.winfo_width()
        panel_h = self.image_panel.winfo_height()

        # Создаем копию для масштабирования
        img_copy = self.current_pil_img.copy()
        img_copy.thumbnail((panel_w - 20, panel_h - 20), Image.LANCZOS)

        self.image_tk = ImageTk.PhotoImage(img_copy)
        self.image_label.config(image=self.image_tk, text="")
        self.image_label.image = self.image_tk

    def _on_resize(self, event):
        """Вызывается при изменении размера окна для перерисовки изображения."""
        self._update_image_preview()

    def _update_ui_state(self):
        """Включает/выключает кнопки в зависимости от наличия изображения."""
        state = tk.NORMAL if self.original_pil_img else tk.DISABLED
        self.filter_grayscale_btn.config(state=state)
        self.filter_invert_btn.config(state=state)
        self.reset_btn.config(state=state)

    # --- Логика графика ---
    def _draw_rgb_bars(self):
        """Рассчитывает данные и рисует столбчатую диаграмму RGB."""
        if not self.current_pil_img:
            return

        arr = np.asarray(self.current_pil_img, dtype=np.uint32)
        sums = np.array([arr[..., 0].sum(), arr[..., 1].sum(),
                        arr[..., 2].sum()], dtype=np.float64)
        total = sums.sum() if sums.sum() > 0 else 1.0
        percents = 100.0 * sums / total

        self.ax.clear()
        self.ax.set_title("RGB-состав изображения")
        if sums.max() > 0:
            self.ax.set_ylim(0, sums.max() * 1.15)
        self.ax.set_xticks([0, 1, 2], ["Red", "Green", "Blue"])
        # self.ax.set_ybound(lower=0, upper=100)
            
        bars = self.ax.bar([0, 1, 2], sums, width=0.6, color=[
                           "#e74c3c", "#2ecc71", "#3498db"])

        for i, b in enumerate(bars):
            self.ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + (sums.max() * 0.02),
                f"{percents[i]:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        self.ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        self.ax.grid(axis="y", linestyle="--", alpha=0.6)
        self.canvas.draw()


if __name__ == "__main__":
    app = RgbAnalyzer()
    app.mainloop()
