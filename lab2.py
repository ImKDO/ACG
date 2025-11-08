import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import math

APP_TITLE = "Filters (Custom Implementation): Grey, Blur, Contrast, Brightness, Invert"

# --------- helpers ---------
# Эта часть остается без изменений
def pil_to_tk(img, max_w, max_h):
    """Конвертирует изображение PIL в формат, понятный Tkinter."""
    if img is None: return None
    w, h = img.size
    # Предотвращение деления на ноль, если изображение имеет нулевой размер
    scale = min(max_w / w, max_h / h) if w and h else 1.0
    scale = max(scale, 1e-6) # Убедимся, что масштаб не нулевой
    thumb = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.LANCZOS)
    return ImageTk.PhotoImage(thumb)

def find_builtin_picture(base_name: str) -> str | None:
    """Ищет встроенные изображения в папке скрипта или в подпапке /assets."""
    exts = ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"]
    here = os.path.dirname(os.path.abspath(__file__))
    for ext in exts:
        cand = os.path.join(here, f"{base_name}.{ext}")
        if os.path.isfile(cand):
            return cand
    assets = os.path.join(here, "assets")
    for ext in exts:
        cand = os.path.join(assets, f"{base_name}.{ext}")
        if os.path.isfile(cand):
            return cand
    matches = glob.glob(os.path.join(here, f"{base_name}.*")) + glob.glob(os.path.join(assets, f"{base_name}.*"))
    for m in matches:
        if os.path.isfile(m):
            return m
    return None

# --------- Custom Image Processing Algorithms (Optimized) ---------

def clamp(value, min_val=0, max_val=255):
    """Ограничивает значение диапазоном [min_val, max_val]."""
    return int(max(min_val, min(value, max_val)))

def apply_custom_grayscale(image: Image.Image) -> Image.Image:
    """Применяет фильтр оттенков серого к изображению."""
    src_pixels = image.load()
    width, height = image.size
    new_pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b = src_pixels[x, y]
            # Стандартная формула для вычисления яркости (Luminosity)
            lum = clamp(0.299 * r + 0.587 * g + 0.114 * b)
            new_pixels.append((lum, lum, lum))
    
    new_image = Image.new("RGB", image.size)
    new_image.putdata(new_pixels)
    return new_image

def apply_custom_invert(image: Image.Image) -> Image.Image:
    """Инвертирует цвета изображения."""
    src_pixels = image.load()
    width, height = image.size
    new_pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b = src_pixels[x, y]
            new_pixels.append((255 - r, 255 - g, 255 - b))
            
    new_image = Image.new("RGB", image.size)
    new_image.putdata(new_pixels)
    return new_image

def apply_custom_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Изменяет яркость изображения."""
    src_pixels = image.load()
    width, height = image.size
    new_pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b = src_pixels[x, y]
            new_r = clamp(r * factor)
            new_g = clamp(g * factor)
            new_b = clamp(b * factor)
            new_pixels.append((new_r, new_g, new_b))
            
    new_image = Image.new("RGB", image.size)
    new_image.putdata(new_pixels)
    return new_image

def apply_custom_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Изменяет контрастность изображения."""
    src_pixels = image.load()
    width, height = image.size
    new_pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b = src_pixels[x, y]
            # Формула для изменения контраста относительно среднего серого (128)
            new_r = clamp(128 + factor * (r - 128))
            new_g = clamp(128 + factor * (g - 128))
            new_b = clamp(128 + factor * (b - 128))
            new_pixels.append((new_r, new_g, new_b))
            
    new_image = Image.new("RGB", image.size)
    new_image.putdata(new_pixels)
    return new_image

def apply_custom_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Применяет размытие по Гауссу."""
    if radius < 0.5:
        return image.copy()

    sigma = max(radius, 0.5)
    kernel_size = int(radius * 3)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    
    kernel = [[0.0] * kernel_size for _ in range(kernel_size)]
    kernel_sum = 0.0
    center = kernel_size // 2

    # Создание ядра Гаусса
    for x in range(kernel_size):
        for y in range(kernel_size):
            dx, dy = x - center, y - center
            value = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            kernel[y][x] = value
            kernel_sum += value

    # Нормализация ядра
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[y][x] /= kernel_sum

    # Применение свертки (convolution)
    src_pixels = image.load()
    width, height = image.size
    new_pixels = []

    for y in range(height):
        for x in range(width):
            r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    # Определение координат пикселя-соседа с учетом границ изображения
                    px = clamp(x + (kx - center), 0, width - 1)
                    py = clamp(y + (ky - center), 0, height - 1)
                    
                    r, g, b = src_pixels[px, py]
                    weight = kernel[ky][kx]
                    
                    r_sum += r * weight
                    g_sum += g * weight
                    b_sum += b * weight
            
            new_pixels.append((clamp(r_sum), clamp(g_sum), clamp(b_sum)))

    new_image = Image.new("RGB", image.size)
    new_image.putdata(new_pixels)
    return new_image


# --------- app ---------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x760")
        self.minsize(900, 600)
        
        self.job_id = None 
        self.src_img = None
        self.result_img = None

        self.active_tool = None
        self.live_base = None

        self._build_ui()
        self.bind("<Configure>", lambda e: self._refresh())
        self.bind("<KeyPress>", self._on_key_press)
        self._update_status()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Grey", command=self.apply_grey).grid(row=0, column=0, padx=6)
        ttk.Button(top, text="Blur", command=lambda: self._activate_live('blur')).grid(row=0, column=1, padx=(0,6))

        ttk.Label(top, text="Radius:").grid(row=0, column=2, padx=(12,4), sticky="e")
        self.radius = tk.DoubleVar(value=1.0)
        ttk.Scale(top, from_=0.0, to=5.0, variable=self.radius, # Уменьшен лимит для производительности
                  command=self._apply_live, 
                  orient="horizontal").grid(row=0, column=3, sticky="we", padx=(0,8))
        self._value_label(top, self.radius, 4)

        ttk.Separator(top, orient="vertical").grid(row=0, column=5, sticky="ns", padx=8)

        ttk.Button(top, text="Contrast", command=lambda: self._activate_live('contrast')).grid(row=0, column=6, padx=(0,6))
        ttk.Label(top, text="Factor:").grid(row=0, column=7, padx=(6,4), sticky="e")
        self.contrast = tk.DoubleVar(value=1.0)
        ttk.Scale(top, from_=0.0, to=3.0, variable=self.contrast,
                  command=self._apply_live,
                  orient="horizontal").grid(row=0, column=8, sticky="we", padx=(0,8))
        self._value_label(top, self.contrast, 9)
        
        ttk.Separator(top, orient="vertical").grid(row=0, column=10, sticky="ns", padx=8)

        ttk.Button(top, text="Bright", command=lambda: self._activate_live('brightness')).grid(row=0, column=11, padx=(4,6))
        self.brightness = tk.DoubleVar(value=1.0)
        ttk.Scale(top, from_=0.0, to=3.0, variable=self.brightness,
                  command=self._apply_live,
                  orient="horizontal").grid(row=0, column=13, sticky="we", padx=(0,8))
        self._value_label(top, self.brightness, 14, col_start=12, label_text="Brightness:")
        
        ttk.Separator(top, orient="vertical").grid(row=0, column=15, sticky="ns", padx=8)

        ttk.Button(top, text="Invert Colors", command=self.apply_invert).grid(row=0, column=16, padx=(0,8))
        ttk.Button(top, text="Save PNG", command=self.save_png).grid(row=0, column=17, padx=(0,8))
        ttk.Button(top, text="Open...", command=self.open_dialog).grid(row=0, column=18, padx=(0,8))
        ttk.Button(top, text="Reset", command=self.reset_result).grid(row=0, column=19, padx=(0,8))

        for col in (3, 8, 13): top.columnconfigure(col, weight=1)

        mid = ttk.Frame(self, padding=8); mid.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(mid); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,4))
        self.canvas_src = tk.Canvas(left, bg="#e9e9e9", highlightthickness=0)
        self.canvas_src.pack(fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Исходное").pack(fill=tk.X, pady=(6,0))
        right = ttk.Frame(mid); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,0))
        self.canvas_res = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0)
        self.canvas_res.pack(fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Результат").pack(fill=tk.X, pady=(6,0))

        bottom = ttk.Frame(self, padding=(8,0,8,8)); bottom.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(bottom, text="Picture 1", command=lambda: self.load_builtin("picture1")).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(bottom, text="Picture 2", command=lambda: self.load_builtin("picture2")).pack(side=tk.LEFT, padx=(0,6))
        
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self, textvariable=self.status_var, padding=5, anchor="w", relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _value_label(self, parent, var, col, col_start=None, label_text=None):
        if label_text:
            ttk.Label(parent, text=label_text).grid(row=0, column=col_start, padx=(6,4), sticky="e")
        lbl = ttk.Label(parent, width=4, anchor="e"); lbl.grid(row=0, column=col+1, sticky="e")
        var.trace_add("write", lambda *_: lbl.config(text=f"{var.get():.1f}")); lbl.config(text=f"{var.get():.1f}")
        
    # ---------- File I/O ----------
    def open_dialog(self):
        path = filedialog.askopenfilename(title="Открыть изображение", filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files","*.*")], initialdir=".")
        if path: self.load_image(path)

    def load_builtin(self, base_name: str):
        path = find_builtin_picture(base_name)
        if not path:
            messagebox.showwarning("Файл не найден", f"Не нашёл {base_name}. Положите файл рядом со скриптом или в папку ./assets.")
            return
        self.load_image(path)

    def load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}"); return
        self.src_img = img; self.result_img = img.copy()
        self._commit_live(); self._refresh(); self._update_status()

    def save_png(self):
        if self.result_img is None: messagebox.showinfo("Сохранение", "Нет результата для сохранения."); return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], title="Сохранить результат как PNG")
        if not path: return
        try:
            self.result_img.save(path, "PNG"); messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as e: messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    # ---------- Filters & Live Mode ----------
    def reset_result(self):
        if self.src_img is None: return
        self.result_img = self.src_img.copy(); self._commit_live(); self._refresh()

    def apply_grey(self):
        if self.result_img is None: return
        self._commit_live(); self.result_img = apply_custom_grayscale(self.result_img); self._refresh()

    def apply_invert(self):
        if self.result_img is None: return
        self._commit_live(); self.result_img = apply_custom_invert(self.result_img); self._refresh()

    def _activate_live(self, tool):
        if self.result_img is None: return
        if self.active_tool: self._commit_live()
        self.active_tool = tool; self.live_base = self.result_img.copy()
        self._apply_live(); self._update_status()

    def _commit_live(self):
        if self.active_tool is None: return
        self.active_tool = None; self.live_base = None
        if self.job_id: self.after_cancel(self.job_id); self.job_id = None
        self._update_status()

    def _cancel_live(self):
        if self.live_base: self.result_img = self.live_base.copy()
        self._commit_live(); self._refresh()

    def _apply_live(self, *args):
        """Планирует отложенное обновление, чтобы не перегружать GUI."""
        if self.job_id: self.after_cancel(self.job_id)
        self.job_id = self.after(250, self._perform_live_update) # Задержка 250 мс

    def _perform_live_update(self):
        """Выполняет фактическую обработку изображения для 'живого' режима."""
        self.job_id = None
        if self.live_base is None or self.active_tool is None: return
        
        img = self.live_base
        
        if self.active_tool == 'blur':
            img = apply_custom_gaussian_blur(img, float(self.radius.get()))
        elif self.active_tool == 'contrast':
            img = apply_custom_contrast(img, float(self.contrast.get()))
        elif self.active_tool == 'brightness':
            img = apply_custom_brightness(img, float(self.brightness.get()))
            
        self.result_img = img; self._refresh()
    
    # ---------- Drawing & Status ----------
    def _refresh(self):
        for cv, img in [(self.canvas_src, self.src_img), (self.canvas_res, self.result_img)]:
            cv.delete("all")
            if img:
                tkimg = pil_to_tk(img, cv.winfo_width(), cv.winfo_height())
                # Сохраняем ссылку на объект, чтобы сборщик мусора его не удалил
                cv.image = tkimg 
                cv.create_image(cv.winfo_width()//2, cv.winfo_height()//2, image=tkimg, anchor="center")
                                         
    def _update_status(self):
        if self.active_tool:
            self.status_var.set(f"{self.active_tool.capitalize()} mode: Use arrows. Enter to confirm, Esc to cancel.")
        elif self.src_img:
            self.status_var.set("Shortcuts: Ctrl+O, Ctrl+S, G (Grey), I (Invert), B (Blur), C (Contrast), H (Brightness), R (Reset)")
        else: self.status_var.set("Load an image to begin (Ctrl+O or Open button).")

    # ---------- Key Press Handling ----------
    def _on_key_press(self, event):
        if event.state & 0x0004: 
            key_map = {'o': self.open_dialog, 's': self.save_png, 'r': self.reset_result}
            if event.keysym.lower() in key_map: key_map[event.keysym.lower()](); return
        
        if self.src_img is None: return
        
        if self.active_tool:
            if event.keysym == "Return": self._commit_live(); return
            if event.keysym == "Escape": self._cancel_live(); return
            
            step, var, increment, limits = 0, None, 0, (0,0)
            if event.keysym == "Right": step = 1
            elif event.keysym == "Left": step = -1
            
            if step != 0:
                tool_map = {
                    'blur': (self.radius, 0.2 * step, (0.0, 5.0)),
                    'contrast': (self.contrast, 0.05 * step, (0.0, 3.0)),
                    'brightness': (self.brightness, 0.05 * step, (0.0, 3.0))
                }
                if self.active_tool in tool_map:
                    var, increment, limits = tool_map[self.active_tool]
                    new_val = round(var.get() + increment, 2)
                    var.set(max(limits[0], min(limits[1], new_val)))
                return
        
        key_map = {'g': self.apply_grey, 'i': self.apply_invert, 'r': self.reset_result,
                   'b': lambda: self._activate_live('blur'), 'c': lambda: self._activate_live('contrast'),
                   'h': lambda: self._activate_live('brightness')}
        if event.keysym.lower() in key_map: key_map[event.keysym.lower()]()

if __name__ == "__main__":
    App().mainloop()