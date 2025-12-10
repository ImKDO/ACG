import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import math

class RenderEngine:
    """Класс, отвечающий только за математику и рендеринг."""

    @staticmethod
    def ray_sphere_intersect(O, D, C, R):
        """
        Векторизированный расчет пересечения лучей со сферой.
        """
        OC = O - C
        a = np.sum(D * D, axis=-1)
        b = 2.0 * np.sum(D * OC, axis=-1)
        c = np.sum(OC * OC, axis=-1) - R ** 2
        disc = b * b - 4 * a * c
        
        hit = disc >= 0
        t = np.full_like(a, np.inf)
        
        # Вычисляем корни только там, где есть пересечение
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        denom = 2 * a + 1e-9
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        
        # Логика выбора ближайшего положительного t
        t_candidate = np.where((t1 > 1e-4) & ((t1 < t2) | (t2 <= 1e-4)), t1, t2)
        t = np.where((t_candidate > 1e-4) & hit, t_candidate, np.inf)
        
        return t, t != np.inf

    @staticmethod
    def render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights):
        """
        Рендерит сцену с тенями и освещением Блинна–Фонга.
        """
        # 1. Генерация лучей
        O = np.array([0.0, 0.0, zO], dtype=np.float64)

        # Сетка координат экрана
        xs = np.linspace(-W / 2, W / 2, Wres)
        ys = np.linspace(H / 2, -H / 2, Hres) # Y вверх
        X, Y = np.meshgrid(xs, ys)
        
        # Формируем массив точек экрана
        P_screen = np.stack([X, Y, np.full_like(X, z_scr)], axis=-1)
        
        # Направления лучей (нормализованные)
        D = P_screen - O
        D = D / np.linalg.norm(D, axis=-1, keepdims=True)

        # Плоские массивы для векторизации
        D_flat = D.reshape(-1, 3)
        P_flat_count = D_flat.shape[0]

        # 2. Поиск ближайших пересечений (Z-buffer)
        depth = np.full(P_flat_count, np.inf)
        sphere_id = np.full(P_flat_count, -1, dtype=int)

        for i, sph in enumerate(spheres):
            C = np.array(sph['center'])
            R = sph['radius']
            t, hit = RenderEngine.ray_sphere_intersect(O, D_flat, C, R)
            
            mask = hit & (t < depth)
            depth[mask] = t[mask]
            sphere_id[mask] = i

        # 3. Шейдинг
        final_color = np.zeros((P_flat_count, 3))
        valid_mask = sphere_id >= 0
        
        # Если ничего не видно, возвращаем пустой фон
        if not np.any(valid_mask):
            return np.zeros((Hres, Wres, 3), dtype=np.uint8), np.zeros((Hres, Wres, 3)), 0.0, 0.0

        # Точки пересечения в мировых координатах
        P_hit = O + D_flat[valid_mask] * depth[valid_mask, np.newaxis]
        
        # Нормали
        centers = np.array([spheres[i]['center'] for i in sphere_id[valid_mask]])
        radii = np.array([spheres[i]['radius'] for i in sphere_id[valid_mask]])
        N = (P_hit - centers) / radii[:, np.newaxis] # Нормализация делением на радиус
        
        # Вектор взгляда (V)
        V = -D_flat[valid_mask] # От точки к наблюдателю
        
        # Накопитель цвета
        colors_acc = np.zeros_like(P_hit)

        # Индексы сфер для каждой точки
        active_ids = sphere_id[valid_mask]

        # Проход по источникам света
        for light in lights:
            L_pos = np.array(light['pos'])
            L_vec = L_pos - P_hit
            L_dist = np.linalg.norm(L_vec, axis=1, keepdims=True)
            L_dir = L_vec / (L_dist + 1e-9)

            # --- ТЕНИ ---
            # Пускаем лучи от точки пересечения к источнику света
            shadow_mask = np.zeros(len(P_hit), dtype=bool)
            shadow_origin = P_hit + N * 1e-3 # Сдвиг (bias) чтобы не пересечь саму себя

            for other_sph in spheres:
                C_other = np.array(other_sph['center'])
                R_other = other_sph['radius']
                t_shadow, hit_shadow = RenderEngine.ray_sphere_intersect(
                    shadow_origin, L_dir, C_other, R_other
                )
                # Если пересечение есть и оно ближе, чем источник света
                is_shadowed = hit_shadow & (t_shadow < (L_dist.flatten() - 1e-2))
                shadow_mask |= is_shadowed

            # --- ОСВЕЩЕНИЕ (Блинн-Фонг) ---
            # Вычисляем только для освещенных пикселей
            lit_indices = ~shadow_mask
            
            if np.any(lit_indices):
                N_lit = N[lit_indices]
                L_lit = L_dir[lit_indices]
                V_lit = V[lit_indices]
                
                # Параметры материала для каждой точки
                kds = np.array([spheres[i]['kd'] for i in active_ids[lit_indices]])[:, np.newaxis]
                kss = np.array([spheres[i]['ks'] for i in active_ids[lit_indices]])[:, np.newaxis]
                shins = np.array([spheres[i]['shininess'] for i in active_ids[lit_indices]])
                obj_colors = np.array([spheres[i]['color'] for i in active_ids[lit_indices]])
                
                light_color = np.array(light['color'])
                I0 = light['I0']

                # Diffuse
                diff = np.maximum(np.einsum('ij,ij->i', N_lit, L_lit), 0.0)[:, np.newaxis]
                
                # Specular (Blinn-Phong)
                H = L_lit + V_lit
                H = H / np.linalg.norm(H, axis=1, keepdims=True)
                spec_dot = np.maximum(np.einsum('ij,ij->i', N_lit, H), 0.0)
                spec = (spec_dot ** shins)[:, np.newaxis]

                # Итоговый цвет от этого источника
                # Color = (Kd * Diff + Ks * Spec) * LightColor * ObjColor * Intensity
                # Примечание: обычно Specular берет цвет источника (белый блик), а Diffuse - цвет объекта.
                # Но в задании была формула: (I_diff + I_spec) * light_color * sph_color
                
                term = (kds * diff + kss * spec) * I0
                colors_acc[lit_indices] += term * light_color * obj_colors

        final_color[valid_mask] = colors_acc
        
        # Решейп обратно в картинку
        img_rgb = final_color.reshape((Hres, Wres, 3))

        # Нормализация
        I_max = np.max(img_rgb)
        I_min = 0.0
        if I_max > 0:
            I_min = np.min(img_rgb[img_rgb > 0]) if np.any(img_rgb > 0) else 0.0
            img_norm = (img_rgb / I_max) * 255
        else:
            img_norm = img_rgb

        img_uint8 = np.clip(img_norm, 0, 255).astype(np.uint8)
        
        return img_uint8, img_rgb, I_max, I_min

    @staticmethod
    def render_projections(W, H, Wres, Hres, spheres, lights):
        """
        Генерирует словарь с тремя проекциями.
        """
        projections = {}
        # Камера издалека для ортогонального эффекта
        zO_proj = 50000.0 
        z_scr = 0.0

        # 1. Фронтальная (XY) - Обычная
        img_xy, _, imax, imin = RenderEngine.render_scene(
            W, H, Wres, Hres, zO_proj, z_scr, spheres, lights
        )
        projections['frontal'] = {
            'image': img_xy, 'max': imax, 'min': imin, 'name': 'Вид спереди (XY)'
        }

        # 2. Горизонтальная (XZ) - Вид сверху
        # Y становится глубиной, Z становится вертикалью на экране (Y_screen)
        spheres_top = []
        for sph in spheres:
            cx, cy, cz = sph['center']
            # Map: x->x, z->y, y->depth
            spheres_top.append({**sph, 'center': (cx, cz, cy)})
        
        lights_top = []
        for l in lights:
            lx, ly, lz = l['pos']
            lights_top.append({**l, 'pos': (lx, lz, ly)})

        img_xz, _, imax, imin = RenderEngine.render_scene(
            W, H, Wres, Hres, zO_proj, z_scr, spheres_top, lights_top
        )
        projections['horizontal'] = {
            'image': img_xz, 'max': imax, 'min': imin, 'name': 'Вид сверху (XZ)'
        }

        # 3. Профильная (YZ) - Вид сбоку
        # X становится глубиной
        spheres_side = []
        for sph in spheres:
            cx, cy, cz = sph['center']
            # Map: y->x, z->y, x->depth
            spheres_side.append({**sph, 'center': (cy, cz, cx)})
            
        lights_side = []
        for l in lights:
            lx, ly, lz = l['pos']
            lights_side.append({**l, 'pos': (ly, lz, lx)})

        img_yz, _, imax, imin = RenderEngine.render_scene(
            W, H, Wres, Hres, zO_proj, z_scr, spheres_side, lights_side
        )
        projections['profile'] = {
            'image': img_yz, 'max': imax, 'min': imin, 'name': 'Вид сбоку (YZ)'
        }

        return projections

# =====================================================================
# ====================== GUI ПРИЛОЖЕНИЕ ===============================
# =====================================================================

class ModernSceneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР5: Визуализация сфер (Блинн-Фонг)")
        self.geometry("1400x900")
        self.configure(bg='#2b2b2b')

        self._init_styles()
        self._init_ui()
        
        # Хранилище данных
        self.last_pil = None
        self.last_projections = None
        
        # Автозапуск первой генерации (через 500мс после старта)
        self.after(500, self.generate_projections)

    def _init_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff', font=('Segoe UI', 9))
        style.configure('TLabelframe', background='#2b2b2b', foreground='#00bcd4', borderwidth=2)
        style.configure('TLabelframe.Label', background='#2b2b2b', foreground='#00bcd4', font=('Segoe UI', 10, 'bold'))
        style.configure('TButton', background='#00bcd4', foreground='#ffffff', borderwidth=0, font=('Segoe UI', 10, 'bold'))
        style.map('TButton', background=[('active', '#0097a7')])
        style.configure('TEntry', fieldbackground='#3c3c3c', foreground='#ffffff', borderwidth=1)
        style.configure('TNotebook', background='#2b2b2b', borderwidth=0)
        style.configure('TNotebook.Tab', background='#3c3c3c', foreground='#ffffff', padding=[15, 5])
        style.map('TNotebook.Tab', background=[('selected', '#00bcd4')], foreground=[('selected', '#000000')])

    def _init_ui(self):
        # Основной контейнер
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # ЛЕВАЯ ПАНЕЛЬ (Превью)
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(left, text="🖼 ПРЕДПРОСМОТР", font=('Segoe UI', 14, 'bold'), foreground='#00bcd4').pack(pady=(0, 10))
        
        self.image_label = tk.Label(left, bg='black', relief='sunken', bd=2)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.info_var = tk.StringVar(value="Ожидание рендеринга...")
        ttk.Label(left, textvariable=self.info_var, font=('Segoe UI', 10), foreground='#4caf50').pack(pady=5, anchor='w')

        # Кнопки под картинкой
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=10)
        ttk.Button(btns, text="🎨 РЕНДЕРИТЬ ТЕКУЩИЙ ВИД", command=self.render_single).pack(side=tk.LEFT, padx=(0, 10), ipadx=10)
        ttk.Button(btns, text="💾 СОХРАНИТЬ КАРТИНКУ", command=self.save_single).pack(side=tk.LEFT, ipadx=10)

        # ПРАВАЯ ПАНЕЛЬ (Настройки)
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        ttk.Label(right, text="⚙ НАСТРОЙКИ", font=('Segoe UI', 14, 'bold'), foreground='#00bcd4').pack(pady=(0, 10))

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Создаем вкладки
        self._setup_camera_tab()
        self._setup_sphere_tabs()
        self._setup_lights_tab()
        self._setup_projections_tab()

    def _create_entry(self, parent, text, default, row, col=0):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, sticky='ew', pady=2, padx=5)
        ttk.Label(frame, text=text, width=20).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default))
        ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.RIGHT)
        return var

    def _create_color_btn(self, parent, text, default_rgb, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky='ew', pady=2, padx=5)
        ttk.Label(frame, text=text, width=20).pack(side=tk.LEFT)
        
        vars_rgb = [tk.StringVar(value=str(c)) for c in default_rgb]
        
        def _pick():
            c = colorchooser.askcolor()
            if c[1]:
                btn.config(bg=c[1])
                for i, val in enumerate(c[0]):
                    vars_rgb[i].set(f"{val/255:.3f}")

        hex_col = "#%02x%02x%02x" % tuple(int(x*255) for x in default_rgb)
        btn = tk.Button(frame, bg=hex_col, width=4, command=_pick)
        btn.pack(side=tk.RIGHT)
        return vars_rgb

    def _setup_camera_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='📷 Камера')
        
        f = ttk.LabelFrame(tab, text="Экран и Наблюдатель", padding=10)
        f.pack(fill=tk.X, padx=10, pady=10)
        
        self.W_var = self._create_entry(f, "Ширина (мм)", 800, 0)
        self.H_var = self._create_entry(f, "Высота (мм)", 600, 1)
        self.Wres_var = self._create_entry(f, "W пикселей", 400, 2)
        self.Hres_var = self._create_entry(f, "H пикселей", 300, 3)
        self.zO_var = self._create_entry(f, "Z наблюдателя", -5000, 4)

    def _setup_sphere_tabs(self):
        # Сфера 1
        t1 = ttk.Frame(self.notebook)
        self.notebook.add(t1, text='🔴 Сфера 1')
        f1 = ttk.LabelFrame(t1, text="Параметры", padding=10)
        f1.pack(fill=tk.X, padx=10, pady=10)
        
        self.S1_geom = [
            self._create_entry(f1, "Центр X", -150, 0),
            self._create_entry(f1, "Центр Y", 0, 1),
            self._create_entry(f1, "Центр Z", 0, 2),
            self._create_entry(f1, "Радиус", 200, 3)
        ]
        self.S1_mat = [
            self._create_entry(f1, "Kd (Диффуз)", 0.7, 4),
            self._create_entry(f1, "Ks (Блик)", 0.3, 5),
            self._create_entry(f1, "Блеск (n)", 30, 6)
        ]
        self.S1_col = self._create_color_btn(f1, "Цвет", (1.0, 0.2, 0.2), 7)

        # Сфера 2
        t2 = ttk.Frame(self.notebook)
        self.notebook.add(t2, text='🟢 Сфера 2')
        f2 = ttk.LabelFrame(t2, text="Параметры", padding=10)
        f2.pack(fill=tk.X, padx=10, pady=10)
        
        self.S2_geom = [
            self._create_entry(f2, "Центр X", 200, 0),
            self._create_entry(f2, "Центр Y", 0, 1),
            self._create_entry(f2, "Центр Z", 0, 2),
            self._create_entry(f2, "Радиус", 120, 3)
        ]
        self.S2_mat = [
            self._create_entry(f2, "Kd (Диффуз)", 0.6, 4),
            self._create_entry(f2, "Ks (Блик)", 0.4, 5),
            self._create_entry(f2, "Блеск (n)", 50, 6)
        ]
        self.S2_col = self._create_color_btn(f2, "Цвет", (0.2, 0.8, 0.2), 7)

    def _setup_lights_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='💡 Свет')
        
        # Свет 1
        l1 = ttk.LabelFrame(tab, text="Источник 1", padding=10)
        l1.pack(fill=tk.X, padx=10, pady=5)
        self.L1_pos = [
            self._create_entry(l1, "X", 2000, 0),
            self._create_entry(l1, "Y", 1500, 1),
            self._create_entry(l1, "Z", -500, 2)
        ]
        self.L1_I = self._create_entry(l1, "Интенсивность", 800, 3)
        self.L1_col = self._create_color_btn(l1, "Цвет", (1.0, 1.0, 1.0), 4)

        # Свет 2
        l2 = ttk.LabelFrame(tab, text="Источник 2", padding=10)
        l2.pack(fill=tk.X, padx=10, pady=5)
        self.L2_pos = [
            self._create_entry(l2, "X", -1000, 0),
            self._create_entry(l2, "Y", -1000, 1),
            self._create_entry(l2, "Z", -800, 2)
        ]
        self.L2_I = self._create_entry(l2, "Интенсивность", 300, 3)
        self.L2_col = self._create_color_btn(l2, "Цвет", (1.0, 0.8, 0.5), 4)

    def _setup_projections_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='📐 Проекции')
        
        f = ttk.Frame(tab, padding=20)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="Генерация чертежных видов", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        ttk.Button(f, text="🔄 СГЕНЕРИРОВАТЬ ПРОЕКЦИИ", command=self.generate_projections).pack(fill=tk.X, pady=5, ipady=10)
        ttk.Button(f, text="💾 СОХРАНИТЬ ВСЕ ВИДЫ", command=self.save_projections).pack(fill=tk.X, pady=5, ipady=10)

        self.proj_status = ttk.Label(f, text="", foreground='#bbbbbb')
        self.proj_status.pack(pady=10)

    def _get_data(self):
        """Сбор данных из GUI"""
        try:
            # Сферы
            s1 = {
                'center': (float(self.S1_geom[0].get()), float(self.S1_geom[1].get()), float(self.S1_geom[2].get())),
                'radius': float(self.S1_geom[3].get()),
                'kd': float(self.S1_mat[0].get()), 'ks': float(self.S1_mat[1].get()), 'shininess': float(self.S1_mat[2].get()),
                'color': tuple(float(v.get()) for v in self.S1_col)
            }
            s2 = {
                'center': (float(self.S2_geom[0].get()), float(self.S2_geom[1].get()), float(self.S2_geom[2].get())),
                'radius': float(self.S2_geom[3].get()),
                'kd': float(self.S2_mat[0].get()), 'ks': float(self.S2_mat[1].get()), 'shininess': float(self.S2_mat[2].get()),
                'color': tuple(float(v.get()) for v in self.S2_col)
            }
            # Свет
            l1 = {
                'pos': tuple(float(v.get()) for v in self.L1_pos),
                'I0': float(self.L1_I.get()),
                'color': tuple(float(v.get()) for v in self.L1_col)
            }
            l2 = {
                'pos': tuple(float(v.get()) for v in self.L2_pos),
                'I0': float(self.L2_I.get()),
                'color': tuple(float(v.get()) for v in self.L2_col)
            }
            
            return {
                'W': float(self.W_var.get()), 'H': float(self.H_var.get()),
                'Wres': int(self.Wres_var.get()), 'Hres': int(self.Hres_var.get()),
                'zO': float(self.zO_var.get()), 'z_scr': 0.0,
                'spheres': [s1, s2],
                'lights': [l1, l2]
            }
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте числовые поля!")
            return None

    def render_single(self):
        d = self._get_data()
        if not d: return
        
        try:
            img_arr, _, imax, imin = RenderEngine.render_scene(
                d['W'], d['H'], d['Wres'], d['Hres'], d['zO'], d['z_scr'], d['spheres'], d['lights']
            )
            self.last_pil = Image.fromarray(img_arr)
            self._update_preview(self.last_pil)
            self.info_var.set(f"Одиночный рендер: Max={imax:.2f}, Min={imin:.2f}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def generate_projections(self):
        d = self._get_data()
        if not d: return

        try:
            self.last_projections = RenderEngine.render_projections(
                d['W'], d['H'], d['Wres'], d['Hres'], d['spheres'], d['lights']
            )
            
            # Создаем композит для превью
            comp = self._create_composite_image()
            self._update_preview(comp)
            
            self.proj_status.config(text="✔ Проекции успешно сгенерированы")
            self.info_var.set("Отображаются 3 проекции")
            
            # Переключаем на вкладку проекций для наглядности (опционально)
            # self.notebook.select(3) 
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _update_preview(self, pil_img):
        w_box = self.image_label.winfo_width()
        h_box = self.image_label.winfo_height()
        if w_box < 10: w_box = 600
        
        # Сохраняем пропорции
        ratio = min(w_box / pil_img.width, h_box / pil_img.height)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        
        show_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(show_img)
        self.image_label.config(image=self.tk_img)

    def _create_composite_image(self):
        """Собирает 3 картинки в одну. Размеры берет из САМИХ картинок."""
        if not self.last_projections: return None
        
        # Берем размеры из первой попавшейся проекции, чтобы не зависеть от UI
        ref_img = self.last_projections['frontal']['image']
        hres, wres, _ = ref_img.shape
        
        margin = 10
        text_h = 30
        
        total_w = wres * 2 + margin * 3
        total_h = (hres + text_h) * 2 + margin * 3
        
        comp = Image.new('RGB', (total_w, total_h), '#2b2b2b')
        draw = ImageDraw.Draw(comp)
        
        # Попытка загрузить шрифт
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        def place(key, col, row):
            data = self.last_projections[key]
            img = Image.fromarray(data['image'])
            x = margin + col * (wres + margin)
            y = margin + row * (hres + text_h + margin)
            
            # Заголовок
            draw.text((x, y), data['name'], fill='white', font=font)
            # Картинка
            comp.paste(img, (x, y + text_h))

        # Размещаем по сетке
        place('horizontal', 1, 0) # Справа сверху
        place('frontal', 0, 1)    # Слева снизу
        place('profile', 1, 1)    # Справа снизу
        
        return comp

    def save_single(self):
        if not self.last_pil: return
        fn = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if fn: self.last_pil.save(fn)

    def save_projections(self):
        if not self.last_projections:
            messagebox.showwarning("Внимание", "Сначала нажмите 'СГЕНЕРИРОВАТЬ ПРОЕКЦИИ'")
            return
            
        fn = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if fn:
            try:
                comp = self._create_composite_image()
                comp.save(fn)
                messagebox.showinfo("Успех", f"Файл сохранен:\n{fn}")
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", str(e))

if __name__ == "__main__":
    app = ModernSceneApp()
    app.mainloop()