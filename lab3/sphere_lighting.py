import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

class SphereIllumination:
    """
    Класс для физически корректного расчета освещенности (Lambertian model).
    """
    def __init__(self, screen_width, screen_height, width_res, height_res,
                 light_sources, observer_pos, sphere_center, sphere_radius):
        self.W = screen_width
        self.H = screen_height
        self.Wres = width_res
        self.Hres = height_res
        self.light_sources = light_sources
        self.observer = np.array(observer_pos)
        self.sphere_center = np.array(sphere_center)
        self.sphere_radius = sphere_radius
        self.kd = 1.0  # Коэффициент диффузного отражения (белый матовый)

    def ray_sphere_intersection(self, ray_origin, ray_direction):
        """Вычисляет пересечение луча со сферой."""
        oc = ray_origin - self.sphere_center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.sphere_radius ** 2

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None, None

        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return None, None

        intersection = ray_origin + t * ray_direction
        return t, intersection

    def calculate_normal(self, point):
        return (point - self.sphere_center) / self.sphere_radius

    def diffuse_shading(self, point, normal, light_pos, light_intensity):
        """
        Расчет освещенности по Ламберту.
        """
        light_dir_vec = light_pos - point
        distance_mm = np.linalg.norm(light_dir_vec)

        if distance_mm < 1e-9:
            return 0

        # Нормализуем вектор света
        light_dir = light_dir_vec / distance_mm

        # Переводим расстояние в метры для закона обратных квадратов
        # (1 Lux = 1 Candela / 1 meter^2)
        distance_m = distance_mm / 1000.0 
        
        # Освещенность падающая (Incident)
        E_incident = light_intensity / (distance_m ** 2)

        # Учитываем угол падения (Lambert's Cosine Law)
        cos_theta = max(0, np.dot(normal, light_dir))
        
        # Итоговая яркость
        brightness = E_incident * cos_theta * self.kd

        return brightness

    def calculate_single_point(self, screen_x, screen_y):
        """Расчет яркость в конкретной точке экрана (x, y)."""
        screen_point = np.array([screen_x, screen_y, 0])
        ray_origin = self.observer
        ray_direction = screen_point - ray_origin
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        t, intersection = self.ray_sphere_intersection(ray_origin, ray_direction)

        if intersection is not None:
            normal = self.calculate_normal(intersection)
            view_dir = -ray_direction # Вектор к глазу
            
            # Проверка Back-face culling (видим ли мы эту точку)
            if np.dot(normal, view_dir) > 0:
                total_brightness = 0
                for light_source in self.light_sources:
                    light_pos = np.array(light_source[:3])
                    light_intensity = light_source[3]
                    
                    val = self.diffuse_shading(intersection, normal, light_pos, light_intensity)
                    total_brightness += val
                return total_brightness
        return 0.0

    def calculate_brightness_distribution(self):
        """Расчет матрицы яркости для всего экрана."""
        x_coords = np.linspace(-self.W / 2, self.W / 2, self.Wres)
        y_coords = np.linspace(-self.H / 2, self.H / 2, self.Hres)
        brightness_matrix = np.zeros((self.Hres, self.Wres))
        sphere_values = []

        # Двойной цикл по пикселям
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                val = self.calculate_single_point(x, y)
                brightness_matrix[i, j] = val
                
                # Собираем статистику только освещенных точек (приближенно)
                if val > 1e-4:
                    sphere_values.append(val)

        return brightness_matrix, np.array(sphere_values)

    def normalize_and_save(self, brightness_matrix):
        """Нормализация 0-255 для отображения."""
        max_brightness = np.max(brightness_matrix)
        if max_brightness > 0:
            normalized = (brightness_matrix / max_brightness * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(brightness_matrix, dtype=np.uint8)
        return normalized


class SphereVisualizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Расчет характеристик освещенности")
        self.geometry("1400x900")

        # --- Frames Layout ---
        control_frame = ttk.Frame(self, padding="15")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        viz_frame = ttk.Frame(right_panel)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.image_frame = ttk.LabelFrame(viz_frame, text="Визуализация", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.plot_frame = ttk.LabelFrame(viz_frame, text="Графики сечений", padding="10")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tables_frame = ttk.LabelFrame(right_panel, text="Таблицы данных", padding="10")
        self.tables_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=5)

        self.create_widgets(control_frame)
        self.create_tables_area()

    def create_widgets(self, parent):
        row = 0
        ttk.Label(parent, text="Экран (мм)", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.add_entry(parent, "Ширина:", "w_var", row, "500")
        row += 1
        self.add_entry(parent, "Высота:", "h_var", row, "500")
        row += 1
        
        ttk.Label(parent, text="Разрешение (px)", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10,0))
        row += 1
        self.add_entry(parent, "Ширина:", "wres_var", row, "300")
        row += 1
        self.add_entry(parent, "Высота:", "hres_var", row, "300")
        row += 1

        ttk.Label(parent, text="Источник (мм, Кд)", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10,0))
        row += 1
        self.add_entry(parent, "X:", "xl_var", row, "200")
        row += 1
        self.add_entry(parent, "Y:", "yl_var", row, "200")
        row += 1
        self.add_entry(parent, "Z:", "zl_var", row, "300")
        row += 1
        self.add_entry(parent, "Сила света (I0):", "i0_var", row, "500")
        row += 1

        ttk.Label(parent, text="Сфера (мм)", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10,0))
        row += 1
        self.add_entry(parent, "Смещение X:", "xc_var", row, "0")
        row += 1
        self.add_entry(parent, "Смещение Y:", "yc_var", row, "0")
        row += 1
        self.add_entry(parent, "Радиус:", "rad_var", row, "1000")
        row += 1

        calc_button = ttk.Button(parent, text="Рассчитать", command=self.run_calculation)
        calc_button.grid(row=row, column=0, columnspan=2, pady=30, sticky="ew")

    def add_entry(self, parent, label_text, var_name, row, default):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=5)
        var = tk.StringVar(value=default)
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky="e", padx=5)
        setattr(self, var_name, var)

    def create_tables_area(self):
        # Frame для таблицы точек
        frame_points = ttk.Frame(self.tables_frame)
        frame_points.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        ttk.Label(frame_points, text="3.2 Контрольные точки", font=("Arial", 9, "bold")).pack(anchor="w")
        
        cols_p = ("Point", "Coords", "Value")
        self.tree_points = ttk.Treeview(frame_points, columns=cols_p, show="headings", height=5)
        self.tree_points.heading("Point", text="Точка")
        self.tree_points.heading("Coords", text="Коорд (отн. R)")
        self.tree_points.heading("Value", text="Осв. (лк)")
        self.tree_points.column("Point", width=120)
        self.tree_points.column("Coords", width=120)
        self.tree_points.column("Value", width=100)
        self.tree_points.pack(fill=tk.BOTH, expand=True)

        # Frame для статистики
        frame_stats = ttk.Frame(self.tables_frame)
        frame_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        ttk.Label(frame_stats, text="3.3 Статистика", font=("Arial", 9, "bold")).pack(anchor="w")
        
        cols_s = ("Param", "Value")
        self.tree_stats = ttk.Treeview(frame_stats, columns=cols_s, show="headings", height=5)
        self.tree_stats.heading("Param", text="Параметр")
        self.tree_stats.heading("Value", text="Значение (лк)")
        self.tree_stats.pack(fill=tk.BOTH, expand=True)

    def run_calculation(self):
        try:
            # Чтение данных с формы
            W, H = float(self.w_var.get()), float(self.h_var.get())
            Wres, Hres = int(self.wres_var.get()), int(self.hres_var.get())
            
            ls = (float(self.xl_var.get()), float(self.yl_var.get()), float(self.zl_var.get()), float(self.i0_var.get()))
            
            sc_x, sc_y = float(self.xc_var.get()), float(self.yc_var.get())
            sc_z = 3000.0  # Фиксированная глубина сферы
            radius = float(self.rad_var.get())
            
            # Позиция наблюдателя (фиксирована)
            observer_pos = (0, 0, -500)

            # Инициализация модели
            model = SphereIllumination(
                W, H, Wres, Hres, 
                [ls], 
                observer_pos, 
                (sc_x, sc_y, sc_z), 
                radius
            )
            
            # Основной расчет
            brightness_matrix, sphere_values = model.calculate_brightness_distribution()
            normalized_img = model.normalize_and_save(brightness_matrix)

            # Обновление UI
            self.display_image(normalized_img)
            self.display_plots(brightness_matrix, W, H, sc_x, sc_y)
            # Передаем параметры для корректного расчета проекции
            self.update_tables(model, sphere_values, sc_x, sc_y, radius, sc_z, observer_pos[2])

        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте введенные числа")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Сбой: {e}")

    def update_tables(self, model, sphere_values, cx, cy, radius, sphere_z, observer_z):
        """
        Обновляет таблицы с учетом перспективной проекции.
        """
        # Очистка
        for i in self.tree_points.get_children(): self.tree_points.delete(i)
        for i in self.tree_stats.get_children(): self.tree_stats.delete(i)

        # --- РАСЧЕТ ПРОЕКЦИИ ---
        # Экран находится в Z=0
        screen_z = 0
        
        dist_to_sphere = sphere_z - observer_z  # 3000 - (-500) = 3500
        dist_to_screen = screen_z - observer_z  # 0 - (-500) = 500
        
        # Коэффициент масштабирования (насколько сфера "уменьшается" на экране)
        # s = 500 / 3500 = 1/7
        projection_scale = dist_to_screen / dist_to_sphere
        
        # Видимый радиус сферы на плоскости экрана
        projected_radius = radius * projection_scale
        
        # Смещение берем относительно ВИДИМОГО радиуса (например 30%)
        # Если взять 30% от реального радиуса (1000*0.3=300), это будет далеко за пределами
        # видимого диска (который всего ~142мм).
        offset = projected_radius * 0.3 

        # Список контрольных точек (координаты на ЭКРАНЕ)
        points = [
            ("Центр", cx, cy, "(0.0, 0.0)"),
            ("Ось X+", cx + offset, cy, "(0.3, 0.0)"),
            ("Ось X-", cx - offset, cy, "(-0.3, 0.0)"),
            ("Ось Y+", cx, cy + offset, "(0.0, 0.3)"),
            ("Ось Y-", cx, cy - offset, "(0.0, -0.3)"),
        ]

        for name, px, py, s_coord in points:
            val = model.calculate_single_point(px, py)
            self.tree_points.insert("", "end", values=(name, s_coord, f"{val:.2f}"))

        # Статистика
        if len(sphere_values) > 0:
            v_max = np.max(sphere_values)
            v_min = np.min(sphere_values)
            v_avg = np.mean(sphere_values)
        else:
            v_max, v_min, v_avg = 0.0, 0.0, 0.0

        self.tree_stats.insert("", "end", values=("Максимум", f"{v_max:.2f}"))
        self.tree_stats.insert("", "end", values=("Минимум", f"{v_min:.2f}"))
        self.tree_stats.insert("", "end", values=("Среднее", f"{v_avg:.2f}"))

    def display_image(self, image_array):
        for w in self.image_frame.winfo_children(): w.destroy()
        img = Image.fromarray(image_array, mode='L')
        img.thumbnail((350, 350))
        self.photo = ImageTk.PhotoImage(image=img)
        ttk.Label(self.image_frame, image=self.photo).pack(expand=True)

    def display_plots(self, matrix, W, H, cx, cy):
        for w in self.plot_frame.winfo_children(): w.destroy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5))
        Hres, Wres = matrix.shape
        x_ax = np.linspace(-W/2, W/2, Wres)
        y_ax = np.linspace(-H/2, H/2, Hres)

        idx_x = (np.abs(x_ax - cx)).argmin()
        idx_y = (np.abs(y_ax - cy)).argmin()

        ax1.plot(x_ax, matrix[idx_y, :], 'b-')
        ax1.set_title("Гор. сечение")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Лк")

        ax2.plot(y_ax, matrix[:, idx_x], 'r-')
        ax2.set_title("Верт. сечение")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel("Лк")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = SphereVisualizationApp()
    app.mainloop()