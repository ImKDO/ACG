"""
Расчет и визуализация распределения яркости на диффузной сфере,
освещенной точечными источниками света (только диффузная модель).

Все расчеты ведутся в миллиметрах (мм).
"""

import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox


class SphereIllumination:
    """
    Класс для расчета распределения яркости на сфере (Диффузная модель).
    """

    def __init__(self, screen_width, screen_height, width_res, height_res,
                 light_sources, observer_pos, sphere_center, sphere_radius,
                 kd):
        """
        Инициализация параметров.

        Args:
            screen_width: Ширина экрана (мм)
            screen_height: Высота экрана (мм)
            width_res: Разрешение по ширине (пиксели)
            height_res: Разрешение по высоте (пиксели)
            light_sources: Список источников света [(x, y, z, I0), ...]
            observer_pos: Позиция наблюдателя (x, y, z)
            sphere_center: Центр сферы (x, y, z)
            sphere_radius: Радиус сферы (мм)
            kd: Коэффициент диффузного отражения
        """
        self.W = screen_width
        self.H = screen_height
        self.Wres = width_res
        self.Hres = height_res
        self.light_sources = light_sources
        self.observer = np.array(observer_pos)
        self.sphere_center = np.array(sphere_center)
        self.sphere_radius = sphere_radius
        self.kd = kd  # Коэффициент диффузного отражения

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
        """Вычисляет нормаль к поверхности сферы."""
        return (point - self.sphere_center) / self.sphere_radius

    def diffuse_shading(self, point, normal, light_pos, light_intensity):
        """
        Вычисляет яркость в точке по диффузной модели (Ламберт).
        """
        light_dir = light_pos - point
        distance = np.linalg.norm(light_dir)

        if distance < 1e-9:
            return 0

        light_dir = light_dir / distance

        # Ламбертовская диаграмма излучения самого источника
        point_dir = -light_dir
        source_normal = (self.sphere_center - light_pos)
        if np.linalg.norm(source_normal) > 1e-9:
            source_normal = source_normal / np.linalg.norm(source_normal)
        else:
            source_normal = np.array([0, 0, -1])

        cos_theta_source = max(0, np.dot(point_dir, source_normal))
        effective_intensity = light_intensity * \
            cos_theta_source / (distance ** 2)

        # Диффузная составляющая (закон Ламберта на поверхности)
        cos_theta = max(0, np.dot(normal, light_dir))
        brightness = self.kd * effective_intensity * cos_theta

        return brightness

    def calculate_brightness_distribution(self):
        """Вычисляет распределение яркости на экране."""
        x_coords = np.linspace(-self.W / 2, self.W / 2, self.Wres)
        y_coords = np.linspace(-self.H / 2, self.H / 2, self.Hres)
        brightness_matrix = np.zeros((self.Hres, self.Wres))

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                screen_point = np.array([x, y, 0])
                ray_origin = self.observer
                ray_direction = screen_point - ray_origin
                ray_direction = ray_direction / np.linalg.norm(ray_direction)

                t, intersection = self.ray_sphere_intersection(
                    ray_origin, ray_direction)

                if intersection is not None:
                    normal = self.calculate_normal(intersection)
                    view_dir = self.observer - intersection
                    view_dir = view_dir / np.linalg.norm(view_dir)

                    if np.dot(normal, view_dir) > 0:
                        total_brightness = 0
                        for light_source in self.light_sources:
                            light_pos = np.array(light_source[:3])
                            light_intensity = light_source[3]
                            brightness = self.diffuse_shading(
                                intersection, normal, light_pos, light_intensity
                            )
                            total_brightness += brightness
                        brightness_matrix[i, j] = total_brightness

        return brightness_matrix

    def normalize_and_save(self, brightness_matrix, filename="sphere_brightness.png"):
        """Нормализует яркость и сохраняет изображение."""
        max_brightness = np.max(brightness_matrix)
        if max_brightness > 0:
            normalized = (brightness_matrix /
                          max_brightness * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(brightness_matrix, dtype=np.uint8)

        img = Image.fromarray(normalized, mode='L')
        img.save(filename)
        return normalized


class SphereVisualizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Распределение яркости на диффузной сфере")
        self.geometry("1400x900")

        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.image_frame = ttk.Frame(self, padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_frame = ttk.Frame(self, padding="10")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.create_widgets(control_frame)

    def create_widgets(self, parent):
        row = 0
        ttk.Label(parent, text="Параметры экрана (мм):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1
        self.add_entry(parent, "Ширина (W):", "w_entry", row, "500")
        row += 1
        self.add_entry(parent, "Высота (H):", "h_entry", row, "500")
        row += 1

        ttk.Label(parent, text="Разрешение (пиксели):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1
        self.add_entry(parent, "Ширина (Wres):", "wres_entry", row, "400")
        row += 1
        self.add_entry(parent, "Высота (Hres):", "hres_entry", row, "400")
        row += 1

        ttk.Label(parent, text="Источник света (мм, Вт/ср):", font=("Arial",
                  11, "bold")).grid(row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1
        self.add_entry(parent, "xL1:", "xl1_entry", row, "200")
        row += 1
        self.add_entry(parent, "yL1:", "yl1_entry", row, "200")
        row += 1
        self.add_entry(parent, "zL1:", "zl1_entry", row, "300")
        row += 1
        self.add_entry(parent, "I0_1:", "i01_entry", row, "500")
        row += 1

        ttk.Label(parent, text="Сфера (мм):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1
        self.add_entry(parent, "xC (центр):", "xc_entry", row, "0")
        row += 1
        self.add_entry(parent, "yC (центр):", "yc_entry", row, "0")
        row += 1
        self.add_entry(parent, "zC (центр):", "zc_entry", row, "3000")
        row += 1
        self.add_entry(parent, "Радиус:", "radius_entry", row, "1000")
        row += 1

        ttk.Label(parent, text="Свойства поверхности:", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1
        self.add_entry(parent, "kd (диффузный):", "kd_entry", row, "0.8")
        row += 1

        calc_button = ttk.Button(
            parent, text="Рассчитать", command=self.run_calculation)
        calc_button.grid(row=row, column=0, columnspan=2, pady=20)

    def add_entry(self, parent, label_text, attr_name, row, default_value):
        ttk.Label(parent, text=label_text).grid(
            row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(parent, width=15, textvariable=var)
        entry.grid(row=row, column=1, sticky="e", padx=5, pady=2)
        setattr(self, attr_name, entry)
        setattr(self, attr_name.replace('_entry', '_var'), var)

    def run_calculation(self):
        try:
            W = float(self.w_var.get())
            H = float(self.h_var.get())
            Wres = int(self.wres_var.get())
            Hres = int(self.hres_var.get())
            light_sources = [(float(self.xl1_var.get()), float(self.yl1_var.get()), float(
                self.zl1_var.get()), float(self.i01_var.get()))]
            sphere_center = (float(self.xc_var.get()), float(
                self.yc_var.get()), float(self.zc_var.get()))
            radius = float(self.radius_var.get())
            kd = float(self.kd_var.get())

            sphere_illum = SphereIllumination(
                W, H, Wres, Hres, light_sources, (0, 0, -500), sphere_center, radius, kd)
            brightness = sphere_illum.calculate_brightness_distribution()
            normalized = sphere_illum.normalize_and_save(brightness)

            self.display_image(normalized)
            self.display_plots(
                brightness, W, H, sphere_center[0], sphere_center[1])
            messagebox.showinfo("Успех", "Расчет завершен!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

    def display_image(self, image_array):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        img = Image.fromarray(image_array, mode='L')
        img.thumbnail((400, 400), Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(image=img)
        ttk.Label(self.image_frame, text="Распределение яркости (Диффузное)", font=(
            "Arial", 12, "bold")).pack(pady=5)
        ttk.Label(self.image_frame, image=self.photo).pack()

    def display_plots(self, brightness, W, H, sphere_x, sphere_y):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
        Hres, Wres = brightness.shape
        x_coords = np.linspace(-W/2, W/2, Wres)
        y_coords = np.linspace(-H/2, H/2, Hres)

        idx_x = np.argmin(np.abs(x_coords - sphere_x))
        idx_y = np.argmin(np.abs(y_coords - sphere_y))

        ax1.plot(x_coords, brightness[idx_y, :])
        ax1.set_title("Горизонтальное сечение")
        ax2.plot(y_coords, brightness[:, idx_x])
        ax2.set_title("Вертикальное сечение")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = SphereVisualizationApp()
    app.mainloop()
