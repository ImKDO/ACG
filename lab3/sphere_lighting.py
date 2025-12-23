import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox


class SphereIllumination:
    """
    Класс для расчета распределения яркости на сфере.
    """

    def __init__(self, screen_width, screen_height, width_res, height_res,
                 light_sources, observer_pos, sphere_center, sphere_radius,
                 blinn_phong_params):
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
            blinn_phong_params: Параметры модели {'kd': ..., 'ks': ..., 'n': ...}
        """
        self.W = screen_width
        self.H = screen_height
        self.Wres = width_res
        self.Hres = height_res
        self.light_sources = light_sources
        self.observer = np.array(observer_pos)
        self.sphere_center = np.array(sphere_center)
        self.sphere_radius = sphere_radius
        self.kd = blinn_phong_params['kd']  # Коэффициент диффузного отражения
        self.ks = blinn_phong_params['ks']  # Коэффициент зеркального отражения
        self.n = blinn_phong_params['n']    # Показатель блеска

    def ray_sphere_intersection(self, ray_origin, ray_direction):
        """
        Вычисляет пересечение луча со сферой.

        Args:
            ray_origin: Точка начала луча (x, y, z)
            ray_direction: Направление луча (нормализованное)

        Returns:
            t: Параметр луча до точки пересечения (или None)
            intersection_point: Точка пересечения (или None)
        """
        # Вектор от начала луча до центра сферы
        oc = ray_origin - self.sphere_center

        # Коэффициенты квадратного уравнения
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.sphere_radius ** 2

        # Дискриминант
        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None, None

        # Берем ближайшее пересечение
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        # Выбираем положительное и ближайшее
        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return None, None

        intersection = ray_origin + t * ray_direction
        return t, intersection

    def calculate_normal(self, point):
        """
        Вычисляет нормаль к поверхности сферы в данной точке.

        Args:
            point: Точка на поверхности сферы

        Returns:
            normal: Нормализованный вектор нормали
        """
        normal = (point - self.sphere_center) / self.sphere_radius
        return normal

    def blinn_phong_shading(self, point, normal, light_pos, light_intensity):
        """
        Вычисляет яркость в точке по модели Блинн-Фонга с Ламбертовской диаграммой.

        Args:
            point: Точка на поверхности
            normal: Нормаль в этой точке
            light_pos: Позиция источника света
            light_intensity: Сила излучения I0 (Вт/ср)

        Returns:
            brightness: Яркость в точке
        """
        # Вектор к источнику света
        light_dir = light_pos - point
        distance = np.linalg.norm(light_dir)

        if distance < 1e-9:
            return 0

        light_dir = light_dir / distance

        # Ламбертовская диаграмма излучения: I(θ) = I0 * cos(θ)
        # где θ - угол между направлением на точку и нормалью источника
        # Направление на точку от источника
        point_dir = -light_dir  # От источника к точке

        # Предполагаем, что источник излучает вниз (по оси -Z)
        # или в направлении к сфере
        source_normal = (self.sphere_center - light_pos)
        if np.linalg.norm(source_normal) > 1e-9:
            source_normal = source_normal / np.linalg.norm(source_normal)
        else:
            source_normal = np.array([0, 0, -1])

        # Косинус угла для Ламбертовской диаграммы
        cos_theta_source = max(0, np.dot(point_dir, source_normal))

        # Интенсивность с учетом Ламбертовской диаграммы
        effective_intensity = light_intensity * \
            cos_theta_source / (distance ** 2)

        # Диффузная составляющая
        cos_theta = max(0, np.dot(normal, light_dir))
        diffuse = self.kd * effective_intensity * cos_theta

        # Направление к наблюдателю
        view_dir = self.observer - point
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Вектор половинного направления (Blinn-Phong)
        half_vector = light_dir + view_dir
        half_norm = np.linalg.norm(half_vector)

        if half_norm > 1e-9:
            half_vector = half_vector / half_norm

            # Зеркальная составляющая
            cos_alpha = max(0, np.dot(normal, half_vector))
            specular = self.ks * effective_intensity * (cos_alpha ** self.n)
        else:
            specular = 0

        brightness = diffuse + specular
        return brightness

    def calculate_brightness_distribution(self):
        """
        Вычисляет распределение яркости на экране.

        Returns:
            brightness_matrix: Матрица яркости (Hres x Wres)
        """
        # Создаем сетку координат экрана
        x_coords = np.linspace(-self.W / 2, self.W / 2, self.Wres)
        y_coords = np.linspace(-self.H / 2, self.H / 2, self.Hres)

        brightness_matrix = np.zeros((self.Hres, self.Wres))

        # Для каждого пикселя
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                # Точка на экране (z = 0)
                screen_point = np.array([x, y, 0])

                # Луч от наблюдателя через точку на экране
                ray_origin = self.observer
                ray_direction = screen_point - ray_origin
                ray_direction = ray_direction / np.linalg.norm(ray_direction)

                # Проверяем пересечение со сферой
                t, intersection = self.ray_sphere_intersection(
                    ray_origin, ray_direction)

                if intersection is not None:
                    # Вычисляем нормаль
                    normal = self.calculate_normal(intersection)

                    # Проверяем, что точка видна наблюдателю (нормаль направлена к нему)
                    view_dir = self.observer - intersection
                    view_dir = view_dir / np.linalg.norm(view_dir)

                    if np.dot(normal, view_dir) > 0:
                        # Суммируем вклад всех источников света
                        total_brightness = 0

                        for light_source in self.light_sources:
                            light_pos = np.array(light_source[:3])
                            light_intensity = light_source[3]

                            # Проверяем, не находится ли источник в тени
                            # (для упрощения не учитываем самозатенение)
                            brightness = self.blinn_phong_shading(
                                intersection, normal, light_pos, light_intensity
                            )
                            total_brightness += brightness

                        brightness_matrix[i, j] = total_brightness

        return brightness_matrix

    def normalize_and_save(self, brightness_matrix, filename="sphere_brightness.png"):
        """
        Нормализует яркость к диапазону 0-255 и сохраняет изображение.

        Args:
            brightness_matrix: Матрица яркости
            filename: Имя файла для сохранения

        Returns:
            normalized_image: Нормализованное изображение (0-255)
        """
        max_brightness = np.max(brightness_matrix)

        if max_brightness > 0:
            normalized = (brightness_matrix /
                          max_brightness * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(brightness_matrix, dtype=np.uint8)

        # Создаем изображение
        img = Image.fromarray(normalized, mode='L')
        img.save(filename)

        print(f"Изображение сохранено: {filename}")
        print(f"Максимальная яркость: {max_brightness:.2e}")
        print(f"Разрешение: {self.Wres}x{self.Hres}")

        return normalized


class SphereVisualizationApp(tk.Tk):
    """
    GUI приложение для интерактивной визуализации.
    """

    def __init__(self):
        super().__init__()
        self.title("Распределение яркости на сфере (Блинн-Фонг)")
        self.geometry("1400x900")

        # Панель управления
        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Область для изображения
        self.image_frame = ttk.Frame(self, padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Область для графиков
        self.plot_frame = ttk.Frame(self, padding="10")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.create_widgets(control_frame)

    def create_widgets(self, parent):
        """Создает элементы управления."""
        row = 0

        # Параметры экрана
        ttk.Label(parent, text="Параметры экрана (мм):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1

        self.add_entry(parent, "Ширина (W):", "w_entry", row, "500")
        row += 1
        self.add_entry(parent, "Высота (H):", "h_entry", row, "500")
        row += 1

        # Разрешение
        ttk.Label(parent, text="Разрешение (пиксели):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1

        self.add_entry(parent, "Ширина (Wres):", "wres_entry", row, "400")
        row += 1
        self.add_entry(parent, "Высота (Hres):", "hres_entry", row, "400")
        row += 1

        # Источник света 1
        ttk.Label(parent, text="Источник света 1 (мм, Вт/ср):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1

        self.add_entry(parent, "xL1:", "xl1_entry", row, "200")
        row += 1
        self.add_entry(parent, "yL1:", "yl1_entry", row, "200")
        row += 1
        self.add_entry(parent, "zL1:", "zl1_entry", row, "300")
        row += 1
        self.add_entry(parent, "I0_1:", "i01_entry", row, "500")
        row += 1

        # Наблюдатель
        ttk.Label(parent, text="Наблюдатель (мм):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1

        self.add_entry(parent, "zO:", "zo_entry", row, "-500")
        row += 1

        # Сфера
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

        # Окружность-маска (ограничение области)
        ttk.Label(parent, text="Окружность-маска (мм):", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1

        self.add_entry(parent, "Центр X маски:", "mask_x_entry", row, "500")
        row += 1
        self.add_entry(parent, "Центр Y маски:", "mask_y_entry", row, "500")
        row += 1
        self.add_entry(parent, "Радиус маски:\n(0 = без маски):",
                       "mask_r_entry", row, "1500")
        row += 1

        # Параметры Блинн-Фонга
        ttk.Label(parent, text="Модель Блинн-Фонга:", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1

        self.add_entry(parent, "kd (диффузный):", "kd_entry", row, "0.6")
        row += 1
        self.add_entry(parent, "ks (зеркальный):", "ks_entry", row, "0.8")
        row += 1
        self.add_entry(parent, "n (блеск):", "n_entry", row, "50")
        row += 1

        # Кнопка расчета
        calc_button = ttk.Button(
            parent, text="Рассчитать и визуализировать",
            command=self.run_calculation
        )
        calc_button.grid(row=row, column=0, columnspan=2, pady=20)

    def add_entry(self, parent, label_text, attr_name, row, default_value):
        """Добавляет поле ввода."""
        ttk.Label(parent, text=label_text).grid(
            row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(parent, width=15, textvariable=var)
        entry.grid(row=row, column=1, sticky="e", padx=5, pady=2)
        setattr(self, attr_name, entry)
        setattr(self, attr_name.replace('_entry', '_var'), var)

    def run_calculation(self):
        """Запускает расчет и визуализацию."""
        try:
            # Считываем параметры
            W = float(self.w_var.get())
            H = float(self.h_var.get())
            Wres = int(self.wres_var.get())
            Hres = int(self.hres_var.get())

            xL1 = float(self.xl1_var.get())
            yL1 = float(self.yl1_var.get())
            zL1 = float(self.zl1_var.get())
            I0_1 = float(self.i01_var.get())

            zO = float(self.zo_var.get())

            xC = float(self.xc_var.get())
            yC = float(self.yc_var.get())
            zC = float(self.zc_var.get())
            radius = float(self.radius_var.get())

            kd = float(self.kd_var.get())
            ks = float(self.ks_var.get())
            n = float(self.n_var.get())

            mask_x = float(self.mask_x_var.get())
            mask_y = float(self.mask_y_var.get())
            mask_r = float(self.mask_r_var.get())

            # Создаем объект
            light_sources = [(xL1, yL1, zL1, I0_1)]
            observer = (0, 0, zO)
            sphere_center = (xC, yC, zC)
            blinn_phong = {'kd': kd, 'ks': ks, 'n': n}

            sphere_illum = SphereIllumination(
                W, H, Wres, Hres, light_sources, observer,
                sphere_center, radius, blinn_phong
            )

            # Расчет
            print("Начинается расчет распределения яркости...")
            brightness = sphere_illum.calculate_brightness_distribution()

            # Применяем маску если радиус > 0
            if mask_r > 0:
                brightness = self.apply_circular_mask(
                    brightness, W, H, mask_x, mask_y, mask_r)

            # Нормализация и сохранение
            normalized = sphere_illum.normalize_and_save(brightness)

            # Визуализация
            self.display_image(normalized)
            self.display_plots(brightness, W, H, xC, yC)

            messagebox.showinfo("Успех", "Расчет завершен!")

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверные входные данные: {e}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

    def apply_circular_mask(self, brightness, W, H, center_x, center_y, radius):
        """Применяет круговую маску к изображению яркости."""
        Hres, Wres = brightness.shape
        y_coords = np.linspace(-H/2, H/2, Hres)
        x_coords = np.linspace(-W/2, W/2, Wres)
        xv, yv = np.meshgrid(x_coords, y_coords)

        # Расстояние от каждой точки до центра маски
        dist_sq = (xv - center_x)**2 + (yv - center_y)**2

        # Маска: True внутри круга, False снаружи
        mask = dist_sq <= radius**2

        # Применяем маску (обнуляем яркость вне круга)
        return brightness * mask

    def display_image(self, image_array):
        """Отображает изображение."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        img = Image.fromarray(image_array, mode='L')

        # Масштабирование для отображения
        max_size = 400
        img.thumbnail((max_size, max_size), Image.Resampling.NEAREST)

        self.photo = ImageTk.PhotoImage(image=img)

        ttk.Label(self.image_frame, text="Распределение яркости на сфере",
                  font=("Arial", 12, "bold")).pack(pady=5)
        label = ttk.Label(self.image_frame, image=self.photo)
        label.pack()
        ttk.Label(self.image_frame,
                  text="Изображение сохранено: sphere_brightness.png").pack(pady=5)

    def display_plots(self, brightness, W, H, sphere_x, sphere_y):
        """Отображает графики сечений через центр сферы."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
        fig.tight_layout(pad=3.0)

        Hres, Wres = brightness.shape

        # Координаты экрана
        x_coords = np.linspace(-W/2, W/2, Wres)
        y_coords = np.linspace(-H/2, H/2, Hres)

        # Находим индекс пикселя, соответствующий центру сферы
        center_x_idx = np.argmin(np.abs(x_coords - sphere_x))
        center_y_idx = np.argmin(np.abs(y_coords - sphere_y))

        # Горизонтальное сечение через центр сферы (Y = sphere_y)
        horizontal = brightness[center_y_idx, :]
        ax1.plot(x_coords, horizontal)
        ax1.set_title(f"Горизонтальное сечение (Y={sphere_y:.0f} мм)")
        ax1.set_xlabel("X (мм)")
        ax1.set_ylabel("Яркость")
        ax1.axvline(x=sphere_x, color='r', linestyle='--',
                    alpha=0.5, label='Центр сферы')
        ax1.legend()
        ax1.grid(True)

        # Вертикальное сечение через центр сферы (X = sphere_x)
        vertical = brightness[:, center_x_idx]
        ax2.plot(y_coords, vertical)
        ax2.set_title(f"Вертикальное сечение (X={sphere_x:.0f} мм)")
        ax2.set_xlabel("Y (мм)")
        ax2.set_ylabel("Яркость")
        ax2.axvline(x=sphere_y, color='r', linestyle='--',
                    alpha=0.5, label='Центр сферы')
        ax2.legend()
        ax2.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def main():
    """Основная функция с примером использования."""

    # Пример параметров
    W = 500  # мм
    H = 500  # мм
    Wres = 400
    Hres = 400

    # Источники света: (x, y, z, I0)
    light_sources = [
        (200, 200, 300, 500),  # Источник справа-сверху
    ]

    # Наблюдатель
    observer = (0, 0, -500)

    # Сфера
    sphere_center = (0, 0, 300)
    sphere_radius = 100

    # Параметры Блинн-Фонга
    blinn_phong_params = {
        'kd': 0.6,  # Диффузный коэффициент
        'ks': 0.8,  # Зеркальный коэффициент
        'n': 50     # Показатель блеска (чем больше, тем острее блик)
    }

    # Создаем объект
    sphere_illum = SphereIllumination(
        W, H, Wres, Hres, light_sources, observer,
        sphere_center, sphere_radius, blinn_phong_params
    )

    # Расчет распределения яркости
    print("Расчет распределения яркости...")
    brightness_matrix = sphere_illum.calculate_brightness_distribution()

    # Нормализация и сохранение
    normalized_image = sphere_illum.normalize_and_save(brightness_matrix)

    # Визуализация
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(normalized_image, cmap='gray', origin='lower')
    plt.title('Распределение яркости')
    plt.colorbar(label='Яркость (0-255)')

    plt.subplot(1, 3, 2)
    # Сечение через центр сферы
    x_coords = np.linspace(-W/2, W/2, Wres)
    y_coords = np.linspace(-H/2, H/2, Hres)
    center_x_idx = np.argmin(np.abs(x_coords - sphere_center[0]))
    center_y_idx = np.argmin(np.abs(y_coords - sphere_center[1]))

    plt.plot(x_coords, brightness_matrix[center_y_idx, :])
    plt.title(f'Горизонтальное сечение (Y={sphere_center[1]:.0f} мм)')
    plt.xlabel('X (мм)')
    plt.ylabel('Яркость')
    plt.axvline(x=sphere_center[0], color='r',
                linestyle='--', alpha=0.5, label='Центр сферы')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(y_coords, brightness_matrix[:, center_x_idx])
    plt.title(f'Вертикальное сечение (X={sphere_center[0]:.0f} мм)')
    plt.xlabel('Y (мм)')
    plt.ylabel('Яркость')
    plt.axvline(x=sphere_center[1], color='r',
                linestyle='--', alpha=0.5, label='Центр сферы')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('sphere_analysis.png', dpi=150)
    print("График анализа сохранен: sphere_analysis.png")
    plt.show()


if __name__ == "__main__":
    # Можно запустить либо с GUI, либо без
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--nogui":
        # Запуск без GUI
        main()
    else:
        # Запуск с GUI
        app = SphereVisualizationApp()
        app.mainloop()
