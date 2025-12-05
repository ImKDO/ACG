# import numpy as np
# from PIL import Image, ImageTk
# import tkinter as tk
# from tkinter import ttk, messagebox, filedialog
#
#
# # =====================================================================
# # ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =======================
# # =====================================================================
#
# def ray_sphere_intersect(O, D, C, R):
#     """
#     Находит ближайшее положительное пересечение луча O + t*D со сферой (C, R).
#     Возвращает: t (массив), hit_mask (boolean)
#     """
#     OC = O - C
#     a = np.sum(D * D, axis=-1)
#     b = 2.0 * np.sum(D * OC, axis=-1)
#     c = np.sum(OC * OC, axis=-1) - R ** 2
#     disc = b * b - 4 * a * c
#     hit = disc >= 0
#     t = np.full_like(a, np.inf)
#     sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
#     denom = 2 * a + 1e-8
#     t1 = (-b - sqrt_disc) / denom
#     t2 = (-b + sqrt_disc) / denom
#     t_candidate = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
#     t = np.where((t_candidate > 0) & hit, t_candidate, np.inf)
#     return t, t != np.inf
#
#
# def render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights):
#     """
#     Рендерит сцену с двумя сферами, тенями и цветом по модели Блинна–Фонга.
#     """
#     O = np.array([0.0, 0.0, zO], dtype=np.float64)
#
#     xs = np.linspace(-W / 2, W / 2, Wres, endpoint=False) + W / (2 * Wres)
#     ys = np.linspace(-H / 2, H / 2, Hres, endpoint=False) + H / (2 * Hres)
#     X, Y = np.meshgrid(xs, ys)
#     Px = X.ravel()
#     Py = Y.ravel()
#     Pz = np.full_like(Px, z_scr)
#     P_screen = np.stack([Px, Py, Pz], axis=1)
#
#     D = P_screen - O
#     D_norm = np.linalg.norm(D, axis=1, keepdims=True)
#     D = D / np.maximum(D_norm, 1e-8)
#
#     depth = np.full(P_screen.shape[0], np.inf, dtype=np.float64)
#     sphere_id = np.full(P_screen.shape[0], -1, dtype=int)
#
#     for i, sph in enumerate(spheres):
#         C = np.array(sph['center'], dtype=np.float64)
#         R = sph['radius']
#         t, hit = ray_sphere_intersect(O, D, C, R)
#         closer = t < depth
#         mask = hit & closer
#         depth[mask] = t[mask]
#         sphere_id[mask] = i
#
#     P = O + D * depth[:, np.newaxis]
#     V = O - P
#     V_norm = np.linalg.norm(V, axis=1, keepdims=True)
#     V = V / np.maximum(V_norm, 1e-8)
#
#     final_color = np.zeros((P_screen.shape[0], 3), dtype=np.float64)
#     valid = sphere_id >= 0
#
#     if not np.any(valid):
#         img_rgb = np.zeros((Hres, Wres, 3))
#         return (np.zeros((Hres, Wres, 3), dtype=np.uint8), img_rgb, 0.0, 0.0)
#
#     N = np.zeros_like(P)
#     centers_arr = np.array([spheres[i]['center'] for i in sphere_id[valid]], dtype=np.float64)
#     N[valid] = P[valid] - centers_arr
#     N_norm = np.linalg.norm(N[valid], axis=1, keepdims=True)
#     N[valid] /= np.maximum(N_norm, 1e-8)
#
#     for idx in np.where(valid)[0]:
#         sph = spheres[sphere_id[idx]]
#         point = P[idx]
#         normal = N[idx]
#         color_acc = np.zeros(3)
#
#         for light in lights:
#             L_pos = np.array(light['pos'], dtype=np.float64)
#             L_dir_vec = L_pos - point
#             L_dist = np.linalg.norm(L_dir_vec)
#             if L_dist < 1e-6:
#                 continue
#             L_dir = L_dir_vec / L_dist
#             light_color = np.array(light['color'], dtype=np.float64)
#             I0 = light['I0']
#
#             # Проверка тени: смещаем точку чуть вдоль нормали
#             shadow_ray_origin = point + normal * 1e-3
#             shadow = False
#             for other_sph in spheres:
#                 C_other = np.array(other_sph['center'], dtype=np.float64)
#                 R_other = other_sph['radius']
#                 t_shadow, hit_shadow = ray_sphere_intersect(shadow_ray_origin, L_dir, C_other, R_other)
#                 if np.any(hit_shadow) and np.min(t_shadow[hit_shadow]) < L_dist - 1e-3:
#                     shadow = True
#                     break
#
#             if shadow:
#                 continue
#
#             # Диффузная компонента (Ламберт)
#             diff = max(np.dot(normal, L_dir), 0.0)
#             I_diff = sph['kd'] * I0 * diff
#
#             # Зеркальная компонента (Блинн–Фонг)
#             H = L_dir + V[idx]
#             H_norm_val = np.linalg.norm(H)
#             if H_norm_val < 1e-8:
#                 spec = 0.0
#             else:
#                 H = H / H_norm_val
#                 spec = max(np.dot(normal, H), 0.0) ** sph['shininess']
#             I_spec = sph['ks'] * I0 * spec
#
#             contrib = (I_diff + I_spec)
#             color_acc += contrib * light_color * np.array(sph['color'])
#
#         final_color[idx] = color_acc
#
#     img_rgb = final_color.reshape((Hres, Wres, 3))
#
#     # Максимальная и минимальная ненулевая яркость
#     I_max = img_rgb.max()
#     I_min = 0.0
#     if I_max > 0:
#         non_zero = img_rgb[img_rgb > 0]
#         if len(non_zero) > 0:
#             I_min = non_zero.min()
#
#     if I_max > 0:
#         img_norm = (img_rgb / I_max) * 255
#     else:
#         img_norm = img_rgb
#     img_uint8 = np.clip(img_norm, 0, 255).astype(np.uint8)
#
#     return img_uint8, img_rgb, I_max, I_min
#
#
# # =====================================================================
# # ============================= GUI ЧАСТЬ ==============================
# # =====================================================================
#
# class SceneApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("ЛР5: Две сферы с тенями и цветом (Блинн–Фонг)")
#         self.geometry("1250x900")
#
#         main_frame = ttk.Frame(self)
#         main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
#
#         # Левая панель: изображение
#         self.image_label = ttk.Label(main_frame, relief="sunken", background="black")
#         self.image_label.grid(row=0, column=0, rowspan=3, padx=(0, 15), sticky="nw")
#
#         # Правая панель: параметры
#         params_frame = ttk.Frame(main_frame)
#         params_frame.grid(row=0, column=1, sticky="nw")
#
#         # === Экран и наблюдатель ===
#         cam_frame = ttk.LabelFrame(params_frame, text="Экран и наблюдатель", padding=8)
#         cam_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
#
#         self.W_var = self._add_param(cam_frame, "Ширина экрана W [мм]", 800, 0)
#         self.H_var = self._add_param(cam_frame, "Высота экрана H [мм]", 600, 1)
#         self.Wres_var = self._add_param(cam_frame, "Разрешение Wres [пикс]", 400, 2)
#         self.zscr_var = self._add_param(cam_frame, "z экрана z_scr [мм]", 0, 3)
#         self.zO_var = self._add_param(cam_frame, "z наблюдателя zO [мм]", -1000, 4)
#
#         # === Сфера 1 ===
#         sph1_frame = ttk.LabelFrame(params_frame, text="Сфера 1", padding=8)
#         sph1_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
#
#         self.R1_var = self._add_param(sph1_frame, "Радиус R1 [мм]", 200, 0)
#         self.C1x_var = self._add_param(sph1_frame, "Cx1 [мм]", -150, 1)
#         self.C1y_var = self._add_param(sph1_frame, "Cy1 [мм]", 0, 2)
#         self.C1z_var = self._add_param(sph1_frame, "Cz1 [мм]", 800, 3)
#         self.kd1_var = self._add_param(sph1_frame, "kd1 (диффузия)", 0.7, 4)
#         self.ks1_var = self._add_param(sph1_frame, "ks1 (зеркальность)", 0.3, 5)
#         self.sh1_var = self._add_param(sph1_frame, "shininess1", 30, 6)
#         self.C1r_var = self._add_param(sph1_frame, "Цвет R1", 1.0, 7)
#         self.C1g_var = self._add_param(sph1_frame, "Цвет G1", 0.2, 8)
#         self.C1b_var = self._add_param(sph1_frame, "Цвет B1", 0.2, 9)
#
#         # === Сфера 2 ===
#         sph2_frame = ttk.LabelFrame(params_frame, text="Сфера 2", padding=8)
#         sph2_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
#
#         self.R2_var = self._add_param(sph2_frame, "Радиус R2 [мм]", 180, 0)
#         self.C2x_var = self._add_param(sph2_frame, "Cx2 [мм]", 200, 1)
#         self.C2y_var = self._add_param(sph2_frame, "Cy2 [мм]", 100, 2)
#         self.C2z_var = self._add_param(sph2_frame, "Cz2 [мм]", 900, 3)
#         self.kd2_var = self._add_param(sph2_frame, "kd2", 0.6, 4)
#         self.ks2_var = self._add_param(sph2_frame, "ks2", 0.4, 5)
#         self.sh2_var = self._add_param(sph2_frame, "shininess2", 50, 6)
#         self.C2r_var = self._add_param(sph2_frame, "Цвет R2", 0.2, 7)
#         self.C2g_var = self._add_param(sph2_frame, "Цвет G2", 0.8, 8)
#         self.C2b_var = self._add_param(sph2_frame, "Цвет B2", 0.2, 9)
#
#         # === Источники света ===
#         light_frame = ttk.LabelFrame(params_frame, text="Источники света", padding=8)
#         light_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nw")
#
#         # Источник 1
#         ttk.Label(light_frame, text="Источник 1", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
#         self.L1x_var = self._add_param(light_frame, "L1x [мм]", 2000, 1)
#         self.L1y_var = self._add_param(light_frame, "L1y [мм]", 1500, 2)
#         self.L1z_var = self._add_param(light_frame, "L1z [мм]", -500, 3)
#         self.I1_var = self._add_param(light_frame, "I01 [Вт/ср]", 800, 4)
#         self.LC1r_var = self._add_param(light_frame, "Цвет L1 (R)", 1.0, 5)
#         self.LC1g_var = self._add_param(light_frame, "Цвет L1 (G)", 1.0, 6)
#         self.LC1b_var = self._add_param(light_frame, "Цвет L1 (B)", 1.0, 7)
#
#         ttk.Separator(light_frame, orient="horizontal").grid(row=8, column=0, columnspan=2, sticky="ew", pady=6)
#
#         # Источник 2
#         ttk.Label(light_frame, text="Источник 2", font=("TkDefaultFont", 10, "bold")).grid(row=9, column=0, columnspan=2, sticky="w")
#         self.L2x_var = self._add_param(light_frame, "L2x [мм]", -1000, 10)
#         self.L2y_var = self._add_param(light_frame, "L2y [мм]", -1000, 11)
#         self.L2z_var = self._add_param(light_frame, "L2z [мм]", -800, 12)
#         self.I2_var = self._add_param(light_frame, "I02 [Вт/ср]", 300, 13)
#         self.LC2r_var = self._add_param(light_frame, "Цвет L2 (R)", 1.0, 14)
#         self.LC2g_var = self._add_param(light_frame, "Цвет L2 (G)", 0.8, 15)
#         self.LC2b_var = self._add_param(light_frame, "Цвет L2 (B)", 0.5, 16)
#
#         # === Кнопки и информация ===
#         control_frame = ttk.Frame(main_frame)
#         control_frame.grid(row=1, column=1, sticky="sw", pady=(10, 0))
#
#         self.info_var = tk.StringVar(value="Max = 0.0, Min>0 = 0.0")
#         info_label = ttk.Label(control_frame, textvariable=self.info_var, foreground="blue", font=("TkDefaultFont", 10, "bold"))
#         info_label.pack(anchor="w", pady=5)
#
#         btn_render = ttk.Button(control_frame, text="Render", command=self.render, width=20)
#         btn_render.pack(pady=5)
#
#         btn_save = ttk.Button(control_frame, text="Save image", command=self.save, width=20)
#         btn_save.pack(pady=5)
#
#         self.last_pil = None
#         self.render()  # первый рендер
#
#     def _add_param(self, parent, label_text, default_value, row):
#         """Вспомогательная функция для создания пары метка–поле ввода."""
#         ttk.Label(parent, text=label_text, width=22, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 5))
#         var = tk.StringVar(value=str(default_value))
#         entry = ttk.Entry(parent, textvariable=var, width=10)
#         entry.grid(row=row, column=1, sticky="w")
#         return var
#
#     def render(self):
#         try:
#             W = float(self.W_var.get())
#             H = float(self.H_var.get())
#             Wres = int(self.Wres_var.get())
#             Hres = Wres  # квадратные пиксели
#             z_scr = float(self.zscr_var.get())
#             zO = float(self.zO_var.get())
#
#             spheres = [
#                 {
#                     'center': (float(self.C1x_var.get()), float(self.C1y_var.get()), float(self.C1z_var.get())),
#                     'radius': float(self.R1_var.get()),
#                     'kd': float(self.kd1_var.get()),
#                     'ks': float(self.ks1_var.get()),
#                     'shininess': float(self.sh1_var.get()),
#                     'color': (float(self.C1r_var.get()), float(self.C1g_var.get()), float(self.C1b_var.get()))
#                 },
#                 {
#                     'center': (float(self.C2x_var.get()), float(self.C2y_var.get()), float(self.C2z_var.get())),
#                     'radius': float(self.R2_var.get()),
#                     'kd': float(self.kd2_var.get()),
#                     'ks': float(self.ks2_var.get()),
#                     'shininess': float(self.sh2_var.get()),
#                     'color': (float(self.C2r_var.get()), float(self.C2g_var.get()), float(self.C2b_var.get()))
#                 }
#             ]
#
#             lights = [
#                 {
#                     'pos': (float(self.L1x_var.get()), float(self.L1y_var.get()), float(self.L1z_var.get())),
#                     'I0': float(self.I1_var.get()),
#                     'color': (float(self.LC1r_var.get()), float(self.LC1g_var.get()), float(self.LC1b_var.get()))
#                 },
#                 {
#                     'pos': (float(self.L2x_var.get()), float(self.L2y_var.get()), float(self.L2z_var.get())),
#                     'I0': float(self.I2_var.get()),
#                     'color': (float(self.LC2r_var.get()), float(self.LC2g_var.get()), float(self.LC2b_var.get()))
#                 }
#             ]
#
#             img_uint8, img_float, I_max, I_min = render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights)
#             pil_img = Image.fromarray(img_uint8, mode="RGB")
#             self.last_pil = pil_img
#
#             # Масштабируем изображение для отображения (не изменяем исходное)
#             display_size = (min(600, Wres), min(600, Hres))
#             display_img = pil_img.resize(display_size, Image.Resampling.LANCZOS)
#             self.tk_img = ImageTk.PhotoImage(display_img)
#             self.image_label.config(image=self.tk_img)
#
#             # Обновляем информацию о яркости
#             self.info_var.set(f"Max = {I_max:.3g}, Min>0 = {I_min:.3g}")
#             pil_img.save("lab5_result.png")
#
#         except Exception as e:
#             messagebox.showerror("Ошибка", str(e))
#
#     def save(self):
#         if self.last_pil is None:
#             messagebox.showwarning("Нет изображения", "Сначала нажмите «Render».")
#             return
#         filename = filedialog.asksaveasfilename(
#             title="Сохранить изображение",
#             defaultextension=".png",
#             filetypes=[("PNG image", "*.png")]
#         )
#         if filename:
#             try:
#                 self.last_pil.save(filename)
#                 messagebox.showinfo("Сохранено", f"Изображение сохранено:\n{filename}")
#             except Exception as e:
#                 messagebox.showerror("Ошибка", str(e))
#
#
# # =====================================================================
# # ======================== ЗАПУСК ПРИЛОЖЕНИЯ ==========================
# # =====================================================================
#
# if __name__ == "__main__":
#     app = SceneApp()
#     app.mainloop()



import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# =====================================================================
# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =======================
# =====================================================================

def ray_sphere_intersect(O, D, C, R):
    """
    Находит ближайшее положительное пересечение луча O + t*D со сферой (C, R).
    Возвращает: t (массив), hit_mask (boolean)
    """
    OC = O - C
    a = np.sum(D * D, axis=-1)
    b = 2.0 * np.sum(D * OC, axis=-1)
    c = np.sum(OC * OC, axis=-1) - R ** 2
    disc = b * b - 4 * a * c
    hit = disc >= 0
    t = np.full_like(a, np.inf)
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denom = 2 * a + 1e-8
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    t_candidate = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
    t = np.where((t_candidate > 0) & hit, t_candidate, np.inf)
    return t, t != np.inf


def render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights):
    """
    Рендерит сцену с двумя сферами, тенями и цветом по модели Блинна–Фонга.
    """
    O = np.array([0.0, 0.0, zO], dtype=np.float64)

    xs = np.linspace(-W / 2, W / 2, Wres, endpoint=False) + W / (2 * Wres)
    ys = np.linspace(-H / 2, H / 2, Hres, endpoint=False) + H / (2 * Hres)
    X, Y = np.meshgrid(xs, ys)
    Px = X.ravel()
    Py = Y.ravel()
    Pz = np.full_like(Px, z_scr)
    P_screen = np.stack([Px, Py, Pz], axis=1)

    D = P_screen - O
    D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D = D / np.maximum(D_norm, 1e-8)

    depth = np.full(P_screen.shape[0], np.inf, dtype=np.float64)
    sphere_id = np.full(P_screen.shape[0], -1, dtype=int)

    for i, sph in enumerate(spheres):
        C = np.array(sph['center'], dtype=np.float64)
        R = sph['radius']
        t, hit = ray_sphere_intersect(O, D, C, R)
        closer = t < depth
        mask = hit & closer
        depth[mask] = t[mask]
        sphere_id[mask] = i

    P = O + D * depth[:, np.newaxis]
    V = O - P
    V_norm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.maximum(V_norm, 1e-8)

    final_color = np.zeros((P_screen.shape[0], 3), dtype=np.float64)
    valid = sphere_id >= 0

    if not np.any(valid):
        img_rgb = np.zeros((Hres, Wres, 3))
        return (np.zeros((Hres, Wres, 3), dtype=np.uint8), img_rgb, 0.0, 0.0)

    N = np.zeros_like(P)
    centers_arr = np.array([spheres[i]['center'] for i in sphere_id[valid]], dtype=np.float64)
    N[valid] = P[valid] - centers_arr
    N_norm = np.linalg.norm(N[valid], axis=1, keepdims=True)
    N[valid] /= np.maximum(N_norm, 1e-8)

    for idx in np.where(valid)[0]:
        sph = spheres[sphere_id[idx]]
        point = P[idx]
        normal = N[idx]
        color_acc = np.zeros(3)

        for light in lights:
            L_pos = np.array(light['pos'], dtype=np.float64)
            L_dir_vec = L_pos - point
            L_dist = np.linalg.norm(L_dir_vec)
            if L_dist < 1e-6:
                continue
            L_dir = L_dir_vec / L_dist
            light_color = np.array(light['color'], dtype=np.float64)
            I0 = light['I0']

            # Проверка тени
            shadow_ray_origin = point + normal * 1e-3
            shadow = False
            for other_sph in spheres:
                C_other = np.array(other_sph['center'], dtype=np.float64)
                R_other = other_sph['radius']
                t_shadow, hit_shadow = ray_sphere_intersect(shadow_ray_origin, L_dir, C_other, R_other)
                if np.any(hit_shadow) and np.min(t_shadow[hit_shadow]) < L_dist - 1e-3:
                    shadow = True
                    break

            if shadow:
                continue

            # Диффузная компонента
            diff = max(np.dot(normal, L_dir), 0.0)
            I_diff = sph['kd'] * I0 * diff

            # Зеркальная компонента
            H = L_dir + V[idx]
            H_norm_val = np.linalg.norm(H)
            if H_norm_val < 1e-8:
                spec = 0.0
            else:
                H = H / H_norm_val
                spec = max(np.dot(normal, H), 0.0) ** sph['shininess']
            I_spec = sph['ks'] * I0 * spec

            contrib = (I_diff + I_spec)
            color_acc += contrib * light_color * np.array(sph['color'])

        final_color[idx] = color_acc

    img_rgb = final_color.reshape((Hres, Wres, 3))

    # Максимальная и минимальная ненулевая яркость
    I_max = img_rgb.max()
    I_min = 0.0
    if I_max > 0:
        non_zero = img_rgb[img_rgb > 0]
        if len(non_zero) > 0:
            I_min = non_zero.min()

    if I_max > 0:
        img_norm = (img_rgb / I_max) * 255
    else:
        img_norm = img_rgb
    img_uint8 = np.clip(img_norm, 0, 255).astype(np.uint8)

    return img_uint8, img_rgb, I_max, I_min


# =====================================================================
# ============================= GUI ЧАСТЬ ==============================
# =====================================================================

class SceneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР5: Две сферы с тенями и цветом (Блинн–Фонг)")
        self.geometry("1250x900")

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель: изображение
        self.image_label = ttk.Label(main_frame, relief="sunken", background="black")
        self.image_label.grid(row=0, column=0, rowspan=3, padx=(0, 15), sticky="nw")

        # Правая панель: параметры
        params_frame = ttk.Frame(main_frame)
        params_frame.grid(row=0, column=1, sticky="nw")

        # === Экран и наблюдатель ===
        cam_frame = ttk.LabelFrame(params_frame, text="Экран и наблюдатель", padding=8)
        cam_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.W_var = self._add_param(cam_frame, "Ширина экрана W [мм]", 800, 0)
        self.H_var = self._add_param(cam_frame, "Высота экрана H [мм]", 600, 1)
        self.Wres_var = self._add_param(cam_frame, "Разрешение Wres [пикс]", 400, 2)
        self.Hres_var = self._add_param(cam_frame, "Разрешение Hres [пикс]", 300, 3)  # Новое поле!
        self.zscr_var = self._add_param(cam_frame, "z экрана z_scr [мм]", 0, 4)
        self.zO_var = self._add_param(cam_frame, "z наблюдателя zO [мм]", -1000, 5)

        # Связываем поля W/H и Wres/Hres для автоматического поддержания квадратных пикселей
        self.W_var.trace_add("write", self._on_W_change)
        self.H_var.trace_add("write", self._on_H_change)
        self.Wres_var.trace_add("write", self._on_Wres_change)
        self.Hres_var.trace_add("write", self._on_Hres_change)

        # === Сфера 1 ===
        sph1_frame = ttk.LabelFrame(params_frame, text="Сфера 1", padding=8)
        sph1_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.R1_var = self._add_param(sph1_frame, "Радиус R1 [мм]", 200, 0)
        self.C1x_var = self._add_param(sph1_frame, "Cx1 [мм]", -150, 1)
        self.C1y_var = self._add_param(sph1_frame, "Cy1 [мм]", 0, 2)
        self.C1z_var = self._add_param(sph1_frame, "Cz1 [мм]", 800, 3)
        self.kd1_var = self._add_param(sph1_frame, "kd1 (диффузия)", 0.7, 4)
        self.ks1_var = self._add_param(sph1_frame, "ks1 (зеркальность)", 0.3, 5)
        self.sh1_var = self._add_param(sph1_frame, "shininess1", 30, 6)
        self.C1r_var = self._add_param(sph1_frame, "Цвет R1", 1.0, 7)
        self.C1g_var = self._add_param(sph1_frame, "Цвет G1", 0.2, 8)
        self.C1b_var = self._add_param(sph1_frame, "Цвет B1", 0.2, 9)

        # === Сфера 2 ===
        sph2_frame = ttk.LabelFrame(params_frame, text="Сфера 2", padding=8)
        sph2_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        self.R2_var = self._add_param(sph2_frame, "Радиус R2 [мм]", 180, 0)
        self.C2x_var = self._add_param(sph2_frame, "Cx2 [мм]", 200, 1)
        self.C2y_var = self._add_param(sph2_frame, "Cy2 [мм]", 100, 2)
        self.C2z_var = self._add_param(sph2_frame, "Cz2 [мм]", 900, 3)
        self.kd2_var = self._add_param(sph2_frame, "kd2", 0.6, 4)
        self.ks2_var = self._add_param(sph2_frame, "ks2", 0.4, 5)
        self.sh2_var = self._add_param(sph2_frame, "shininess2", 50, 6)
        self.C2r_var = self._add_param(sph2_frame, "Цвет R2", 0.2, 7)
        self.C2g_var = self._add_param(sph2_frame, "Цвет G2", 0.8, 8)
        self.C2b_var = self._add_param(sph2_frame, "Цвет B2", 0.2, 9)

        # === Источники света ===
        light_frame = ttk.LabelFrame(params_frame, text="Источники света", padding=8)
        light_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nw")

        # Источник 1
        ttk.Label(light_frame, text="Источник 1", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        self.L1x_var = self._add_param(light_frame, "L1x [мм]", 2000, 1)
        self.L1y_var = self._add_param(light_frame, "L1y [мм]", 1500, 2)
        self.L1z_var = self._add_param(light_frame, "L1z [мм]", -500, 3)
        self.I1_var = self._add_param(light_frame, "I01 [Вт/ср]", 800, 4)
        self.LC1r_var = self._add_param(light_frame, "Цвет L1 (R)", 1.0, 5)
        self.LC1g_var = self._add_param(light_frame, "Цвет L1 (G)", 1.0, 6)
        self.LC1b_var = self._add_param(light_frame, "Цвет L1 (B)", 1.0, 7)

        ttk.Separator(light_frame, orient="horizontal").grid(row=8, column=0, columnspan=2, sticky="ew", pady=6)

        # Источник 2
        ttk.Label(light_frame, text="Источник 2", font=("TkDefaultFont", 10, "bold")).grid(row=9, column=0, columnspan=2, sticky="w")
        self.L2x_var = self._add_param(light_frame, "L2x [мм]", -1000, 10)
        self.L2y_var = self._add_param(light_frame, "L2y [мм]", -1000, 11)
        self.L2z_var = self._add_param(light_frame, "L2z [мм]", -800, 12)
        self.I2_var = self._add_param(light_frame, "I02 [Вт/ср]", 300, 13)
        self.LC2r_var = self._add_param(light_frame, "Цвет L2 (R)", 1.0, 14)
        self.LC2g_var = self._add_param(light_frame, "Цвет L2 (G)", 0.8, 15)
        self.LC2b_var = self._add_param(light_frame, "Цвет L2 (B)", 0.5, 16)

        # === Кнопки и информация ===
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky="sw", pady=(10, 0))

        self.info_var = tk.StringVar(value="Max = 0.0, Min>0 = 0.0")
        info_label = ttk.Label(control_frame, textvariable=self.info_var, foreground="blue", font=("TkDefaultFont", 10, "bold"))
        info_label.pack(anchor="w", pady=5)

        btn_render = ttk.Button(control_frame, text="Render", command=self.render, width=20)
        btn_render.pack(pady=5)

        btn_save = ttk.Button(control_frame, text="Save image", command=self.save, width=20)
        btn_save.pack(pady=5)

        self.last_pil = None
        self.render()  # первый рендер

    def _add_param(self, parent, label_text, default_value, row):
        """Вспомогательная функция для создания пары метка–поле ввода."""
        ttk.Label(parent, text=label_text, width=22, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 5))
        var = tk.StringVar(value=str(default_value))
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky="w")
        return var

    def _on_W_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Hres = int(self.Hres_var.get())
            Wres_new = int(round(Hres * W / H))
            self.Wres_var.set(str(Wres_new))
        except:
            pass

    def _on_H_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres_new = int(round(Wres * H / W))
            self.Hres_var.set(str(Hres_new))
        except:
            pass

    def _on_Wres_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres_new = int(round(Wres * H / W))
            self.Hres_var.set(str(Hres_new))
        except:
            pass

    def _on_Hres_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Hres = int(self.Hres_var.get())
            Wres_new = int(round(Hres * W / H))
            self.Wres_var.set(str(Wres_new))
        except:
            pass

    def render(self):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres = int(self.Hres_var.get())  # Теперь Hres тоже можно менять!
            z_scr = float(self.zscr_var.get())
            zO = float(self.zO_var.get())

            spheres = [
                {
                    'center': (float(self.C1x_var.get()), float(self.C1y_var.get()), float(self.C1z_var.get())),
                    'radius': float(self.R1_var.get()),
                    'kd': float(self.kd1_var.get()),
                    'ks': float(self.ks1_var.get()),
                    'shininess': float(self.sh1_var.get()),
                    'color': (float(self.C1r_var.get()), float(self.C1g_var.get()), float(self.C1b_var.get()))
                },
                {
                    'center': (float(self.C2x_var.get()), float(self.C2y_var.get()), float(self.C2z_var.get())),
                    'radius': float(self.R2_var.get()),
                    'kd': float(self.kd2_var.get()),
                    'ks': float(self.ks2_var.get()),
                    'shininess': float(self.sh2_var.get()),
                    'color': (float(self.C2r_var.get()), float(self.C2g_var.get()), float(self.C2b_var.get()))
                }
            ]

            lights = [
                {
                    'pos': (float(self.L1x_var.get()), float(self.L1y_var.get()), float(self.L1z_var.get())),
                    'I0': float(self.I1_var.get()),
                    'color': (float(self.LC1r_var.get()), float(self.LC1g_var.get()), float(self.LC1b_var.get()))
                },
                {
                    'pos': (float(self.L2x_var.get()), float(self.L2y_var.get()), float(self.L2z_var.get())),
                    'I0': float(self.I2_var.get()),
                    'color': (float(self.LC2r_var.get()), float(self.LC2g_var.get()), float(self.LC2b_var.get()))
                }
            ]

            img_uint8, img_float, I_max, I_min = render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights)
            pil_img = Image.fromarray(img_uint8, mode="RGB")
            self.last_pil = pil_img

            # Масштабируем изображение для отображения (не изменяем исходное)
            display_size = (min(600, Wres), min(600, Hres))
            display_img = pil_img.resize(display_size, Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=self.tk_img)

            # Обновляем информацию о яркости
            self.info_var.set(f"Max = {I_max:.3g}, Min>0 = {I_min:.3g}")
            pil_img.save("lab5_result.png")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save(self):
        if self.last_pil is None:
            messagebox.showwarning("Нет изображения", "Сначала нажмите «Render».")
            return
        filename = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if filename:
            try:
                self.last_pil.save(filename)
                messagebox.showinfo("Сохранено", f"Изображение сохранено:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))


# =====================================================================
# ======================== ЗАПУСК ПРИЛОЖЕНИЯ ==========================
# =====================================================================

if __name__ == "__main__":
    app = SceneApp()
    app.mainloop()