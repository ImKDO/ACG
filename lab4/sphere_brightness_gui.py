"""
Лабораторная работа 4: GUI приложение для расчета яркости на сфере
Модель освещения: Блинн-Фонг с использованием PyQt5

Возможности:
- Интерактивное управление всеми параметрами сцены
- Режим реального времени с автообновлением
- Множественные источники света
- Сохранение/загрузка конфигураций
- Экспорт изображений в различных форматах
"""

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QSlider, QPushButton,
                                 QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                                 QComboBox, QFileDialog, QMessageBox, QTabWidget,
                                 QScrollArea, QGridLayout, QSplitter)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 не установлен. Используйте: pip install PyQt5")

import numpy as np
from PIL import Image
import json
import sys
from pathlib import Path
from sphere_brightness import PointLight, Sphere, Scene


class SphereVisualizerGUI(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.scene = Scene('config.json')
        self.brightness_map = None
        self.image_array = None
        self.auto_update = False
        self.selected_light_idx = 0

        # Для быстрого рендеринга
        self.preview_resolution = 200
        self.full_resolution = 800
        self.current_resolution = self.preview_resolution

        self.init_ui()
        self.render_scene()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle(
            'Интерактивная визуализация яркости на сфере - ЛР4')
        self.setGeometry(100, 100, 1400, 900)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Разделитель
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Левая панель - управление
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)

        # Правая панель - визуализация
        view_panel = self.create_view_panel()
        splitter.addWidget(view_panel)

        splitter.setSizes([400, 1000])

        # Статусная строка
        self.statusBar().showMessage('Готов к работе')

    def create_control_panel(self):
        """Создание панели управления"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)

        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)

        # Вкладки для разных групп параметров
        tabs = QTabWidget()

        # Вкладка "Сфера"
        sphere_tab = self.create_sphere_controls()
        tabs.addTab(sphere_tab, "Сфера")

        # Вкладка "Источники света"
        lights_tab = self.create_lights_controls()
        tabs.addTab(lights_tab, "Источники света")

        # Вкладка "Рендеринг"
        render_tab = self.create_render_controls()
        tabs.addTab(render_tab, "Рендеринг")

        layout.addWidget(tabs)

        # Кнопки действий
        buttons_layout = QVBoxLayout()

        self.btn_render = QPushButton('🔄 Обновить изображение')
        self.btn_render.clicked.connect(self.render_scene)
        buttons_layout.addWidget(self.btn_render)

        self.btn_save_image = QPushButton('💾 Сохранить изображение')
        self.btn_save_image.clicked.connect(self.save_image)
        buttons_layout.addWidget(self.btn_save_image)

        self.btn_save_config = QPushButton('📄 Сохранить конфигурацию')
        self.btn_save_config.clicked.connect(self.save_config)
        buttons_layout.addWidget(self.btn_save_config)

        self.btn_load_config = QPushButton('📂 Загрузить конфигурацию')
        self.btn_load_config.clicked.connect(self.load_config)
        buttons_layout.addWidget(self.btn_load_config)

        self.btn_reset = QPushButton('🔃 Сброс к умолчаниям')
        self.btn_reset.clicked.connect(self.reset_scene)
        buttons_layout.addWidget(self.btn_reset)

        layout.addLayout(buttons_layout)

        # Информация
        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 10px; }")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        layout.addStretch()
        scroll.setWidget(control_widget)
        return scroll

    def create_sphere_controls(self):
        """Создание элементов управления сферой"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Позиция сферы
        pos_group = QGroupBox("Позиция центра сферы (мм)")
        pos_layout = QGridLayout()

        pos_layout.addWidget(QLabel("X:"), 0, 0)
        self.sphere_x = QSpinBox()
        self.sphere_x.setRange(-5000, 5000)
        self.sphere_x.setValue(int(self.scene.sphere.center[0]))
        self.sphere_x.setSingleStep(10)
        self.sphere_x.valueChanged.connect(self.on_params_changed)
        pos_layout.addWidget(self.sphere_x, 0, 1)

        pos_layout.addWidget(QLabel("Y:"), 1, 0)
        self.sphere_y = QSpinBox()
        self.sphere_y.setRange(-5000, 5000)
        self.sphere_y.setValue(int(self.scene.sphere.center[1]))
        self.sphere_y.setSingleStep(10)
        self.sphere_y.valueChanged.connect(self.on_params_changed)
        pos_layout.addWidget(self.sphere_y, 1, 1)

        pos_layout.addWidget(QLabel("Z:"), 2, 0)
        self.sphere_z = QSpinBox()
        self.sphere_z.setRange(-5000, 2000)
        self.sphere_z.setValue(int(self.scene.sphere.center[2]))
        self.sphere_z.setSingleStep(10)
        self.sphere_z.valueChanged.connect(self.on_params_changed)
        pos_layout.addWidget(self.sphere_z, 2, 1)

        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        # Радиус
        radius_group = QGroupBox("Радиус сферы (мм)")
        radius_layout = QVBoxLayout()

        self.sphere_radius = QSpinBox()
        self.sphere_radius.setRange(50, 1000)
        self.sphere_radius.setValue(int(self.scene.sphere.radius))
        self.sphere_radius.setSingleStep(5)
        self.sphere_radius.valueChanged.connect(self.on_params_changed)
        radius_layout.addWidget(self.sphere_radius)

        radius_group.setLayout(radius_layout)
        layout.addWidget(radius_group)

        # Материал (Блинн-Фонг)
        material_group = QGroupBox("Свойства материала (Блинн-Фонг)")
        material_layout = QGridLayout()

        material_layout.addWidget(QLabel("Kd (диффузное):"), 0, 0)
        self.sphere_kd = QDoubleSpinBox()
        self.sphere_kd.setRange(0.0, 1.0)
        self.sphere_kd.setValue(self.scene.sphere.kd)
        self.sphere_kd.setSingleStep(0.05)
        self.sphere_kd.setDecimals(2)
        self.sphere_kd.valueChanged.connect(self.on_params_changed)
        material_layout.addWidget(self.sphere_kd, 0, 1)

        material_layout.addWidget(QLabel("Ks (зеркальное):"), 1, 0)
        self.sphere_ks = QDoubleSpinBox()
        self.sphere_ks.setRange(0.0, 1.0)
        self.sphere_ks.setValue(self.scene.sphere.ks)
        self.sphere_ks.setSingleStep(0.05)
        self.sphere_ks.setDecimals(2)
        self.sphere_ks.valueChanged.connect(self.on_params_changed)
        material_layout.addWidget(self.sphere_ks, 1, 1)

        material_layout.addWidget(QLabel("Блеск (shininess):"), 2, 0)
        self.sphere_shininess = QSpinBox()
        self.sphere_shininess.setRange(1, 256)
        self.sphere_shininess.setValue(int(self.scene.sphere.shininess))
        self.sphere_shininess.setSingleStep(1)
        self.sphere_shininess.valueChanged.connect(self.on_params_changed)
        material_layout.addWidget(self.sphere_shininess, 2, 1)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        layout.addStretch()
        return widget

    def create_lights_controls(self):
        """Создание элементов управления источниками света"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Выбор источника
        select_group = QGroupBox("Выбор источника света")
        select_layout = QHBoxLayout()

        self.light_combo = QComboBox()
        self.update_light_combo()
        self.light_combo.currentIndexChanged.connect(self.on_light_selected)
        select_layout.addWidget(self.light_combo)

        self.btn_add_light = QPushButton("➕")
        self.btn_add_light.setMaximumWidth(40)
        self.btn_add_light.clicked.connect(self.add_light)
        select_layout.addWidget(self.btn_add_light)

        self.btn_remove_light = QPushButton("➖")
        self.btn_remove_light.setMaximumWidth(40)
        self.btn_remove_light.clicked.connect(self.remove_light)
        select_layout.addWidget(self.btn_remove_light)

        select_group.setLayout(select_layout)
        layout.addWidget(select_group)

        # Позиция источника
        light_pos_group = QGroupBox("Позиция источника (мм)")
        light_pos_layout = QGridLayout()

        light_pos_layout.addWidget(QLabel("X:"), 0, 0)
        self.light_x = QSpinBox()
        self.light_x.setRange(-10000, 10000)
        self.light_x.setSingleStep(10)
        self.light_x.valueChanged.connect(self.on_light_params_changed)
        light_pos_layout.addWidget(self.light_x, 0, 1)

        light_pos_layout.addWidget(QLabel("Y:"), 1, 0)
        self.light_y = QSpinBox()
        self.light_y.setRange(-10000, 10000)
        self.light_y.setSingleStep(10)
        self.light_y.valueChanged.connect(self.on_light_params_changed)
        light_pos_layout.addWidget(self.light_y, 1, 1)

        light_pos_layout.addWidget(QLabel("Z:"), 2, 0)
        self.light_z = QSpinBox()
        self.light_z.setRange(100, 10000)
        self.light_z.setSingleStep(10)
        self.light_z.valueChanged.connect(self.on_light_params_changed)
        light_pos_layout.addWidget(self.light_z, 2, 1)

        light_pos_group.setLayout(light_pos_layout)
        layout.addWidget(light_pos_group)

        # Интенсивность
        intensity_group = QGroupBox("Интенсивность (Вт/ср)")
        intensity_layout = QVBoxLayout()

        self.light_intensity = QSpinBox()
        self.light_intensity.setRange(10, 5000000)
        self.light_intensity.setSingleStep(10000)
        self.light_intensity.valueChanged.connect(self.on_light_params_changed)
        intensity_layout.addWidget(self.light_intensity)

        intensity_group.setLayout(intensity_layout)
        layout.addWidget(intensity_group)

        self.update_light_controls()

        layout.addStretch()
        return widget

    def create_render_controls(self):
        """Создание элементов управления рендерингом"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Качество
        quality_group = QGroupBox("Качество рендеринга")
        quality_layout = QVBoxLayout()

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(['Быстрое (200x200)', 'Среднее (400x400)',
                                     'Высокое (800x800)', 'Очень высокое (1600x1600)'])
        self.quality_combo.setCurrentIndex(0)
        self.quality_combo.currentIndexChanged.connect(self.on_quality_changed)
        quality_layout.addWidget(self.quality_combo)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # Автообновление
        auto_group = QGroupBox("Режим обновления")
        auto_layout = QVBoxLayout()

        self.auto_update_check = QCheckBox("Автоматическое обновление")
        self.auto_update_check.stateChanged.connect(
            self.on_auto_update_changed)
        auto_layout.addWidget(self.auto_update_check)

        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)
        
        # Нормализация яркости
        normalize_group = QGroupBox("Нормализация яркости")
        normalize_layout = QVBoxLayout()
        
        self.normalize_mode = QComboBox()
        self.normalize_mode.addItems([
            'Автоматическая (0-255)',
            'Фиксированная (показывает реальную разницу)'
        ])
        self.normalize_mode.setCurrentIndex(1)  # По умолчанию фиксированная
        self.normalize_mode.currentIndexChanged.connect(self.on_normalize_changed)
        normalize_layout.addWidget(self.normalize_mode)
        
        # Параметр для фиксированной нормализации
        self.fixed_normalize_layout = QHBoxLayout()
        self.fixed_normalize_layout.addWidget(QLabel("Масштаб:"))
        self.normalize_scale = QDoubleSpinBox()
        self.normalize_scale.setRange(0.00001, 10.0)
        self.normalize_scale.setValue(0.0001)
        self.normalize_scale.setDecimals(6)
        self.normalize_scale.setSingleStep(0.00001)
        self.normalize_scale.valueChanged.connect(self.on_normalize_scale_changed)
        self.fixed_normalize_layout.addWidget(self.normalize_scale)
        normalize_layout.addLayout(self.fixed_normalize_layout)
        
        normalize_group.setLayout(normalize_layout)
        layout.addWidget(normalize_group)

        # Экран
        screen_group = QGroupBox("Размеры экрана (мм)")
        screen_layout = QGridLayout()

        screen_layout.addWidget(QLabel("Ширина:"), 0, 0)
        self.screen_width = QSpinBox()
        self.screen_width.setRange(100, 10000)
        self.screen_width.setValue(int(self.scene.screen_width))
        self.screen_width.setSingleStep(10)
        self.screen_width.valueChanged.connect(self.on_screen_params_changed)
        screen_layout.addWidget(self.screen_width, 0, 1)

        screen_layout.addWidget(QLabel("Высота:"), 1, 0)
        self.screen_height = QSpinBox()
        self.screen_height.setRange(100, 10000)
        self.screen_height.setValue(int(self.scene.screen_height))
        self.screen_height.setSingleStep(10)
        self.screen_height.valueChanged.connect(self.on_screen_params_changed)
        screen_layout.addWidget(self.screen_height, 1, 1)

        screen_group.setLayout(screen_layout)
        layout.addWidget(screen_group)

        layout.addStretch()
        return widget

    def create_view_panel(self):
        """Создание панели визуализации"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Заголовок
        title = QLabel("Распределение яркости на сфере")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Изображение
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 600)
        self.image_label.setStyleSheet("QLabel { background-color: black; }")
        layout.addWidget(self.image_label, 1)

        return widget

    def update_light_combo(self):
        """Обновление комбобокса источников света"""
        self.light_combo.clear()
        for i in range(len(self.scene.lights)):
            self.light_combo.addItem(f"Источник света {i+1}")

    def update_light_controls(self):
        """Обновление элементов управления текущего источника"""
        if len(self.scene.lights) > self.selected_light_idx:
            light = self.scene.lights[self.selected_light_idx]
            # Блокируем сигналы при обновлении, чтобы не вызывать on_light_params_changed
            self.light_x.blockSignals(True)
            self.light_y.blockSignals(True)
            self.light_z.blockSignals(True)
            self.light_intensity.blockSignals(True)

            self.light_x.setValue(int(light.position[0]))
            self.light_y.setValue(int(light.position[1]))
            self.light_z.setValue(int(light.position[2]))
            self.light_intensity.setValue(int(light.intensity))

            # Разблокируем сигналы
            self.light_x.blockSignals(False)
            self.light_y.blockSignals(False)
            self.light_z.blockSignals(False)
            self.light_intensity.blockSignals(False)

    def on_light_selected(self, index):
        """Обработка выбора источника света"""
        self.selected_light_idx = index
        self.update_light_controls()

    def add_light(self):
        """Добавление нового источника света"""
        # Создаем новый источник с уникальной интенсивностью
        new_intensity = 500000 + len(self.scene.lights) * 100000
        new_light = PointLight([0, 0, 500], new_intensity)
        self.scene.lights.append(new_light)
        self.update_light_combo()
        self.light_combo.setCurrentIndex(len(self.scene.lights) - 1)
        self.statusBar().showMessage(
            f'Добавлен источник света {len(self.scene.lights)} с интенсивностью {new_intensity} Вт/ср',
            3000
        )
        self.on_params_changed()

    def remove_light(self):
        """Удаление текущего источника света"""
        if len(self.scene.lights) > 1:
            del self.scene.lights[self.selected_light_idx]
            self.update_light_combo()
            self.selected_light_idx = min(
                self.selected_light_idx, len(self.scene.lights) - 1)
            self.light_combo.setCurrentIndex(self.selected_light_idx)
            self.on_params_changed()
        else:
            QMessageBox.warning(
                self, "Внимание", "Должен быть хотя бы один источник света!")

    def on_params_changed(self):
        """Обработка изменения параметров сферы"""
        self.scene.sphere.center[0] = self.sphere_x.value()
        self.scene.sphere.center[1] = self.sphere_y.value()
        self.scene.sphere.center[2] = self.sphere_z.value()
        self.scene.sphere.radius = self.sphere_radius.value()
        self.scene.sphere.kd = self.sphere_kd.value()
        self.scene.sphere.ks = self.sphere_ks.value()
        self.scene.sphere.shininess = self.sphere_shininess.value()

        if self.auto_update:
            self.render_scene()

    def on_light_params_changed(self):
        """Обработка изменения параметров источника света"""
        if len(self.scene.lights) > self.selected_light_idx:
            light = self.scene.lights[self.selected_light_idx]
            light.position[0] = self.light_x.value()
            light.position[1] = self.light_y.value()
            light.position[2] = self.light_z.value()
            light.intensity = self.light_intensity.value()

            if self.auto_update:
                self.render_scene()

    def on_screen_params_changed(self):
        """Обработка изменения параметров экрана"""
        self.scene.screen_width = self.screen_width.value()
        self.scene.screen_height = self.screen_height.value()
        self.scene.pixel_width = self.scene.screen_width / self.scene.resolution_width
        self.scene.pixel_height = self.scene.screen_height / self.scene.resolution_height

        if self.auto_update:
            self.render_scene()

    def on_quality_changed(self, index):
        """Изменение качества рендеринга"""
        resolutions = [200, 400, 800, 1600]
        self.current_resolution = resolutions[index]

    def on_auto_update_changed(self, state):
        """Включение/выключение автообновления"""
        self.auto_update = (state == Qt.Checked)
        if self.auto_update:
            self.render_scene()
    
    def on_normalize_changed(self, index):
        """Изменение режима нормализации"""
        if self.brightness_map is not None:
            self.apply_normalization()
            self.display_image()
    
    def on_normalize_scale_changed(self, value):
        """Изменение масштаба для фиксированной нормализации"""
        if self.brightness_map is not None and self.normalize_mode.currentIndex() == 1:
            self.apply_normalization()
            self.display_image()

    def render_scene(self):
        """Рендеринг сцены"""
        self.statusBar().showMessage('Рендеринг...')
        QApplication.processEvents()

        # Временно изменяем разрешение
        old_width = self.scene.resolution_width
        old_height = self.scene.resolution_height

        self.scene.resolution_width = self.current_resolution
        self.scene.resolution_height = self.current_resolution
        self.scene.pixel_width = self.scene.screen_width / self.scene.resolution_width
        self.scene.pixel_height = self.scene.screen_height / self.scene.resolution_height

        # Рендеринг
        self.brightness_map = self.scene.render()

        # Нормализация
        max_brightness = np.max(self.brightness_map)
        min_brightness = np.min(self.brightness_map[self.brightness_map > 0]) if np.any(
            self.brightness_map > 0) else 0

        if max_brightness > 0:
            self.image_array = (self.brightness_map /
                                max_brightness * 255).astype(np.uint8)
        else:
            self.image_array = np.zeros_like(
                self.brightness_map, dtype=np.uint8)

        # Отображение
        self.display_image()

        # Обновление информации
        info_text = f"Разрешение: {self.current_resolution}x{self.current_resolution}\n"
        info_text += f"Мин. яркость: {min_brightness:.2e}\n"
        info_text += f"Макс. яркость: {max_brightness:.2e}\n"
        info_text += f"Источников света: {len(self.scene.lights)}"
        self.info_label.setText(info_text)

        self.statusBar().showMessage('Готово', 3000)

    def display_image(self):
        """Отображение изображения"""
        if self.image_array is not None:
            height, width = self.image_array.shape
            bytes_per_line = width
            q_image = QImage(self.image_array.data, width, height,
                             bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

    def save_image(self):
        """Сохранение изображения"""
        if self.image_array is None:
            QMessageBox.warning(
                self, "Внимание", "Нет изображения для сохранения!")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение",
                                                  "output/sphere_interactive.png",
                                                  "PNG Files (*.png);;All Files (*)")
        if filename:
            img = Image.fromarray(self.image_array, mode='L')
            img.save(filename)
            self.statusBar().showMessage(
                f'Изображение сохранено: {filename}', 5000)

    def save_config(self):
        """Сохранение конфигурации"""
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить конфигурацию",
                                                  "output/config_gui.json",
                                                  "JSON Files (*.json);;All Files (*)")
        if filename:
            config = {
                "screen": {
                    "width": self.scene.screen_width,
                    "height": self.scene.screen_height,
                    "resolution_width": self.current_resolution,
                    "resolution_height": self.current_resolution
                },
                "observer": self.scene.observer_pos.tolist(),
                "sphere": {
                    "center": self.scene.sphere.center.tolist(),
                    "radius": float(self.scene.sphere.radius),
                    "material": {
                        "kd": float(self.scene.sphere.kd),
                        "ks": float(self.scene.sphere.ks),
                        "shininess": float(self.scene.sphere.shininess)
                    }
                },
                "lights": [
                    {
                        "position": light.position.tolist(),
                        "intensity": float(light.intensity)
                    }
                    for light in self.scene.lights
                ]
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(
                f'Конфигурация сохранена: {filename}', 5000)

    def load_config(self):
        """Загрузка конфигурации"""
        filename, _ = QFileDialog.getOpenFileName(self, "Загрузить конфигурацию",
                                                  "",
                                                  "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                self.scene = Scene(filename)
                self.selected_light_idx = 0

                # Обновление всех контролов
                self.sphere_x.setValue(int(self.scene.sphere.center[0]))
                self.sphere_y.setValue(int(self.scene.sphere.center[1]))
                self.sphere_z.setValue(int(self.scene.sphere.center[2]))
                self.sphere_radius.setValue(int(self.scene.sphere.radius))
                self.sphere_kd.setValue(self.scene.sphere.kd)
                self.sphere_ks.setValue(self.scene.sphere.ks)
                self.sphere_shininess.setValue(
                    int(self.scene.sphere.shininess))

                self.update_light_combo()
                self.update_light_controls()

                self.render_scene()
                self.statusBar().showMessage(
                    f'Конфигурация загружена: {filename}', 5000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Не удалось загрузить конфигурацию: {e}")

    def reset_scene(self):
        """Сброс к умолчаниям"""
        reply = QMessageBox.question(self, 'Подтверждение',
                                     'Сбросить все параметры к значениям по умолчанию?',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.scene = Scene('config.json')
            self.selected_light_idx = 0

            # Обновление контролов
            self.sphere_x.setValue(int(self.scene.sphere.center[0]))
            self.sphere_y.setValue(int(self.scene.sphere.center[1]))
            self.sphere_z.setValue(int(self.scene.sphere.center[2]))
            self.sphere_radius.setValue(int(self.scene.sphere.radius))
            self.sphere_kd.setValue(self.scene.sphere.kd)
            self.sphere_ks.setValue(self.scene.sphere.ks)
            self.sphere_shininess.setValue(int(self.scene.sphere.shininess))

            self.update_light_combo()
            self.update_light_controls()

            self.render_scene()


def main():
    """Главная функция"""
    if not PYQT_AVAILABLE:
        print("\nПожалуйста, установите PyQt5:")
        print("  pip install PyQt5")
        print("\nИли используйте версию с matplotlib:")
        print("  python sphere_brightness_interactive.py")
        return

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = SphereVisualizerGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
