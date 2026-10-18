
"""Навигационна лента на приложението."""

from collections.abc import Callable

from PySide6.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class Sidebar(QWidget):
    """Странична лента за навигация между страниците."""

    def __init__(
        self,
        on_dashboard: Callable[[], None],
        on_datasets: Callable[[], None],
        on_experiments: Callable[[], None],
        on_models: Callable[[], None],
        on_benchmarks: Callable[[], None],
        on_settings: Callable[[], None],
    ) -> None:
        super().__init__()

        self.setMinimumWidth(220)

        layout = QVBoxLayout(self)

        # Бутони за навигация.
        self.dashboard_button = QPushButton("Табло")
        self.datasets_button = QPushButton("Набори от данни")
        self.experiments_button = QPushButton("Експерименти")
        self.models_button = QPushButton("Модели")
        self.benchmarks_button = QPushButton("Сравнения")
        self.settings_button = QPushButton("Настройки")

        # Свързваме бутоните със съответните страници.
        self.dashboard_button.clicked.connect(on_dashboard)
        self.datasets_button.clicked.connect(on_datasets)
        self.experiments_button.clicked.connect(on_experiments)
        self.models_button.clicked.connect(on_models)
        self.benchmarks_button.clicked.connect(on_benchmarks)
        self.settings_button.clicked.connect(on_settings)

        layout.addWidget(self.dashboard_button)
        layout.addWidget(self.datasets_button)
        layout.addWidget(self.experiments_button)
        layout.addWidget(self.models_button)
        layout.addWidget(self.benchmarks_button)

        # Запазваме настройките отделно в долната част.
        layout.addStretch()

        layout.addWidget(self.settings_button)
