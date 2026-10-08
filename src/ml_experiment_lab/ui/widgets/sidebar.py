"""Sidebar navigation widget."""

from collections.abc import Callable

from PySide6.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class Sidebar(QWidget):
    """Application sidebar used for page navigation."""

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

        dashboard_button = QPushButton("Dashboard")
        datasets_button = QPushButton("Datasets")
        experiments_button = QPushButton("Experiments")
        models_button = QPushButton("Models")
        benchmarks_button = QPushButton("Benchmarks")
        settings_button = QPushButton("Settings")

        dashboard_button.clicked.connect(on_dashboard)
        datasets_button.clicked.connect(on_datasets)
        experiments_button.clicked.connect(on_experiments)
        models_button.clicked.connect(on_models)
        benchmarks_button.clicked.connect(on_benchmarks)
        settings_button.clicked.connect(on_settings)

        layout.addWidget(dashboard_button)
        layout.addWidget(datasets_button)
        layout.addWidget(experiments_button)
        layout.addWidget(models_button)
        layout.addWidget(benchmarks_button)

        layout.addStretch()

        layout.addWidget(settings_button)