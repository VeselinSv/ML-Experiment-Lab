
"""Главен прозорец на приложението."""

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from ml_experiment_lab.config import (
    APP_NAME,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from ml_experiment_lab.ui.pages.benchmarks_page import BenchmarksPage
from ml_experiment_lab.ui.pages.dashboard_page import DashboardPage
from ml_experiment_lab.ui.pages.datasets_page import DatasetsPage
from ml_experiment_lab.ui.pages.experiments_page import ExperimentsPage
from ml_experiment_lab.ui.pages.models_page import ModelsPage
from ml_experiment_lab.ui.pages.settings_page import SettingsPage
from ml_experiment_lab.ui.widgets.sidebar import Sidebar


class MainWindow(QMainWindow):
    """Главен прозорец на ML Experiment Lab."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(
            f"{APP_NAME} — Платформа за машинно обучение"
        )
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Създаваме страниците на приложението.
        self.dashboard_page = DashboardPage()
        self.datasets_page = DatasetsPage()
        self.experiments_page = ExperimentsPage()
        self.models_page = ModelsPage()
        self.benchmarks_page = BenchmarksPage()
        self.settings_page = SettingsPage()

        # Контейнер за превключване между страниците.
        self.page_stack = QStackedWidget()

        self.page_stack.addWidget(self.dashboard_page)
        self.page_stack.addWidget(self.datasets_page)
        self.page_stack.addWidget(self.experiments_page)
        self.page_stack.addWidget(self.models_page)
        self.page_stack.addWidget(self.benchmarks_page)
        self.page_stack.addWidget(self.settings_page)

        # Лява навигационна лента.
        sidebar = Sidebar(
            on_dashboard=self.show_dashboard,
            on_datasets=self.show_datasets,
            on_experiments=self.show_experiments,
            on_models=self.show_models,
            on_benchmarks=self.show_benchmarks,
            on_settings=self.show_settings,
        )

        # Основно разположение на прозореца.
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.page_stack)

        self.setCentralWidget(central_widget)

    def show_dashboard(self) -> None:
        """Показва началното табло."""
        self.page_stack.setCurrentWidget(self.dashboard_page)

    def show_datasets(self) -> None:
        """Показва страницата с наборите от данни."""
        self.page_stack.setCurrentWidget(self.datasets_page)

    def show_experiments(self) -> None:
        """Показва страницата с експериментите."""
        self.page_stack.setCurrentWidget(self.experiments_page)

    def show_models(self) -> None:
        """Показва страницата с моделите."""
        self.page_stack.setCurrentWidget(self.models_page)

    def show_benchmarks(self) -> None:
        """Показва страницата за сравняване на резултатите."""
        self.page_stack.setCurrentWidget(self.benchmarks_page)

    def show_settings(self) -> None:
        """Показва страницата с настройките."""
        self.page_stack.setCurrentWidget(self.settings_page)
