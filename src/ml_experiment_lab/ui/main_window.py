"""Main application window."""

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
    """Top-level window of ML Experiment Lab."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self.dashboard_page = DashboardPage()
        self.datasets_page = DatasetsPage()
        self.experiments_page = ExperimentsPage()
        self.models_page = ModelsPage()
        self.benchmarks_page = BenchmarksPage()
        self.settings_page = SettingsPage()

        self.page_stack = QStackedWidget()

        self.page_stack.addWidget(self.dashboard_page)
        self.page_stack.addWidget(self.datasets_page)
        self.page_stack.addWidget(self.experiments_page)
        self.page_stack.addWidget(self.models_page)
        self.page_stack.addWidget(self.benchmarks_page)
        self.page_stack.addWidget(self.settings_page)

        sidebar = Sidebar(
            on_dashboard=self.show_dashboard,
            on_datasets=self.show_datasets,
            on_experiments=self.show_experiments,
            on_models=self.show_models,
            on_benchmarks=self.show_benchmarks,
            on_settings=self.show_settings,
        )

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.page_stack)

        self.setCentralWidget(central_widget)

    def show_dashboard(self) -> None:
        """Show the dashboard page."""
        self.page_stack.setCurrentWidget(self.dashboard_page)

    def show_datasets(self) -> None:
        """Show the datasets page."""
        self.page_stack.setCurrentWidget(self.datasets_page)

    def show_experiments(self) -> None:
        """Show the experiments page."""
        self.page_stack.setCurrentWidget(self.experiments_page)

    def show_models(self) -> None:
        """Show the models page."""
        self.page_stack.setCurrentWidget(self.models_page)

    def show_benchmarks(self) -> None:
        """Show the benchmarks page."""
        self.page_stack.setCurrentWidget(self.benchmarks_page)

    def show_settings(self) -> None:
        """Show the settings page."""
        self.page_stack.setCurrentWidget(self.settings_page)