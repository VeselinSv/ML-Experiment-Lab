"""Main application window."""

from PySide6.QtWidgets import QMainWindow

from ml_experiment_lab.config import (
    APP_NAME,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from ml_experiment_lab.ui.pages.dashboard_page import DashboardPage


class MainWindow(QMainWindow):
    """Top-level window of ML Experiment Lab."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self.setCentralWidget(DashboardPage())
