"""Dashboard page for the first milestone."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DashboardPage(QWidget):
    """Minimal first dashboard used to verify the desktop stack."""

    def __init__(self) -> None:
        super().__init__()

        title = QLabel("ML Experiment Lab")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700;")

        subtitle = QLabel("Project foundation is working. Dashboard module comes next.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
