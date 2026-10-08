from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DatasetsPage(QWidget):
    """Datasets page."""

    def __init__(self) -> None:
        super().__init__()

        label = QLabel("Datasets")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(label)