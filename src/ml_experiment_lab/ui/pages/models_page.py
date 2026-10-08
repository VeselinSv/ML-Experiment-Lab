from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ModelsPage(QWidget):
    """Models page."""

    def __init__(self) -> None:
        super().__init__()

        label = QLabel("Models")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(label)