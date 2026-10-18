
"""Компонент за бързи действия в началното табло."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QuickActionsWidget(QWidget):
    """Компонент с най-често използваните действия."""

    new_experiment_requested = Signal()
    upload_dataset_requested = Signal()
    open_model_requested = Signal()

    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Бързи действия")

        new_experiment_button = QPushButton("Нов експеримент")
        upload_dataset_button = QPushButton(
            "Импортиране на набор от данни"
        )
        open_model_button = QPushButton("Отваряне на модел")

        new_experiment_button.clicked.connect(
            self.new_experiment_requested.emit
        )
        upload_dataset_button.clicked.connect(
            self.upload_dataset_requested.emit
        )
        open_model_button.clicked.connect(
            self.open_model_requested.emit
        )

        layout = QVBoxLayout(self)

        layout.addWidget(title)
        layout.addWidget(new_experiment_button)
        layout.addWidget(upload_dataset_button)
        layout.addWidget(open_model_button)
