"""Quick actions widget for the dashboard."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QuickActionsWidget(QWidget):
    """Widget containing common actions available from the dashboard."""

    new_experiment_requested = Signal()
    upload_dataset_requested = Signal()
    open_model_requested = Signal()

    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Quick Actions")

        new_experiment_button = QPushButton("New Experiment")
        upload_dataset_button = QPushButton("Upload Dataset")
        open_model_button = QPushButton("Open Model")

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