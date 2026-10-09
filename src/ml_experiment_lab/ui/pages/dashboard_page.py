"""Dashboard page."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ml_experiment_lab.ui.widgets.dataset_overview_widget import (
    DatasetOverviewWidget,
)
from ml_experiment_lab.ui.widgets.quick_actions_widget import (
    QuickActionsWidget,
)


class DashboardPage(QWidget):
    """Main dashboard page."""

    def __init__(self) -> None:
        super().__init__()

        # Dashboard title.
        title = QLabel("ML Experiment Lab")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Project foundation is working...")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Status message for Quick Actions.
        self.action_status = QLabel("No action selected.")
        self.action_status.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        # Quick Actions widget.
        self.quick_actions = QuickActionsWidget()

        self.quick_actions.new_experiment_requested.connect(
            self.handle_new_experiment
        )
        self.quick_actions.upload_dataset_requested.connect(
            self.handle_upload_dataset
        )
        self.quick_actions.open_model_requested.connect(
            self.handle_open_model
        )

        # Dataset overview widget.
        self.dataset_overview = DatasetOverviewWidget()

        # Main dashboard layout.
        layout = QVBoxLayout(self)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.quick_actions)
        layout.addWidget(self.dataset_overview)
        layout.addWidget(self.action_status)

    def handle_new_experiment(self) -> None:
        """Handle a request to create a new experiment."""
        self.action_status.setText(
            "New Experiment was requested."
        )

    def handle_upload_dataset(self) -> None:
        """Handle a request to upload a dataset."""
        self.action_status.setText(
            "Upload Dataset was requested."
        )

    def handle_open_model(self) -> None:
        """Handle a request to open a model."""
        self.action_status.setText(
            "Open Model was requested."
        )