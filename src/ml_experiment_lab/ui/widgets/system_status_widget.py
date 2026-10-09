"""System status widget for the dashboard."""

from PySide6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)


class SystemStatusWidget(QWidget):
    """Display the current status of application components."""

    def __init__(self) -> None:
        super().__init__()

        self.backend_status_value = QLabel(
            "Not configured"
        )

        self.ml_engine_status_value = QLabel(
            "Not initialized"
        )

        self.training_status_value = QLabel(
            "Not tracked yet"
        )

        backend_row = QHBoxLayout()
        backend_row.addWidget(
            QLabel("Backend connection:")
        )
        backend_row.addWidget(
            self.backend_status_value
        )

        ml_engine_row = QHBoxLayout()
        ml_engine_row.addWidget(
            QLabel("Local ML engine:")
        )
        ml_engine_row.addWidget(
            self.ml_engine_status_value
        )

        training_row = QHBoxLayout()
        training_row.addWidget(
            QLabel("Training jobs:")
        )
        training_row.addWidget(
            self.training_status_value
        )

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("System Status"))
        layout.addLayout(backend_row)
        layout.addLayout(ml_engine_row)
        layout.addLayout(training_row)

    def update_status(
        self,
        backend_status: str,
        ml_engine_status: str,
        training_status: str,
    ) -> None:
        """Update the displayed status values."""

        self.backend_status_value.setText(
            backend_status
        )

        self.ml_engine_status_value.setText(
            ml_engine_status
        )

        self.training_status_value.setText(
            training_status
        )