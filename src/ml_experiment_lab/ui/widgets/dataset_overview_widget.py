"""Dataset overview widget for the dashboard."""

from PySide6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)


class DatasetOverviewWidget(QWidget):
    """Display summary information about available datasets."""

    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Dataset Overview")

        self.dataset_count_value = QLabel("0")
        self.last_dataset_value = QLabel("No dataset uploaded")
        self.storage_used_value = QLabel("0 MB")

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Number of datasets:"))
        count_row.addWidget(self.dataset_count_value)

        last_dataset_row = QHBoxLayout()
        last_dataset_row.addWidget(QLabel("Last uploaded dataset:"))
        last_dataset_row.addWidget(self.last_dataset_value)

        storage_row = QHBoxLayout()
        storage_row.addWidget(QLabel("Storage used:"))
        storage_row.addWidget(self.storage_used_value)

        layout = QVBoxLayout(self)

        layout.addWidget(title)
        layout.addLayout(count_row)
        layout.addLayout(last_dataset_row)
        layout.addLayout(storage_row)

    def set_summary(
        self,
        dataset_count: int,
        last_dataset_name: str | None,
        storage_used_mb: float,
    ) -> None:
        """Update the displayed dataset summary."""

        self.dataset_count_value.setText(str(dataset_count))

        if last_dataset_name:
            self.last_dataset_value.setText(last_dataset_name)
        else:
            self.last_dataset_value.setText("No dataset uploaded")

        self.storage_used_value.setText(
            f"{storage_used_mb:.2f} MB"
        )