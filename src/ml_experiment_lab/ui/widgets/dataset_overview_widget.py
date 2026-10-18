
"""Компонент с обобщена информация за наборите от данни."""

from PySide6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)


class DatasetOverviewWidget(QWidget):
    """Показва обобщена информация за наличните набори от данни."""

    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Обобщение на наборите от данни")

        self.dataset_count_value = QLabel("0")
        self.last_dataset_value = QLabel(
            "Няма добавен набор от данни"
        )
        self.storage_used_value = QLabel("0 МБ")

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Брой набори от данни:"))
        count_row.addWidget(self.dataset_count_value)

        last_dataset_row = QHBoxLayout()
        last_dataset_row.addWidget(
            QLabel("Последно добавен набор:")
        )
        last_dataset_row.addWidget(self.last_dataset_value)

        storage_row = QHBoxLayout()
        storage_row.addWidget(QLabel("Използвано пространство:"))
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
        """Обновява обобщената информация за наборите от данни."""

        self.dataset_count_value.setText(str(dataset_count))

        if last_dataset_name:
            self.last_dataset_value.setText(last_dataset_name)
        else:
            self.last_dataset_value.setText(
                "Няма добавен набор от данни"
            )

        self.storage_used_value.setText(
            f"{storage_used_mb:.2f} МБ"
        )
