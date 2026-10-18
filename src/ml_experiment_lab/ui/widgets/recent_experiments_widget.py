
"""Компонент за последните експерименти в началното табло."""

from collections.abc import Mapping, Sequence

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RecentExperimentsWidget(QWidget):
    """Показва таблица с последните експерименти."""

    def __init__(self) -> None:
        super().__init__()

        self.title_label = QLabel("Последни експерименти")
        self.empty_label = QLabel(
            "Все още няма създадени експерименти."
        )

        self.experiments_table = QTableWidget(0, 3)

        self.experiments_table.setHorizontalHeaderLabels(
            ["Експеримент", "Модел", "Състояние"]
        )

        self.experiments_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.experiments_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.experiments_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.experiments_table.verticalHeader().setVisible(False)
        self.experiments_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.experiments_table.setMinimumHeight(150)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.experiments_table)

        self.set_experiments([])

    @staticmethod
    def _translate_status(status: str) -> str:
        """Превежда стандартните състояния на експериментите."""
        translations = {
            "pending": "Изчаква",
            "queued": "На опашка",
            "running": "В процес на изпълнение",
            "completed": "Завършен",
            "complete": "Завършен",
            "success": "Успешен",
            "succeeded": "Успешен",
            "failed": "Неуспешен",
            "error": "Грешка",
            "cancelled": "Отменен",
            "canceled": "Отменен",
            "stopped": "Спрян",
            "unknown": "Неизвестно",
        }

        return translations.get(status.strip().casefold(), status)

    def set_experiments(
        self,
        experiments: Sequence[Mapping[str, str]],
    ) -> None:
        """Заменя показаните експерименти с подадените данни.

        Всеки експеримент трябва да съдържа ключовете:
        - name
        - model
        - status
        """
        self.experiments_table.setRowCount(len(experiments))

        for row_index, experiment in enumerate(experiments):
            values = (
                experiment.get("name", "Експеримент без име"),
                experiment.get("model", "Неизвестен модел"),
                self._translate_status(
                    experiment.get("status", "Неизвестно")
                ),
            )

            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.experiments_table.setItem(
                    row_index,
                    column_index,
                    item,
                )

        has_experiments = len(experiments) > 0

        self.empty_label.setVisible(not has_experiments)
        self.experiments_table.setVisible(has_experiments)
