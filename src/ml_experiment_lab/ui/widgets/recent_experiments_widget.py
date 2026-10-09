"""Recent experiments widget for the dashboard."""

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
    """Display a table containing recent ML experiments."""

    def __init__(self) -> None:
        super().__init__()

        self.title_label = QLabel("Recent Experiments")

        self.empty_label = QLabel(
            "No experiments yet."
        )

        self.experiments_table = QTableWidget(0, 3)

        self.experiments_table.setHorizontalHeaderLabels(
            ["Experiment", "Model", "Status"]
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

        self.experiments_table.verticalHeader().setVisible(
            False
        )

        self.experiments_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.experiments_table.setMinimumHeight(150)

        layout = QVBoxLayout(self)

        layout.addWidget(self.title_label)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.experiments_table)

        self.set_experiments([])

    def set_experiments(
        self,
        experiments: Sequence[Mapping[str, str]],
    ) -> None:
        """Replace the displayed experiments with the supplied data.

        Each experiment must provide these keys:
        - name
        - model
        - status
        """

        self.experiments_table.setRowCount(
            len(experiments)
        )

        for row_index, experiment in enumerate(experiments):
            values = (
                experiment.get("name", "Unnamed experiment"),
                experiment.get("model", "Unknown model"),
                experiment.get("status", "Unknown"),
            )

            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)

                self.experiments_table.setItem(
                    row_index,
                    column_index,
                    item,
                )

        has_experiments = len(experiments) > 0

        self.empty_label.setVisible(
            not has_experiments
        )

        self.experiments_table.setVisible(
            has_experiments
        )