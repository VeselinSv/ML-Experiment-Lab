"""Страница за избор на набори от данни за бъдещи експерименти."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.ui.pages.datasets_page import DatasetsPage


class ExperimentsPage(QWidget):
    """Начална страница за конфигуриране на ML експерименти."""

    def __init__(
        self,
        datasets_page: DatasetsPage | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._datasets_page = datasets_page
        self._datasets_by_key: dict[str, LoadedDataset] = {}

        self._build_ui()

        if self._datasets_page is not None:
            self._datasets_page.datasets_changed.connect(
                self.refresh_datasets
            )
            self.refresh_datasets()

    @property
    def selected_dataset(self) -> LoadedDataset | None:
        """Връща избрания набор от данни, ако има такъв."""
        dataset_key = self._dataset_combo.currentData()
        return self._datasets_by_key.get(dataset_key)

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)

        title = QLabel("Експерименти")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Изберете набор от данни, с който ще работите в новия експеримент. "
            "Следващите стъпки ще позволят избор на целева колона "
            "и входни характеристики."
        )
        subtitle.setWordWrap(True)

        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)

        dataset_group = QGroupBox("Избор на набор от данни")
        dataset_layout = QFormLayout(dataset_group)

        self._dataset_combo = QComboBox()
        self._dataset_combo.addItem(
            "Няма заредени набори от данни",
            None,
        )
        self._dataset_combo.setEnabled(False)
        self._dataset_combo.setMinimumWidth(300)
        self._dataset_combo.setToolTip(
            "Изберете един от наборите, заредени в раздел „Набори от данни“."
        )

        dataset_layout.addRow(
            "Набор от данни:",
            self._dataset_combo,
        )

        self._dataset_details_label = QLabel(
            "Заредете CSV или Excel файл от раздел „Набори от данни“, "
            "за да го изберете тук."
        )
        self._dataset_details_label.setWordWrap(True)

        dataset_layout.addRow(
            "Информация:",
            self._dataset_details_label,
        )

        page_layout.addWidget(dataset_group)

        self._status_label = QLabel(
            "Все още няма избран набор от данни."
        )
        self._status_label.setWordWrap(True)

        page_layout.addWidget(self._status_label)
        page_layout.addStretch(1)

        self._dataset_combo.currentIndexChanged.connect(
            self._on_dataset_changed
        )

    def refresh_datasets(self) -> None:
        """Обновява избора при добавяне или премахване на набор от данни."""
        if self._datasets_page is None:
            return

        previous_key = self._dataset_combo.currentData()
        previous_datasets = self._datasets_by_key
        new_datasets = self._datasets_page.loaded_datasets

        self._datasets_by_key = new_datasets

        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()

        self._dataset_combo.addItem(
            "— Изберете набор от данни —",
            None,
        )

        for dataset_key, dataset in self._datasets_by_key.items():
            display_name = dataset.path.name

            if dataset.sheet_name is not None:
                display_name += f" — лист: {dataset.sheet_name}"

            self._dataset_combo.addItem(
                display_name,
                dataset_key,
            )

        if previous_key in self._datasets_by_key:
            selected_key = previous_key

        elif not previous_datasets and self._datasets_by_key:
            # При първото зареждане автоматично избираме първия набор.
            selected_key = next(iter(self._datasets_by_key))

        elif previous_key is not None and self._datasets_by_key:
            # Ако избраният набор е премахнат, избираме наличен набор.
            selected_key = next(iter(self._datasets_by_key))

        else:
            selected_key = None

        if selected_key is not None:
            self._dataset_combo.setCurrentIndex(
                self._dataset_combo.findData(selected_key)
            )
        else:
            self._dataset_combo.setCurrentIndex(0)

        self._dataset_combo.setEnabled(bool(self._datasets_by_key))
        self._dataset_combo.blockSignals(False)

        self._on_dataset_changed(self._dataset_combo.currentIndex())

    def _on_dataset_changed(self, index: int) -> None:
        """Показва информация за избрания набор от данни."""
        del index

        dataset = self.selected_dataset

        if dataset is None:
            if self._datasets_by_key:
                self._dataset_details_label.setText(
                    "Няма избран набор от данни."
                )
                self._status_label.setText(
                    "Изберете набор от данни от падащия списък."
                )
            else:
                self._dataset_details_label.setText(
                    "Няма налична информация. Заредете набор от данни, "
                    "за да продължите."
                )
                self._status_label.setText(
                    "Заредете CSV или Excel файл от раздел „Набори от данни“."
                )
            return

        details = [
            f"Файл: {dataset.path.name}",
            f"Редове: {dataset.row_count}",
            f"Колони: {dataset.column_count}",
        ]

        if dataset.sheet_name is not None:
            details.append(
                f"Работен лист: {dataset.sheet_name}"
            )

        self._dataset_details_label.setText(
            "\n".join(details)
        )

        self._status_label.setText(
            "Наборът от данни е избран. Следващата стъпка ще добави "
            "избор на целева колона и входни характеристики."
        )