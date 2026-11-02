"""Страница за конфигуриране на ML експерименти."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.ui.pages.datasets_page import DatasetsPage


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Конфигурация на ML експеримент."""

    dataset: LoadedDataset
    target_column: str
    feature_columns: tuple[str, ...]


class ExperimentsPage(QWidget):
    """Страница за избор и конфигуриране на ML експерименти."""

    def __init__(
        self,
        datasets_page: DatasetsPage | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._datasets_page = datasets_page
        self._datasets_by_key: dict[str, LoadedDataset] = {}
        self._active_dataset_key: str | None = None

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

    @property
    def selected_target_column(self) -> str | None:
        """Връща избраната целева колона."""
        return self._target_combo.currentData()

    @property
    def selected_feature_columns(self) -> list[str]:
        """Връща избраните входни колони."""
        selected_columns = []

        for index in range(self._features_list.count()):
            item = self._features_list.item(index)

            if item.checkState() == Qt.CheckState.Checked:
                column_name = item.data(
                    Qt.ItemDataRole.UserRole
                )
                selected_columns.append(column_name)

        return selected_columns

    @property
    def has_valid_configuration(self) -> bool:
        """Проверява дали са избрани Target и поне една Feature колона."""
        return (
            self.selected_dataset is not None
            and self.selected_target_column is not None
            and bool(self.selected_feature_columns)
        )

    @property
    def experiment_configuration(
        self,
    ) -> ExperimentConfiguration | None:
        """Връща пълната конфигурация на експеримента.

        Връща None, ако липсва набор от данни,
        целева колона или поне една входна колона.
        """

        if not self.has_valid_configuration:
            return None

        dataset = self.selected_dataset
        target_column = self.selected_target_column
        feature_columns = self.selected_feature_columns

        if (
            dataset is None
            or target_column is None
            or not feature_columns
        ):
            return None

        return ExperimentConfiguration(
            dataset=dataset,
            target_column=target_column,
            feature_columns=tuple(feature_columns),
        )

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)

        title = QLabel("Експерименти")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Изберете набор от данни, целева колона и входни "
            "характеристики за бъдещия ML експеримент."
        )
        subtitle.setWordWrap(True)

        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)

        # Група за избор на набор от данни.
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
            "Изберете набор, зареден в раздел „Набори от данни“."
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

        # Група за избор на Target и Features.
        columns_group = QGroupBox(
            "Целева колона и входни характеристики"
        )
        columns_layout = QVBoxLayout(columns_group)

        target_layout = QFormLayout()

        self._target_combo = QComboBox()
        self._target_combo.addItem(
            "— Изберете целева колона —",
            None,
        )
        self._target_combo.setEnabled(False)
        self._target_combo.setMinimumWidth(300)
        self._target_combo.setToolTip(
            "Изберете колоната, която ML моделът трябва да прогнозира."
        )

        target_layout.addRow(
            "Целева колона (Target):",
            self._target_combo,
        )

        columns_layout.addLayout(target_layout)

        features_label = QLabel(
            "Входни колони (Features):"
        )
        columns_layout.addWidget(features_label)

        features_hint = QLabel(
            "Отбележете колоните, които моделът ще използва "
            "като входни данни."
        )
        features_hint.setWordWrap(True)
        columns_layout.addWidget(features_hint)

        self._features_list = QListWidget()
        self._features_list.setEnabled(False)
        self._features_list.setMinimumHeight(120)
        self._features_list.setMaximumHeight(220)
        self._features_list.setToolTip(
            "Отбележете една или повече входни колони."
        )

        columns_layout.addWidget(self._features_list)

        page_layout.addWidget(columns_group)

        # Статус на конфигурацията.
        self._status_label = QLabel(
            "Все още няма избран набор от данни."
        )
        self._status_label.setWordWrap(True)

        page_layout.addWidget(self._status_label)
        page_layout.addStretch(1)

        # Свързване на сигналите.
        self._dataset_combo.currentIndexChanged.connect(
            self._on_dataset_changed
        )

        self._target_combo.currentIndexChanged.connect(
            self._on_target_changed
        )

        self._features_list.itemChanged.connect(
            self._on_feature_changed
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
            selected_key = next(iter(self._datasets_by_key))

        elif previous_key is not None and self._datasets_by_key:
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
        """Показва информация и колоните на избрания набор от данни."""
        del index

        dataset_key = self._dataset_combo.currentData()
        dataset = self.selected_dataset

        if dataset_key != self._active_dataset_key:
            self._active_dataset_key = dataset_key
            self._populate_column_selectors(dataset)

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

        self._update_configuration_status()

    def _populate_column_selectors(
        self,
        dataset: LoadedDataset | None,
    ) -> None:
        """Зарежда колоните в списъците за Target и Features."""
        columns = dataset.columns if dataset is not None else []

        self._target_combo.blockSignals(True)
        self._features_list.blockSignals(True)

        self._target_combo.clear()
        self._target_combo.addItem(
            "— Изберете целева колона —",
            None,
        )

        self._features_list.clear()

        for column in columns:
            self._target_combo.addItem(
                column,
                column,
            )

            feature_item = QListWidgetItem(column)
            feature_item.setData(
                Qt.ItemDataRole.UserRole,
                column,
            )
            feature_item.setFlags(
                feature_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            feature_item.setCheckState(
                Qt.CheckState.Unchecked
            )

            self._features_list.addItem(feature_item)

        has_enough_columns = len(columns) >= 2

        self._target_combo.setEnabled(has_enough_columns)
        self._features_list.setEnabled(has_enough_columns)

        self._target_combo.blockSignals(False)
        self._features_list.blockSignals(False)

        self._apply_target_exclusion(None)

    def _on_target_changed(self, index: int) -> None:
        """Изключва целевата колона от списъка с входни колони."""
        del index

        self._apply_target_exclusion(
            self.selected_target_column
        )

        self._update_configuration_status()

    def _apply_target_exclusion(
        self,
        target_column: str | None,
    ) -> None:
        """Не позволява Target да бъде избрана и като Feature."""
        self._features_list.blockSignals(True)

        for index in range(self._features_list.count()):
            item = self._features_list.item(index)

            column_name = item.data(
                Qt.ItemDataRole.UserRole
            )

            if column_name == target_column:
                item.setCheckState(Qt.CheckState.Unchecked)

                item.setFlags(
                    item.flags()
                    & ~Qt.ItemFlag.ItemIsUserCheckable
                )
            else:
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                )

        self._features_list.blockSignals(False)

    def _on_feature_changed(
        self,
        item: QListWidgetItem,
    ) -> None:
        """Обновява статуса при промяна на входните колони."""
        del item

        self._update_configuration_status()

    def _update_configuration_status(self) -> None:
        """Показва текущото състояние на конфигурацията."""
        dataset = self.selected_dataset

        if dataset is None:
            return

        if dataset.column_count < 2:
            self._status_label.setText(
                "Наборът от данни трябва да съдържа поне две колони: "
                "една целева и поне една входна."
            )
            return

        target_column = self.selected_target_column
        feature_columns = self.selected_feature_columns

        if target_column is None:
            self._status_label.setText(
                "Изберете целева колона (Target)."
            )
            return

        if not feature_columns:
            self._status_label.setText(
                f"Целева колона: „{target_column}“. "
                "Изберете поне една входна колона (Feature)."
            )
            return

        self._status_label.setText(
            f"Конфигурацията е готова: целева колона "
            f"„{target_column}“ и {len(feature_columns)} "
            "избрани входни колони."
        )