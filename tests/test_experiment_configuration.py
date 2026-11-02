"""Тестове за конфигурацията на ML експерименти."""

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.ui.pages.experiments_page import (
    ExperimentConfiguration,
    ExperimentsPage,
)


class FakeDatasetsPage(QObject):
    """Замества DatasetsPage при тестовете."""

    datasets_changed = Signal()

    def __init__(self, dataset: LoadedDataset) -> None:
        super().__init__()

        self._datasets = {
            str(dataset.path): dataset
        }

    @property
    def loaded_datasets(self) -> dict[str, LoadedDataset]:
        return dict(self._datasets)


class ExperimentConfigurationTests(unittest.TestCase):
    """Проверки за предаването на избраните данни."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance()

        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self) -> None:
        self.dataset = LoadedDataset(
            path=Path("sample.csv"),
            columns=("Възраст", "Град", "Клас"),
            rows=(
                ("20", "София", "A"),
                ("30", "Пловдив", "B"),
                ("40", "Варна", "A"),
            ),
        )

        self.datasets_page = FakeDatasetsPage(self.dataset)

        self.page = ExperimentsPage(
            datasets_page=self.datasets_page
        )

    def tearDown(self) -> None:
        self.page.close()

    def test_configuration_is_none_without_target(self) -> None:
        self.assertIsNone(self.page.experiment_configuration)

    def test_configuration_is_none_without_features(self) -> None:
        target_index = self.page._target_combo.findData("Клас")
        self.page._target_combo.setCurrentIndex(target_index)

        self.assertIsNone(self.page.experiment_configuration)

    def test_returns_complete_experiment_configuration(self) -> None:
        target_index = self.page._target_combo.findData("Клас")
        self.page._target_combo.setCurrentIndex(target_index)

        age_item = self.page._features_list.item(0)
        city_item = self.page._features_list.item(1)

        age_item.setCheckState(Qt.CheckState.Checked)
        city_item.setCheckState(Qt.CheckState.Checked)

        configuration = self.page.experiment_configuration

        self.assertIsInstance(
            configuration,
            ExperimentConfiguration,
        )

        self.assertIs(configuration.dataset, self.dataset)
        self.assertEqual(configuration.target_column, "Клас")
        self.assertEqual(
            configuration.feature_columns,
            ("Възраст", "Град"),
        )

    def test_target_column_is_not_in_features(self) -> None:
        target_index = self.page._target_combo.findData("Клас")
        self.page._target_combo.setCurrentIndex(target_index)

        target_item = self.page._features_list.item(2)

        self.assertEqual(
            target_item.checkState(),
            Qt.CheckState.Unchecked,
        )

        self.assertFalse(
            target_item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        )

    def test_configuration_is_immutable(self) -> None:
        target_index = self.page._target_combo.findData("Клас")
        self.page._target_combo.setCurrentIndex(target_index)

        self.page._features_list.item(0).setCheckState(
            Qt.CheckState.Checked
        )

        configuration = self.page.experiment_configuration

        with self.assertRaises(AttributeError):
            configuration.target_column = "Възраст"


if __name__ == "__main__":
    unittest.main()