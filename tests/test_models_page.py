"""Тестове за страницата за избор и настройка на ML модели."""

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.ui.pages.experiments_page import (
    ExperimentConfiguration,
)
from ml_experiment_lab.ui.pages.models_page import ModelsPage
from ml_experiment_lab.training import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
)


class FakeExperimentsPage:
    """Замества ExperimentsPage при тестовете."""

    def __init__(
        self,
        configuration: ExperimentConfiguration | None = None,
    ) -> None:
        self._configuration = configuration

    @property
    def experiment_configuration(
        self,
    ) -> ExperimentConfiguration | None:
        return self._configuration


class ModelsPageTests(unittest.TestCase):
    """Проверки за интерфейса и настройките на ML моделите."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance()

        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self) -> None:
        dataset = LoadedDataset(
            path=Path("customers.csv"),
            columns=("Възраст", "Доход", "Покупка"),
            rows=(
                ("20", "2000", "Да"),
                ("30", "3000", "Не"),
                ("40", "4000", "Да"),
            ),
        )

        self.configuration = ExperimentConfiguration(
            dataset=dataset,
            target_column="Покупка",
            feature_columns=("Възраст", "Доход"),
        )

        self.experiments_page = FakeExperimentsPage(
            self.configuration
        )

        self.page = ModelsPage(
            experiments_page=self.experiments_page
        )

    def tearDown(self) -> None:
        self.page.close()

    def _algorithm_ids(self) -> list[str]:
        """Връща идентификаторите на показаните алгоритми."""
        return [
            self.page._algorithm_combo.itemData(index)
            for index in range(self.page._algorithm_combo.count())
        ]

    def test_default_task_type_is_classification(self) -> None:
        self.assertEqual(
            self.page.selected_task_type,
            "classification",
        )

    def test_classification_algorithms_are_displayed(self) -> None:
        self.assertEqual(
            self._algorithm_ids(),
            list(CLASSIFICATION_ALGORITHMS.keys()),
        )

    def test_regression_algorithms_are_displayed_after_selection(
        self,
    ) -> None:
        index = self.page._task_combo.findData("regression")
        self.page._task_combo.setCurrentIndex(index)

        self.assertEqual(
            self.page.selected_task_type,
            "regression",
        )

        self.assertEqual(
            self._algorithm_ids(),
            list(REGRESSION_ALGORITHMS.keys()),
        )

    def test_algorithm_selection_updates_property(self) -> None:
        index = self.page._algorithm_combo.findData(
            "random_forest"
        )
        self.page._algorithm_combo.setCurrentIndex(index)

        self.assertEqual(
            self.page.selected_algorithm,
            "random_forest",
        )

    def test_default_test_size_is_twenty_percent(self) -> None:
        self.assertEqual(
            self.page.test_size,
            0.2,
        )

    def test_custom_test_size_is_returned_as_decimal(self) -> None:
        self.page._test_size_spin.setValue(30)

        self.assertEqual(
            self.page.test_size,
            0.3,
        )

    def test_default_random_state_is_42(self) -> None:
        self.assertEqual(
            self.page.random_state,
            42,
        )

    def test_custom_random_state_is_returned(self) -> None:
        self.page._random_state_spin.setValue(123)

        self.assertEqual(
            self.page.random_state,
            123,
        )

    def test_experiment_configuration_is_displayed(self) -> None:
        self.page.refresh_experiment_configuration()

        displayed_text = self.page._configuration_label.text()

        self.assertIn("customers.csv", displayed_text)
        self.assertIn("Покупка", displayed_text)
        self.assertIn("Възраст", displayed_text)
        self.assertIn("Доход", displayed_text)

    def test_configuration_is_updated_when_refreshed(self) -> None:
        self.page.refresh_experiment_configuration()

        updated_configuration = ExperimentConfiguration(
            dataset=self.configuration.dataset,
            target_column="Доход",
            feature_columns=("Възраст",),
        )

        self.experiments_page._configuration = updated_configuration

        self.page.refresh_experiment_configuration()

        displayed_text = self.page._configuration_label.text()

        self.assertIn("Доход", displayed_text)
        self.assertIn("Възраст", displayed_text)

    def test_missing_configuration_displays_helpful_message(self) -> None:
        self.experiments_page._configuration = None

        self.page.refresh_experiment_configuration()

        self.assertIn(
            "Няма завършена конфигурация",
            self.page._configuration_label.text(),
        )

    def test_missing_experiments_page_is_handled(self) -> None:
        page = ModelsPage()

        try:
            page.refresh_experiment_configuration()

            self.assertIn(
                "не е свързана",
                page._configuration_label.text(),
            )
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()