"""Страница за избор и настройка на ML модели."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.training import (
    TaskType,
    get_available_algorithms,
)
from ml_experiment_lab.ui.pages.experiments_page import (
    ExperimentsPage,
)


class ModelsPage(QWidget):
    """Страница за конфигуриране и обучение на ML модели."""

    def __init__(
        self,
        experiments_page: ExperimentsPage | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._experiments_page = experiments_page

        self._build_ui()

    @property
    def selected_task_type(self) -> TaskType:
        """Връща избрания тип ML задача."""
        return self._task_combo.currentData()

    @property
    def selected_algorithm(self) -> str | None:
        """Връща избрания ML алгоритъм."""
        return self._algorithm_combo.currentData()

    @property
    def test_size(self) -> float:
        """Връща дела на тестовата извадка между 0 и 1."""
        return self._test_size_spin.value() / 100

    @property
    def random_state(self) -> int:
        """Връща началната стойност за възпроизводимо разделяне."""
        return self._random_state_spin.value()

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)

        # Заглавие на страницата.
        title = QLabel("Модели")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Изберете тип задача, алгоритъм и настройки "
            "за обучение на машинно обучаващ модел."
        )
        subtitle.setWordWrap(True)

        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)

        # Информация за конфигурацията от ExperimentsPage.
        configuration_group = QGroupBox(
            "Конфигурация на експеримента"
        )
        configuration_layout = QVBoxLayout(configuration_group)

        self._configuration_label = QLabel(
            "Информацията за експеримента ще се показва тук."
        )
        self._configuration_label.setWordWrap(True)

        configuration_layout.addWidget(
            self._configuration_label
        )

        page_layout.addWidget(configuration_group)

        # Настройки на ML модела.
        model_group = QGroupBox("Настройки на ML модела")
        model_layout = QFormLayout(model_group)

        # Тип на ML задачата.
        self._task_combo = QComboBox()

        self._task_combo.addItem(
            "Класификация",
            "classification",
        )

        self._task_combo.addItem(
            "Регресия",
            "regression",
        )

        self._task_combo.setToolTip(
            "Изберете дали моделът трябва да прогнозира "
            "категория или числова стойност."
        )

        model_layout.addRow(
            "Тип задача:",
            self._task_combo,
        )

        # Алгоритъм.
        self._algorithm_combo = QComboBox()

        self._algorithm_combo.setMinimumWidth(280)
        self._algorithm_combo.setToolTip(
            "Изберете алгоритъм, подходящ за типа на задачата."
        )

        model_layout.addRow(
            "Алгоритъм:",
            self._algorithm_combo,
        )

        # Размер на тестовата извадка.
        self._test_size_spin = QDoubleSpinBox()
        self._test_size_spin.setRange(5, 50)
        self._test_size_spin.setDecimals(0)
        self._test_size_spin.setSingleStep(5)
        self._test_size_spin.setValue(20)
        self._test_size_spin.setSuffix(" %")
        self._test_size_spin.setToolTip(
            "Определя какъв процент от редовете ще бъде "
            "използван за тестване на модела."
        )

        model_layout.addRow(
            "Тестова извадка:",
            self._test_size_spin,
        )

        # Random state.
        self._random_state_spin = QSpinBox()
        self._random_state_spin.setRange(
            0,
            2_147_483_647,
        )
        self._random_state_spin.setValue(42)
        self._random_state_spin.setToolTip(
            "Фиксирана начална стойност за възпроизводимо "
            "разделяне на данните и обучение."
        )

        model_layout.addRow(
            "Random state:",
            self._random_state_spin,
        )

        page_layout.addWidget(model_group)

        # Допълнителна информация.
        information_label = QLabel(
            "Можете да променяте настройките преди обучението. "
            "Избраният алгоритъм зависи от типа на ML задачата. "
            "Обучението и оценяването ще бъдат добавени."
        )
        information_label.setWordWrap(True)

        page_layout.addWidget(information_label)

        page_layout.addStretch(1)

        # Свързване на сигналите.
        self._task_combo.currentIndexChanged.connect(
            self._on_task_type_changed
        )

        # Зареждаме алгоритмите за първоначално избрания тип задача.
        self._update_algorithms()

    def _on_task_type_changed(self, index: int) -> None:
        """Обновява алгоритмите при смяна на типа задача."""
        del index

        self._update_algorithms()

    def _update_algorithms(self) -> None:
        """Зарежда подходящите алгоритми за избрания тип задача."""
        task_type = self.selected_task_type

        self._algorithm_combo.blockSignals(True)
        self._algorithm_combo.clear()

        for algorithm_id, algorithm_name in get_available_algorithms(
            task_type
        ):
            self._algorithm_combo.addItem(
                algorithm_name,
                algorithm_id,
            )

        self._algorithm_combo.blockSignals(False)

    def refresh_experiment_configuration(self) -> None:
        """Обновява информацията за избрания експеримент."""
        if self._experiments_page is None:
            self._configuration_label.setText(
                "Страницата за експерименти не е свързана."
            )
            return

        configuration = (
            self._experiments_page.experiment_configuration
        )

        if configuration is None:
            self._configuration_label.setText(
                "Няма завършена конфигурация. "
                "Отворете раздел „Експерименти“ и изберете "
                "набор от данни, Target и поне една Feature колона."
            )
            return

        dataset = configuration.dataset

        details = [
            f"Файл: {dataset.path.name}",
            f"Целева колона: {configuration.target_column}",
            (
                "Входни колони: "
                + ", ".join(configuration.feature_columns)
            ),
            f"Брой входни характеристики: "
            f"{len(configuration.feature_columns)}",
        ]

        if dataset.sheet_name is not None:
            details.append(
                f"Работен лист: {dataset.sheet_name}"
            )

        self._configuration_label.setText(
            "\n".join(details)
        )

    def showEvent(self, event) -> None:
        """Обновява конфигурацията при отваряне на страницата."""
        super().showEvent(event)

        self.refresh_experiment_configuration()