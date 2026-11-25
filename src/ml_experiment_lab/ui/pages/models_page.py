"""Страница за избор и настройка на ML модели."""

from __future__ import annotations

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.preprocessing import (
    CATEGORICAL_IMPUTATION_STRATEGIES,
    ENCODING_METHODS,
    NUMERIC_IMPUTATION_STRATEGIES,
    SCALING_METHODS,
    PreprocessingConfiguration,
)

from ml_experiment_lab.training import (
    TaskType,
    get_available_algorithms,
)

from ml_experiment_lab.ui.pages.experiments_page import (
    ExperimentsPage,
)

from ml_experiment_lab.ui.widgets.preprocessing_settings_widget import (
    PreprocessingSettingsWidget,
)


TASK_TYPE_NAMES = {
    "classification": "Класификация",
    "regression": "Регресия",
}


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

        self.refresh_experiment_configuration()

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

    @property
    def preprocessing_configuration(
        self,
    ) -> PreprocessingConfiguration | None:
        """Връща настройките за предварителна обработка.

        При невалидни настройки връща None.
        """

        return (
            self._preprocessing_settings_widget
            .preprocessing_configuration
        )

    @property
    def is_preprocessing_valid(self) -> bool:
        """Проверява валидността на Preprocessing настройките."""

        return (
            self._preprocessing_settings_widget
            .is_valid
        )

    @property
    def is_configuration_valid(self) -> bool:
        """Проверява дали основната конфигурация е завършена.

        Проверката включва:
            - избран експеримент;
            - избран тип ML задача;
            - наличен ML алгоритъм;
            - валидни Preprocessing настройки.

        Не проверява дали конкретните данни позволяват
        успешно обучение на избрания алгоритъм.
        """

        if self._experiments_page is None:
            return False

        if (
            self._experiments_page.experiment_configuration
            is None
        ):
            return False

        task_type = self.selected_task_type

        if task_type not in TASK_TYPE_NAMES:
            return False

        available_algorithms = dict(
            get_available_algorithms(task_type)
        )

        if self.selected_algorithm not in available_algorithms:
            return False

        return self.is_preprocessing_valid

    def _build_ui(self) -> None:
        """Изгражда графичния интерфейс на страницата."""

        page_layout = QVBoxLayout(self)

        page_layout.setSpacing(12)

        # Заглавието остава извън scroll областта.
        title = QLabel("Модели")

        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Изберете тип задача, алгоритъм, настройки "
            "за обучение и предварителна обработка на данните."
        )

        subtitle.setWordWrap(True)

        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)

        # Scroll област за основното съдържание.
        self._scroll_area = QScrollArea()

        self._scroll_area.setWidgetResizable(True)

        self._scroll_area.setFrameShape(
            QFrame.Shape.NoFrame
        )

        self._scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        page_layout.addWidget(
            self._scroll_area,
            1,
        )

        # Контейнер за съдържанието на страницата.
        content_widget = QWidget()

        content_layout = QVBoxLayout(
            content_widget
        )

        content_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        content_layout.setSpacing(12)

        self._scroll_area.setWidget(
            content_widget
        )

        # Информация за избрания експеримент.
        content_layout.addWidget(
            self._build_experiment_group()
        )

        # Настройки на ML модела.
        content_layout.addWidget(
            self._build_model_group()
        )

        # Настройки за предварителна обработка.
        content_layout.addWidget(
            self._build_preprocessing_group()
        )

        # Обобщение на цялостната конфигурация.
        content_layout.addWidget(
            self._build_summary_group()
        )

        information_label = QLabel(
            "Можете да променяте настройките преди обучението. "
            "Параметрите на предварителната обработка ще бъдат "
            "научени единствено от тренировъчните данни. "
            "Стартирането на обучението и визуализацията "
            "на резултатите ще бъдат добавени впоследствие."
        )

        information_label.setWordWrap(True)

        content_layout.addWidget(
            information_label
        )

        content_layout.addStretch(1)

        # Свързваме сигналите след създаване
        # на всички графични компоненти.
        self._connect_signals()

        # Зареждаме алгоритмите за първоначалната задача.
        self._update_algorithms()

    def _build_experiment_group(self) -> QGroupBox:
        """Изгражда секцията за избрания експеримент."""

        group = QGroupBox(
            "Конфигурация на експеримента"
        )

        layout = QVBoxLayout(group)

        self._configuration_label = QLabel(
            "Информацията за експеримента ще се показва тук."
        )

        self._configuration_label.setWordWrap(True)

        self._configuration_label.setTextFormat(
            Qt.TextFormat.PlainText
        )

        layout.addWidget(
            self._configuration_label
        )

        return group

    def _build_model_group(self) -> QGroupBox:
        """Изгражда секцията за настройките на ML модела."""

        group = QGroupBox(
            "Настройки на ML модела"
        )

        layout = QFormLayout(group)

        layout.setSpacing(8)

        layout.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )

        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        # Тип ML задача.
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

        layout.addRow(
            "Тип задача:",
            self._task_combo,
        )

        # Алгоритъм.
        self._algorithm_combo = QComboBox()

        self._algorithm_combo.setMinimumWidth(280)

        self._algorithm_combo.setToolTip(
            "Изберете алгоритъм, подходящ за типа на задачата."
        )

        layout.addRow(
            "Алгоритъм:",
            self._algorithm_combo,
        )

        # Размер на тестовата извадка.
        self._test_size_spin = QDoubleSpinBox()

        self._test_size_spin.setRange(
            5,
            50,
        )

        self._test_size_spin.setDecimals(0)

        self._test_size_spin.setSingleStep(5)

        self._test_size_spin.setValue(20)

        self._test_size_spin.setSuffix(" %")

        self._test_size_spin.setToolTip(
            "Определя какъв процент от редовете ще бъде "
            "използван за тестване на модела."
        )

        layout.addRow(
            "Тестова извадка:",
            self._test_size_spin,
        )

        # Random State.
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

        layout.addRow(
            "Начално състояние (Random State):",
            self._random_state_spin,
        )

        return group

    def _build_preprocessing_group(self) -> QGroupBox:
        """Изгражда секцията за предварителна обработка."""

        group = QGroupBox(
            "Предварителна обработка на данните"
        )

        layout = QVBoxLayout(group)

        self._preprocessing_settings_widget = (
            PreprocessingSettingsWidget()
        )

        layout.addWidget(
            self._preprocessing_settings_widget
        )

        return group

    def _build_summary_group(self) -> QGroupBox:
        """Изгражда обобщението на текущите настройки."""

        group = QGroupBox(
            "Обобщение на конфигурацията"
        )

        layout = QVBoxLayout(group)

        layout.setSpacing(8)

        self._model_summary_label = QLabel()

        self._model_summary_label.setWordWrap(True)

        self._model_summary_label.setTextFormat(
            Qt.TextFormat.PlainText
        )

        layout.addWidget(
            self._model_summary_label
        )

        separator = QFrame()

        separator.setFrameShape(
            QFrame.Shape.HLine
        )

        separator.setFrameShadow(
            QFrame.Shadow.Sunken
        )

        layout.addWidget(
            separator
        )

        self._preprocessing_summary_label = QLabel()

        self._preprocessing_summary_label.setWordWrap(
            True
        )

        self._preprocessing_summary_label.setTextFormat(
            Qt.TextFormat.PlainText
        )

        layout.addWidget(
            self._preprocessing_summary_label
        )

        self._configuration_status_label = QLabel()

        self._configuration_status_label.setWordWrap(
            True
        )

        self._configuration_status_label.setTextFormat(
            Qt.TextFormat.PlainText
        )

        layout.addWidget(
            self._configuration_status_label
        )

        return group

    def _connect_signals(self) -> None:
        """Свързва промените в настройките с обобщението."""

        self._task_combo.currentIndexChanged.connect(
            self._on_task_type_changed
        )

        self._algorithm_combo.currentIndexChanged.connect(
            self._on_model_settings_changed
        )

        self._test_size_spin.valueChanged.connect(
            self._on_model_settings_changed
        )

        self._random_state_spin.valueChanged.connect(
            self._on_model_settings_changed
        )

        self._preprocessing_settings_widget.configuration_changed.connect(
            self._on_preprocessing_settings_changed
        )

    def _on_task_type_changed(
        self,
        index: int,
    ) -> None:
        """Обновява алгоритмите при смяна на типа задача."""

        del index

        self._update_algorithms()

    def _on_model_settings_changed(
        self,
        *_args,
    ) -> None:
        """Обновява отчета при промяна на ML настройките."""

        self._update_configuration_summary()

    def _on_preprocessing_settings_changed(
        self,
    ) -> None:
        """Обновява отчета при промяна на Preprocessing."""

        self._update_configuration_summary()

    def _update_algorithms(self) -> None:
        """Зарежда алгоритмите за избрания тип задача."""

        task_type = self.selected_task_type

        previous_signal_state = (
            self._algorithm_combo.blockSignals(True)
        )

        try:

            self._algorithm_combo.clear()

            if task_type in TASK_TYPE_NAMES:

                for (
                    algorithm_id,
                    algorithm_name,
                ) in get_available_algorithms(task_type):

                    self._algorithm_combo.addItem(
                        algorithm_name,
                        algorithm_id,
                    )

        finally:

            self._algorithm_combo.blockSignals(
                previous_signal_state
            )

        self._update_configuration_summary()

    def _update_model_summary(self) -> None:
        """Обновява обобщението за ML модела."""

        task_name = TASK_TYPE_NAMES.get(
            self.selected_task_type,
            "Не е избран",
        )

        algorithm_name = (
            self._algorithm_combo.currentText()
        )

        if not algorithm_name:
            algorithm_name = "Не е избран"

        test_percentage = (
            self._test_size_spin.value()
        )

        training_percentage = (
            100 - test_percentage
        )

        details = [
            "Настройки на ML модела:",
            f"Тип задача: {task_name}",
            f"Алгоритъм: {algorithm_name}",
            (
                "Тренировъчна извадка: "
                f"{training_percentage:.0f}%"
            ),
            (
                "Тестова извадка: "
                f"{test_percentage:.0f}%"
            ),
            (
                "Начално състояние (Random State): "
                f"{self.random_state}"
            ),
        ]

        self._model_summary_label.setText(
            "\n".join(details)
        )

    def _update_preprocessing_summary(self) -> None:
        """Обновява обобщението за предварителната обработка."""

        configuration = (
            self.preprocessing_configuration
        )

        if configuration is None:

            self._preprocessing_summary_label.setText(
                "Предварителна обработка:\n"
                "Невалидни настройки. "
                "Коригирайте въведените стойности "
                "в секцията за предварителна обработка."
            )

            return

        numeric_strategy_name = (
            NUMERIC_IMPUTATION_STRATEGIES[
                configuration.numeric_imputation_strategy
            ]
        )

        categorical_strategy_name = (
            CATEGORICAL_IMPUTATION_STRATEGIES[
                configuration.categorical_imputation_strategy
            ]
        )

        scaling_name = SCALING_METHODS[
            configuration.scaling_method
        ]

        encoding_name = ENCODING_METHODS[
            configuration.encoding_method
        ]

        drop_empty_text = (
            "Да"
            if configuration.drop_empty_columns
            else "Не"
        )

        drop_constant_text = (
            "Да"
            if configuration.drop_constant_features
            else "Не"
        )

        details = [
            "Предварителна обработка:",
            f"Числово попълване: {numeric_strategy_name}",
        ]

        # Числовата константа се показва само
        # когато действително ще се използва.
        if (
            configuration.numeric_imputation_strategy
            == "constant"
        ):

            details.append(
                "Числова константа: "
                f"{configuration.numeric_fill_value}"
            )

        details.append(
            "Категориално попълване: "
            f"{categorical_strategy_name}"
        )

        # Категориалната константа може да се използва
        # и като резервна стойност за празни колони.
        if (
            configuration.categorical_imputation_strategy
            == "constant"
        ):

            details.append(
                "Категориална константа: "
                f"{configuration.categorical_fill_value}"
            )

        elif not configuration.drop_empty_columns:

            details.append(
                "Резервна категориална константа: "
                f"{configuration.categorical_fill_value}"
            )

        details.extend(
            [
                f"Мащабиране: {scaling_name}",
                f"Категориално кодиране: {encoding_name}",
                (
                    "Премахване на изцяло празни колони: "
                    f"{drop_empty_text}"
                ),
                (
                    "Премахване на постоянни характеристики: "
                    f"{drop_constant_text}"
                ),
                "Автоматично извличане на датни характеристики: Да",
            ]
        )

        self._preprocessing_summary_label.setText(
            "\n".join(details)
        )

    def _update_configuration_status(self) -> None:
        """Обновява състоянието на цялостната конфигурация."""

        problems = []

        # Проверяваме връзката с ExperimentsPage.
        if self._experiments_page is None:

            problems.append(
                "Страницата за експерименти не е свързана."
            )

        elif (
            self._experiments_page.experiment_configuration
            is None
        ):

            problems.append(
                "Не е завършена конфигурацията "
                "на експеримента."
            )

        # Проверяваме типа задача.
        if self.selected_task_type not in TASK_TYPE_NAMES:

            problems.append(
                "Не е избран валиден тип ML задача."
            )

        else:

            available_algorithms = dict(
                get_available_algorithms(
                    self.selected_task_type
                )
            )

            if (
                self.selected_algorithm
                not in available_algorithms
            ):

                problems.append(
                    "Не е избран валиден ML алгоритъм."
                )

        # Проверяваме настройките за Preprocessing.
        if not self.is_preprocessing_valid:

            problems.append(
                "Настройките за предварителна "
                "обработка са невалидни."
            )

        if problems:

            status_lines = [
                "Състояние: Конфигурацията не е готова."
            ]

            status_lines.extend(
                f"• {problem}"
                for problem in problems
            )

            self._configuration_status_label.setText(
                "\n".join(status_lines)
            )

        else:

            self._configuration_status_label.setText(
                "Състояние: Конфигурацията е валидна.\n"
                "Настройките са готови за предаване "
                "към механизма за обучение."
            )

    def _update_configuration_summary(self) -> None:
        """Обновява всички секции на обобщението."""

        self._update_model_summary()

        self._update_preprocessing_summary()

        self._update_configuration_status()

    def refresh_experiment_configuration(self) -> None:
        """Обновява информацията за избрания експеримент."""

        if self._experiments_page is None:

            self._configuration_label.setText(
                "Страницата за експерименти не е свързана."
            )

            self._update_configuration_summary()

            return

        configuration = (
            self._experiments_page.experiment_configuration
        )

        if configuration is None:

            self._configuration_label.setText(
                "Няма завършена конфигурация. "
                "Отворете раздел „Експерименти“ и изберете "
                "набор от данни, целева колона "
                "и поне една входна колона."
            )

            self._update_configuration_summary()

            return

        dataset = configuration.dataset

        details = [
            f"Файл: {dataset.path.name}",
            f"Целева колона: {configuration.target_column}",
            (
                "Входни колони: "
                + ", ".join(
                    configuration.feature_columns
                )
            ),
            (
                "Брой входни характеристики: "
                f"{len(configuration.feature_columns)}"
            ),
        ]

        if dataset.sheet_name is not None:

            details.append(
                f"Работен лист: {dataset.sheet_name}"
            )

        self._configuration_label.setText(
            "\n".join(details)
        )

        self._update_configuration_summary()

    def showEvent(self, event) -> None:
        """Обновява конфигурацията при отваряне на страницата."""

        super().showEvent(event)

        self.refresh_experiment_configuration()