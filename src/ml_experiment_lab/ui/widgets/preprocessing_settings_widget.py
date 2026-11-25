"""Графичен компонент за настройки на предварителната обработка."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.preprocessing import (
    CATEGORICAL_IMPUTATION_STRATEGIES,
    ENCODING_METHODS,
    NUMERIC_IMPUTATION_STRATEGIES,
    SCALING_METHODS,
    PreprocessingConfiguration,
    PreprocessingError,
    parse_numeric_value,
)


NUMERIC_IMPUTATION_DESCRIPTIONS = {
    "mean": (
        "Липсващите числови стойности се заменят със "
        "средноаритметичната стойност от тренировъчните данни. "
        "Методът е чувствителен към крайни стойности."
    ),
    "median": (
        "Липсващите числови стойности се заменят с "
        "медианата от тренировъчните данни. "
        "Методът е по-устойчив на крайни стойности."
    ),
    "most_frequent": (
        "Липсващите числови стойности се заменят с "
        "най-често срещаната числова стойност "
        "в тренировъчните данни."
    ),
    "constant": (
        "Липсващите числови стойности се заменят "
        "с въведената от потребителя константа."
    ),
}


CATEGORICAL_IMPUTATION_DESCRIPTIONS = {
    "most_frequent": (
        "Липсващите категории се заменят с най-често "
        "срещаната категория в тренировъчните данни. "
        "При изцяло празна колона се използва "
        "въведената резервна константа, ако колоната "
        "не е премахната."
    ),
    "constant": (
        "Липсващите категориални стойности се заменят "
        "с въведената от потребителя текстова константа."
    ),
}


SCALING_DESCRIPTIONS = {
    "standard": (
        "Стандартизира числовите характеристики чрез "
        "средна стойност и стандартно отклонение. "
        "Подходящ метод за много алгоритми, включително "
        "логистична регресия."
    ),
    "minmax": (
        "Мащабира непостоянните тренировъчни характеристики "
        "в диапазона от 0 до 1. "
        "Тестовите стойности могат да излизат извън "
        "този диапазон."
    ),
    "robust": (
        "Мащабира чрез медиана и интерквартилен размах. "
        "По-устойчив на крайни стойности в сравнение "
        "със StandardScaler."
    ),
    "maxabs": (
        "Разделя стойностите на максималната абсолютна "
        "стойност от тренировъчните данни. "
        "Запазва знаците и нулевите стойности."
    ),
    "none": (
        "Числовите характеристики запазват своя мащаб. "
        "Подходящ вариант за алгоритми като Decision Tree "
        "и Random Forest, които обикновено не изискват "
        "мащабиране."
    ),
}


ENCODING_DESCRIPTIONS = {
    "onehot": (
        "Създава отделна бинарна характеристика за всяка "
        "научена категория. "
        "Не въвежда изкуствена числова подредба, но може "
        "значително да увеличи броя характеристики. "
        "Непознатите категории се представят чрез нули."
    ),
    "ordinal": (
        "Представя всяка научена категория чрез числов код. "
        "Запазва броя входни колони, но може да въведе "
        "изкуствена подредба между категориите. "
        "Непознатите категории получават код -1."
    ),
    "frequency": (
        "Представя категориите чрез относителната им честота "
        "в тренировъчните данни. "
        "Различни категории с еднаква честота могат да "
        "получат еднакъв код. "
        "Непознатите категории получават стойност 0."
    ),
}


class PreprocessingSettingsWidget(QWidget):
    """Компонент за конфигуриране на предварителната обработка.

    Позволява:
        - избор на стратегии за попълване;
        - въвеждане на константи;
        - избор на метод за мащабиране;
        - избор на метод за кодиране;
        - филтриране на характеристики;
        - валидиране на настройките;
        - възстановяване на настройките по подразбиране.

    Компонентът не обработва данни и не обучава модели.
    """

    configuration_changed = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._build_ui()
        self._connect_signals()

        self.reset_to_defaults()

    @staticmethod
    def _create_combo_box(
        options: dict[str, str],
        accessible_name: str,
    ) -> QComboBox:
        """Създава падащо меню с идентификатори и етикети."""

        combo_box = QComboBox()

        combo_box.setAccessibleName(
            accessible_name
        )

        for value, label in options.items():

            combo_box.addItem(
                label,
                value,
            )

        return combo_box

    @staticmethod
    def _create_description_label(
        text: str = "",
    ) -> QLabel:
        """Създава етикет за пояснителен текст."""

        label = QLabel(text)

        label.setWordWrap(True)

        label.setTextFormat(
            Qt.TextFormat.PlainText
        )

        return label

    @staticmethod
    def _create_form_layout(
        parent: QGroupBox,
    ) -> QFormLayout:
        """Създава формуляр с адаптивно разположение."""

        layout = QFormLayout(parent)

        layout.setSpacing(8)

        layout.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )

        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        return layout

    def _build_ui(self) -> None:
        """Изгражда графичния интерфейс."""

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        main_layout.setSpacing(12)

        introduction_label = (
            self._create_description_label(
                "Настройте начина, по който входните данни "
                "ще бъдат подготвени преди обучението на модела. "
                "Параметрите на трансформаторите се определят "
                "единствено от тренировъчните данни."
            )
        )

        main_layout.addWidget(
            introduction_label
        )

        main_layout.addWidget(
            self._build_imputation_group()
        )

        main_layout.addWidget(
            self._build_transformation_group()
        )

        main_layout.addWidget(
            self._build_feature_selection_group()
        )

        self.validation_label = (
            self._create_description_label()
        )

        self.validation_label.setVisible(False)

        main_layout.addWidget(
            self.validation_label
        )

        self.reset_button = QPushButton(
            "Възстанови настройките по подразбиране"
        )

        self.reset_button.setToolTip(
            "Възстановява първоначалната конфигурация "
            "за предварителна обработка."
        )

        main_layout.addWidget(
            self.reset_button
        )

        self._configuration_controls = (
            self.numeric_imputation_combo,
            self.numeric_fill_input,
            self.categorical_imputation_combo,
            self.categorical_fill_input,
            self.scaling_combo,
            self.encoding_combo,
            self.drop_empty_checkbox,
            self.drop_constant_checkbox,
        )

    def _build_imputation_group(self) -> QGroupBox:
        """Изгражда секцията за липсващи стойности."""

        group = QGroupBox(
            "Обработка на липсващи стойности"
        )

        layout = self._create_form_layout(
            group
        )

        # Числово попълване.
        self.numeric_imputation_combo = (
            self._create_combo_box(
                NUMERIC_IMPUTATION_STRATEGIES,
                "Стратегия за числово попълване",
            )
        )

        layout.addRow(
            "Числови характеристики:",
            self.numeric_imputation_combo,
        )

        self.numeric_imputation_description = (
            self._create_description_label()
        )

        layout.addRow(
            self.numeric_imputation_description
        )

        self.numeric_fill_input = QLineEdit()

        self.numeric_fill_input.setAccessibleName(
            "Числова константа"
        )

        self.numeric_fill_input.setPlaceholderText(
            "Например: 0, 10.5 или -1"
        )

        self.numeric_fill_input.setToolTip(
            "Приемат се крайни числови стойности, "
            "десетична точка, десетична запетая "
            "и научна нотация."
        )

        self.numeric_fill_label = QLabel(
            "Числова константа:"
        )

        self.numeric_fill_label.setBuddy(
            self.numeric_fill_input
        )

        layout.addRow(
            self.numeric_fill_label,
            self.numeric_fill_input,
        )

        # Категориално попълване.
        self.categorical_imputation_combo = (
            self._create_combo_box(
                CATEGORICAL_IMPUTATION_STRATEGIES,
                "Стратегия за категориално попълване",
            )
        )

        layout.addRow(
            "Категориални характеристики:",
            self.categorical_imputation_combo,
        )

        self.categorical_imputation_description = (
            self._create_description_label()
        )

        layout.addRow(
            self.categorical_imputation_description
        )

        self.categorical_fill_input = QLineEdit()

        self.categorical_fill_input.setAccessibleName(
            "Категориална константа"
        )

        self.categorical_fill_input.setPlaceholderText(
            "Например: Неизвестен"
        )

        self.categorical_fill_input.setToolTip(
            "Текстова стойност за попълване на липсващи "
            "категории или резервна стойност за "
            "изцяло празни колони."
        )

        layout.addRow(
            "Категориална константа:",
            self.categorical_fill_input,
        )

        return group

    def _build_transformation_group(self) -> QGroupBox:
        """Изгражда секцията за мащабиране и кодиране."""

        group = QGroupBox(
            "Трансформация на характеристики"
        )

        layout = self._create_form_layout(
            group
        )

        # Мащабиране.
        self.scaling_combo = (
            self._create_combo_box(
                SCALING_METHODS,
                "Метод за мащабиране",
            )
        )

        layout.addRow(
            "Мащабиране:",
            self.scaling_combo,
        )

        self.scaling_description = (
            self._create_description_label()
        )

        layout.addRow(
            self.scaling_description
        )

        scaling_note = (
            self._create_description_label(
                "Мащабирането се прилага само върху "
                "числовите и извлечените датни характеристики."
            )
        )

        layout.addRow(
            scaling_note
        )

        # Категориално кодиране.
        self.encoding_combo = (
            self._create_combo_box(
                ENCODING_METHODS,
                "Метод за категориално кодиране",
            )
        )

        layout.addRow(
            "Категориално кодиране:",
            self.encoding_combo,
        )

        self.encoding_description = (
            self._create_description_label()
        )

        layout.addRow(
            self.encoding_description
        )

        # Информация за датните характеристики.
        datetime_note = (
            self._create_description_label(
                "Датните колони се разпознават автоматично. "
                "От всяка датна колона се извличат девет "
                "числови характеристики, които също преминават "
                "през попълване и мащабиране."
            )
        )

        layout.addRow(
            datetime_note
        )

        return group

    def _build_feature_selection_group(self) -> QGroupBox:
        """Изгражда секцията за филтриране на характеристики."""

        group = QGroupBox(
            "Филтриране на характеристики"
        )

        layout = QVBoxLayout(group)

        layout.setSpacing(8)

        self.drop_empty_checkbox = QCheckBox(
            "Премахване на изцяло празни колони"
        )

        self.drop_empty_checkbox.setToolTip(
            "Премахва колоните без нито една попълнена "
            "стойност в тренировъчните данни."
        )

        layout.addWidget(
            self.drop_empty_checkbox
        )

        empty_description = (
            self._create_description_label(
                "Колоните, които са изцяло празни "
                "в тренировъчната извадка, няма да участват "
                "в последващата обработка."
            )
        )

        layout.addWidget(
            empty_description
        )

        self.drop_constant_checkbox = QCheckBox(
            "Премахване на постоянни характеристики"
        )

        self.drop_constant_checkbox.setToolTip(
            "Премахва трансформираните характеристики "
            "с нулева дисперсия в тренировъчните данни."
        )

        layout.addWidget(
            self.drop_constant_checkbox
        )

        constant_description = (
            self._create_description_label(
                "След трансформация се премахват "
                "характеристиките, които имат еднаква "
                "стойност във всички тренировъчни наблюдения. "
                "Те не предоставят информация за "
                "разграничаване на тренировъчните наблюдения."
            )
        )

        layout.addWidget(
            constant_description
        )

        warning_description = (
            self._create_description_label(
                "Ако след филтрирането не остане нито една "
                "входна характеристика, обучението "
                "няма да може да започне успешно."
            )
        )

        layout.addWidget(
            warning_description
        )

        return group

    def _connect_signals(self) -> None:
        """Свързва промените в настройките с обновяването."""

        self.numeric_imputation_combo.currentIndexChanged.connect(
            self._on_settings_changed
        )

        self.numeric_fill_input.textChanged.connect(
            self._on_settings_changed
        )

        self.categorical_imputation_combo.currentIndexChanged.connect(
            self._on_settings_changed
        )

        self.categorical_fill_input.textChanged.connect(
            self._on_settings_changed
        )

        self.scaling_combo.currentIndexChanged.connect(
            self._on_settings_changed
        )

        self.encoding_combo.currentIndexChanged.connect(
            self._on_settings_changed
        )

        self.drop_empty_checkbox.toggled.connect(
            self._on_settings_changed
        )

        self.drop_constant_checkbox.toggled.connect(
            self._on_settings_changed
        )

        self.reset_button.clicked.connect(
            self.reset_to_defaults
        )

    def _update_descriptions(self) -> None:
        """Обновява поясненията според избраните настройки."""

        numeric_method = (
            self.numeric_imputation_combo.currentData()
        )

        categorical_method = (
            self.categorical_imputation_combo.currentData()
        )

        scaling_method = (
            self.scaling_combo.currentData()
        )

        encoding_method = (
            self.encoding_combo.currentData()
        )

        self.numeric_imputation_description.setText(
            NUMERIC_IMPUTATION_DESCRIPTIONS.get(
                numeric_method,
                "",
            )
        )

        self.categorical_imputation_description.setText(
            CATEGORICAL_IMPUTATION_DESCRIPTIONS.get(
                categorical_method,
                "",
            )
        )

        self.scaling_description.setText(
            SCALING_DESCRIPTIONS.get(
                scaling_method,
                "",
            )
        )

        self.encoding_description.setText(
            ENCODING_DESCRIPTIONS.get(
                encoding_method,
                "",
            )
        )

        # Числовата константа се използва само при
        # стратегия constant.
        numeric_constant_enabled = (
            numeric_method == "constant"
        )

        self.numeric_fill_input.setEnabled(
            numeric_constant_enabled
        )

        self.numeric_fill_label.setEnabled(
            numeric_constant_enabled
        )

        # Категориалната константа остава достъпна,
        # защото се използва и като резервна стойност.
        self.categorical_fill_input.setEnabled(True)

    def _update_validation(self) -> None:
        """Проверява настройките и показва евентуална грешка."""

        try:

            self.get_configuration()

        except PreprocessingError as exc:

            self.validation_label.setText(
                f"Невалидна конфигурация: {exc}"
            )

            self.validation_label.setVisible(True)

        else:

            self.validation_label.clear()

            self.validation_label.setVisible(False)

    def _on_settings_changed(self, *_args) -> None:
        """Обработва промяна в конфигурацията."""

        self._update_descriptions()

        self._update_validation()

        self.configuration_changed.emit()

    def get_configuration(
        self,
    ) -> PreprocessingConfiguration:
        """Създава валидирана конфигурация от текущите настройки.

        При невалидна конфигурация повдига PreprocessingError.
        """

        numeric_strategy = (
            self.numeric_imputation_combo.currentData()
        )

        numeric_text = (
            self.numeric_fill_input.text()
        )

        numeric_fill_value = parse_numeric_value(
            numeric_text
        )

        if numeric_fill_value is None:

            if numeric_strategy == "constant":

                raise PreprocessingError(
                    "Въведете валидна крайна числова "
                    "константа за попълване."
                )

            # Невалидна, но неизползвана числова константа
            # се заменя със стойността по подразбиране.
            numeric_fill_value = (
                PreprocessingConfiguration()
                .numeric_fill_value
            )

        return PreprocessingConfiguration(
            numeric_imputation_strategy=numeric_strategy,
            categorical_imputation_strategy=(
                self.categorical_imputation_combo.currentData()
            ),
            numeric_fill_value=numeric_fill_value,
            categorical_fill_value=(
                self.categorical_fill_input.text()
            ),
            drop_empty_columns=(
                self.drop_empty_checkbox.isChecked()
            ),
            drop_constant_features=(
                self.drop_constant_checkbox.isChecked()
            ),
            scaling_method=(
                self.scaling_combo.currentData()
            ),
            encoding_method=(
                self.encoding_combo.currentData()
            ),
        )

    @property
    def preprocessing_configuration(
        self,
    ) -> PreprocessingConfiguration | None:
        """Връща конфигурацията или None при невалидни настройки."""

        try:

            return self.get_configuration()

        except PreprocessingError:

            return None

    @property
    def is_valid(self) -> bool:
        """Проверява дали текущите настройки са валидни."""

        return (
            self.preprocessing_configuration
            is not None
        )

    @staticmethod
    def _set_combo_value(
        combo_box: QComboBox,
        value: str,
    ) -> None:
        """Избира елемент чрез неговия вътрешен идентификатор."""

        index = combo_box.findData(value)

        if index < 0:

            raise PreprocessingError(
                f"Неподдържана настройка: „{value}“."
            )

        combo_box.setCurrentIndex(index)

    def set_configuration(
        self,
        configuration: PreprocessingConfiguration,
    ) -> None:
        """Зарежда съществуваща конфигурация в интерфейса."""

        if not isinstance(
            configuration,
            PreprocessingConfiguration,
        ):

            raise PreprocessingError(
                "Невалидна конфигурация "
                "за предварителна обработка."
            )

        # Запазваме първоначалното състояние на сигналите
        # и временно ги блокираме.
        previous_signal_states = [
            control.blockSignals(True)
            for control in self._configuration_controls
        ]

        try:

            self._set_combo_value(
                self.numeric_imputation_combo,
                configuration.numeric_imputation_strategy,
            )

            self.numeric_fill_input.setText(
                str(configuration.numeric_fill_value)
            )

            self._set_combo_value(
                self.categorical_imputation_combo,
                configuration.categorical_imputation_strategy,
            )

            self.categorical_fill_input.setText(
                configuration.categorical_fill_value
            )

            self._set_combo_value(
                self.scaling_combo,
                configuration.scaling_method,
            )

            self._set_combo_value(
                self.encoding_combo,
                configuration.encoding_method,
            )

            self.drop_empty_checkbox.setChecked(
                configuration.drop_empty_columns
            )

            self.drop_constant_checkbox.setChecked(
                configuration.drop_constant_features
            )

        finally:

            # Възстановяваме първоначалното състояние
            # на сигналите дори при възникване на грешка.
            for control, previous_state in zip(
                self._configuration_controls,
                previous_signal_states,
            ):

                control.blockSignals(
                    previous_state
                )

        # Изпращаме само един общ сигнал след
        # зареждането на цялата конфигурация.
        self._on_settings_changed()

    def reset_to_defaults(self) -> None:
        """Възстановява настройките по подразбиране."""

        self.set_configuration(
            PreprocessingConfiguration()
        )