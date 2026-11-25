"""Графичен компонент за отчет на предварителната обработка."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.preprocessing import (
    CATEGORICAL_IMPUTATION_STRATEGIES,
    ENCODING_METHODS,
    NUMERIC_IMPUTATION_STRATEGIES,
    SCALING_METHODS,
)

from ml_experiment_lab.preprocessing_report import (
    COLUMN_TYPE_NAMES,
    ColumnPreprocessingReport,
    PreprocessingReport,
)


SCALING_PARAMETER_NAMES = {
    "mean": "Средноаритметична стойност",
    "variance": "Дисперсия",
    "scale": "Коефициент на мащабиране",
    "minimum": "Минимална стойност",
    "maximum": "Максимална стойност",
    "range": "Размах",
    "offset": "Отместване",
    "center": "Център (медиана)",
    "max_abs": "Максимална абсолютна стойност",
}


class PreprocessingReportWidget(QWidget):
    """Визуализира подробен отчет за предварителната обработка.

    Компонентът:
        - показва обобщение на експеримента;
        - визуализира качеството на входните данни;
        - показва научените параметри;
        - проследява генерираните характеристики;
        - показва премахнатите характеристики;
        - визуализира диагностични предупреждения.

    Компонентът не обработва данни, не обучава модели
    и не променя получения отчет.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._report: PreprocessingReport | None = None

        self._build_ui()

        self.clear_report()

    @property
    def report(self) -> PreprocessingReport | None:
        """Връща текущо заредения отчет."""

        return self._report

    @property
    def has_report(self) -> bool:
        """Проверява дали има зареден отчет."""

        return self._report is not None

    @staticmethod
    def _format_value(value: object) -> str:
        """Форматира стойност за визуализация."""

        if value is None:
            return "—"

        if isinstance(value, bool):
            return "Да" if value else "Не"

        if isinstance(value, float):
            return format(value, ".12g")

        return str(value)

    @staticmethod
    def _create_label(
        text: str = "",
    ) -> QLabel:
        """Създава етикет с автоматично пренасяне на текста."""

        label = QLabel(text)

        label.setWordWrap(True)

        label.setTextFormat(
            Qt.TextFormat.PlainText
        )

        return label

    @staticmethod
    def _create_table(
        headers: tuple[str, ...],
    ) -> QTableWidget:
        """Създава таблица само за четене."""

        table = QTableWidget(
            0,
            len(headers),
        )

        table.setHorizontalHeaderLabels(
            list(headers)
        )

        table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )

        table.setAlternatingRowColors(True)

        table.setSortingEnabled(False)

        table.setWordWrap(False)

        table.verticalHeader().setVisible(False)

        header = table.horizontalHeader()

        header.setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )

        header.setDefaultSectionSize(170)

        header.setMinimumSectionSize(80)

        header.setStretchLastSection(True)

        return table

    def _fill_table(
        self,
        table: QTableWidget,
        rows: Iterable[tuple[object, ...]],
    ) -> None:
        """Зарежда редове в таблица."""

        rows = tuple(rows)

        previous_updates_state = (
            table.updatesEnabled()
        )

        table.setUpdatesEnabled(False)

        try:

            table.clearContents()

            table.setRowCount(
                len(rows)
            )

            for row_index, row in enumerate(rows):

                if len(row) != table.columnCount():

                    raise ValueError(
                        "Невалиден брой стойности "
                        "в ред от таблицата."
                    )

                for column_index, value in enumerate(row):

                    formatted_value = (
                        self._format_value(value)
                    )

                    item = QTableWidgetItem(
                        formatted_value
                    )

                    item.setToolTip(
                        formatted_value
                    )

                    item.setFlags(
                        item.flags()
                        & ~Qt.ItemFlag.ItemIsEditable
                    )

                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ):

                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight
                            | Qt.AlignmentFlag.AlignVCenter
                        )

                    table.setItem(
                        row_index,
                        column_index,
                        item,
                    )

        finally:

            table.setUpdatesEnabled(
                previous_updates_state
            )

    def _build_ui(self) -> None:
        """Изгражда графичния интерфейс."""

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        main_layout.setSpacing(10)

        title = QLabel(
            "Отчет за предварителна обработка"
        )

        title.setObjectName("pageTitle")

        main_layout.addWidget(title)

        self._description_label = self._create_label(
            "Подробна информация за приложените "
            "трансформации и качеството на данните."
        )

        main_layout.addWidget(
            self._description_label
        )

        # Контейнер с две състояния:
        # 1. Няма зареден отчет.
        # 2. Има зареден отчет.
        self._stack = QStackedWidget()

        main_layout.addWidget(
            self._stack,
            1,
        )

        self._empty_page = (
            self._build_empty_page()
        )

        self._stack.addWidget(
            self._empty_page
        )

        self._tabs = QTabWidget()

        self._tabs.setMovable(False)

        self._tabs.setTabsClosable(False)

        self._tabs.setUsesScrollButtons(True)

        self._tabs.addTab(
            self._build_summary_tab(),
            "Обобщение",
        )

        self._tabs.addTab(
            self._build_columns_tab(),
            "Колони",
        )

        self._tabs.addTab(
            self._build_parameters_tab(),
            "Параметри",
        )

        self._tabs.addTab(
            self._build_features_tab(),
            "Характеристики",
        )

        self._tabs.addTab(
            self._build_warnings_tab(),
            "Предупреждения",
        )

        self._stack.addWidget(
            self._tabs
        )

        self._column_combo.currentIndexChanged.connect(
            self._on_column_changed
        )

    def _build_empty_page(self) -> QWidget:
        """Изгражда началното състояние без отчет."""

        page = QWidget()

        layout = QVBoxLayout(page)

        layout.addStretch(1)

        self._empty_label = self._create_label(
            "Все още няма отчет за предварителна обработка.\n\n"
            "Отчетът ще бъде достъпен след успешно "
            "обучение на ML модел."
        )

        self._empty_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self._empty_label
        )

        layout.addStretch(1)

        return page

    def _build_summary_tab(self) -> QWidget:
        """Изгражда раздела за обобщение."""

        page = QWidget()

        layout = QVBoxLayout(page)

        description = self._create_label(
            "Обща информация за наблюденията, "
            "размерността на данните и приложената "
            "конфигурация."
        )

        layout.addWidget(description)

        self._summary_table = self._create_table(
            (
                "Показател",
                "Стойност",
            )
        )

        self._summary_table.setColumnWidth(
            0,
            340,
        )

        layout.addWidget(
            self._summary_table,
            1,
        )

        return page

    def _build_columns_tab(self) -> QWidget:
        """Изгражда раздела за входните колони."""

        page = QWidget()

        layout = QVBoxLayout(page)

        description = self._create_label(
            "Типовете на колоните са определени от "
            "тренировъчните данни. Липсващите и "
            "невалидните стойности се отчитат отделно."
        )

        layout.addWidget(description)

        self._columns_table = self._create_table(
            (
                "Входна колона",
                "Тип",
                "Смесена",
                "Изцяло празна",
                "Липсващи (Training)",
                "Липсващи (Test)",
                "Невалидни (Training)",
                "Невалидни (Test)",
                "Непознати (Training)",
                "Непознати (Test)",
                "Генерирани",
                "Запазени",
                "Премахнати постоянни",
                "Премахната празна",
            )
        )

        self._columns_table.setColumnWidth(
            0,
            220,
        )

        layout.addWidget(
            self._columns_table,
            1,
        )

        note = self._create_label(
            "Непознатите категории се отчитат след "
            "попълването на липсващите стойности. "
            "Използвайте хоризонталното превъртане "
            "за достъп до всички показатели."
        )

        layout.addWidget(note)

        return page

    def _build_parameters_tab(self) -> QWidget:
        """Изгражда раздела за научените параметри."""

        page = QWidget()

        layout = QVBoxLayout(page)

        description = self._create_label(
            "Изберете входна колона, за да разгледате "
            "приложената обработка и научените параметри."
        )

        layout.addWidget(description)

        self._column_combo = QComboBox()

        self._column_combo.setAccessibleName(
            "Входна колона за анализ"
        )

        layout.addWidget(
            self._column_combo
        )

        self._column_details_label = (
            self._create_label()
        )

        layout.addWidget(
            self._column_details_label
        )

        self._parameters_table = self._create_table(
            (
                "Обработка",
                "Характеристика / категория",
                "Параметър",
                "Стойност",
            )
        )

        self._parameters_table.setColumnWidth(
            1,
            240,
        )

        self._parameters_table.setColumnWidth(
            2,
            230,
        )

        layout.addWidget(
            self._parameters_table,
            1,
        )

        return page

    def _build_features_tab(self) -> QWidget:
        """Изгражда раздела за генерираните характеристики."""

        page = QWidget()

        layout = QVBoxLayout(page)

        self._features_summary_label = (
            self._create_label()
        )

        layout.addWidget(
            self._features_summary_label
        )

        self._features_table = self._create_table(
            (
                "Оригинална колона",
                "Техническо име",
                "Четимо име",
                "Състояние",
                "Причина",
            )
        )

        self._features_table.setColumnWidth(
            2,
            250,
        )

        self._features_table.setColumnWidth(
            4,
            280,
        )

        layout.addWidget(
            self._features_table,
            1,
        )

        return page

    def _build_warnings_tab(self) -> QWidget:
        """Изгражда раздела за диагностични предупреждения."""

        page = QWidget()

        layout = QVBoxLayout(page)

        self._warnings_count_label = (
            self._create_label()
        )

        layout.addWidget(
            self._warnings_count_label
        )

        self._warnings_text = QPlainTextEdit()

        self._warnings_text.setReadOnly(True)

        self._warnings_text.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )

        layout.addWidget(
            self._warnings_text,
            1,
        )

        return page

    def set_report(
        self,
        report: PreprocessingReport,
    ) -> None:
        """Зарежда готов отчет в графичния компонент."""

        if not isinstance(
            report,
            PreprocessingReport,
        ):

            raise TypeError(
                "Очаква се валиден PreprocessingReport."
            )

        self._report = report

        self._populate_summary()

        self._populate_columns()

        self._populate_column_selector()

        self._populate_features()

        self._populate_warnings()

        # При зареждане на нов отчет започваме
        # от раздела за обобщение.
        self._tabs.setCurrentIndex(0)

        self._stack.setCurrentWidget(
            self._tabs
        )

    def clear_report(self) -> None:
        """Изчиства текущия отчет и всички резултати."""

        self._report = None

        previous_signal_state = (
            self._column_combo.blockSignals(True)
        )

        try:

            self._column_combo.clear()

            self._column_combo.setEnabled(False)

        finally:

            self._column_combo.blockSignals(
                previous_signal_state
            )

        tables = (
            self._summary_table,
            self._columns_table,
            self._parameters_table,
            self._features_table,
        )

        for table in tables:

            table.clearContents()

            table.setRowCount(0)

        self._column_details_label.clear()

        self._features_summary_label.clear()

        self._warnings_count_label.clear()

        self._warnings_text.clear()

        self._tabs.setCurrentIndex(0)

        self._stack.setCurrentWidget(
            self._empty_page
        )

    def _populate_summary(self) -> None:
        """Зарежда общите показатели на отчета."""

        report = self._report

        if report is None:
            return

        configuration = report.configuration

        rows = (
            (
                "Целева колона",
                report.target_column,
            ),
            (
                "Оригинални наблюдения",
                report.original_row_count,
            ),
            (
                "Изключени наблюдения",
                report.excluded_row_count,
            ),
            (
                "Валидни наблюдения",
                report.valid_row_count,
            ),
            (
                "Тренировъчни наблюдения",
                report.training_row_count,
            ),
            (
                "Тестови наблюдения",
                report.test_row_count,
            ),
            (
                "Заявен размер на тестовата извадка",
                f"{report.requested_test_size * 100:.2f}%",
            ),
            (
                "Действителен размер на тестовата извадка",
                f"{report.actual_test_size * 100:.2f}%",
            ),
            (
                "Random State",
                report.random_state,
            ),
            (
                "Оригинални входни колони",
                report.input_feature_count,
            ),
            (
                "Числови колони",
                report.numeric_column_count,
            ),
            (
                "Категориални колони",
                report.categorical_column_count,
            ),
            (
                "Датни колони",
                report.datetime_column_count,
            ),
            (
                "Изцяло празни колони",
                report.empty_column_count,
            ),
            (
                "Колони със смесени типове",
                report.mixed_column_count,
            ),
            (
                "Липсващи входни стойности",
                report.total_missing_count,
            ),
            (
                "Невалидни входни стойности",
                report.total_invalid_count,
            ),
            (
                "Непознати категории",
                report.total_unknown_count,
            ),
            (
                "Генерирани характеристики",
                report.generated_feature_count,
            ),
            (
                "Крайни характеристики",
                report.final_feature_count,
            ),
            (
                "Премахнати празни колони",
                report.removed_empty_count,
            ),
            (
                "Премахнати постоянни характеристики",
                report.removed_constant_count,
            ),
            (
                "Числово попълване",
                NUMERIC_IMPUTATION_STRATEGIES[
                    configuration.numeric_imputation_strategy
                ],
            ),
            (
                "Категориално попълване",
                CATEGORICAL_IMPUTATION_STRATEGIES[
                    configuration.categorical_imputation_strategy
                ],
            ),
            (
                "Настроена числова константа",
                configuration.numeric_fill_value,
            ),
            (
                "Настроена категориална константа",
                configuration.categorical_fill_value,
            ),
            (
                "Мащабиране",
                SCALING_METHODS[
                    configuration.scaling_method
                ],
            ),
            (
                "Категориално кодиране",
                ENCODING_METHODS[
                    configuration.encoding_method
                ],
            ),
            (
                "Премахване на празни колони",
                configuration.drop_empty_columns,
            ),
            (
                "Премахване на постоянни характеристики",
                configuration.drop_constant_features,
            ),
            (
                "Диагностични предупреждения",
                len(report.warnings),
            ),
        )

        self._fill_table(
            self._summary_table,
            rows,
        )

    def _populate_columns(self) -> None:
        """Зарежда информацията за входните колони."""

        report = self._report

        if report is None:
            return

        rows = []

        for column in report.columns:

            rows.append(
                (
                    column.column_name,
                    COLUMN_TYPE_NAMES[
                        column.column_type
                    ],
                    column.is_mixed,
                    column.is_empty,
                    column.training_missing_count,
                    column.test_missing_count,
                    column.training_invalid_count,
                    column.test_invalid_count,
                    column.training_unknown_count,
                    column.test_unknown_count,
                    column.generated_feature_count,
                    column.retained_feature_count,
                    column.removed_constant_count,
                    column.removed_as_empty,
                )
            )

        self._fill_table(
            self._columns_table,
            rows,
        )

    def _populate_column_selector(self) -> None:
        """Зарежда входните колони в падащото меню."""

        report = self._report

        if report is None:
            return

        previous_signal_state = (
            self._column_combo.blockSignals(True)
        )

        try:

            self._column_combo.clear()

            for column in report.columns:

                self._column_combo.addItem(
                    column.column_name,
                    column.column_name,
                )

            self._column_combo.setEnabled(
                bool(report.columns)
            )

            if report.columns:

                self._column_combo.setCurrentIndex(0)

        finally:

            self._column_combo.blockSignals(
                previous_signal_state
            )

        self._populate_selected_column()

    def _on_column_changed(
        self,
        _index: int,
    ) -> None:
        """Обновява параметрите при избор на колона."""

        self._populate_selected_column()

    def _populate_selected_column(self) -> None:
        """Зарежда научените параметри на избраната колона."""

        self._parameters_table.setRowCount(0)

        self._column_details_label.clear()

        report = self._report

        if report is None:
            return

        column_name = self._column_combo.currentData()

        if column_name is None:
            return

        column = report.get_column_report(
            column_name
        )

        details = [
            f"Колона: {column.column_name}",
            (
                "Тип: "
                + COLUMN_TYPE_NAMES[
                    column.column_type
                ]
            ),
            (
                "Генерирани характеристики: "
                f"{column.generated_feature_count}"
            ),
            (
                "Запазени характеристики: "
                f"{column.retained_feature_count}"
            ),
            (
                "Научени категории: "
                f"{column.learned_category_count}"
            ),
        ]

        if column.is_mixed:

            details.append(
                "Колоната съдържа смесени типове."
            )

        if column.removed_as_empty:

            details.append(
                "Колоната е премахната, защото е "
                "изцяло празна в тренировъчните данни."
            )

        self._column_details_label.setText(
            "\n".join(details)
        )

        if column.removed_as_empty:
            return

        rows = []

        rows.extend(
            self._get_imputation_rows(column)
        )

        rows.extend(
            self._get_scaling_rows(column)
        )

        rows.extend(
            self._get_encoding_rows(column)
        )

        self._fill_table(
            self._parameters_table,
            rows,
        )

    @staticmethod
    def _get_imputation_rows(
        column: ColumnPreprocessingReport,
    ) -> list[tuple[object, ...]]:
        """Подготвя редове за научените Imputation параметри."""

        rows = []

        strategy = column.imputation_strategy

        if strategy is None:
            return rows

        if column.column_type in (
            "numeric",
            "datetime",
        ):

            strategy_name = (
                NUMERIC_IMPUTATION_STRATEGIES[
                    strategy
                ]
            )

        else:

            strategy_name = (
                CATEGORICAL_IMPUTATION_STRATEGIES[
                    strategy
                ]
            )

        rows.append(
            (
                "Попълване",
                column.column_name,
                "Приложена стратегия",
                strategy_name,
            )
        )

        if column.categorical_imputation_value is not None:

            rows.append(
                (
                    "Попълване",
                    column.column_name,
                    "Стойност за попълване",
                    column.categorical_imputation_value,
                )
            )

        for feature in column.features:

            if feature.imputation_value is None:
                continue

            rows.append(
                (
                    "Попълване",
                    feature.display_name,
                    "Стойност за попълване",
                    feature.imputation_value,
                )
            )

        return rows

    @staticmethod
    def _get_scaling_rows(
        column: ColumnPreprocessingReport,
    ) -> list[tuple[object, ...]]:
        """Подготвя редове за научените Scaling параметри."""

        rows = []

        method = column.scaling_method

        if method is None:
            return rows

        rows.append(
            (
                "Мащабиране",
                column.column_name,
                "Приложен метод",
                SCALING_METHODS[method],
            )
        )

        for feature in column.features:

            for parameter_name, value in (
                feature.scaling_parameters
            ):

                display_parameter_name = (
                    SCALING_PARAMETER_NAMES.get(
                        parameter_name,
                        parameter_name,
                    )
                )

                rows.append(
                    (
                        "Мащабиране",
                        feature.display_name,
                        display_parameter_name,
                        value,
                    )
                )

        return rows

    @staticmethod
    def _get_encoding_rows(
        column: ColumnPreprocessingReport,
    ) -> list[tuple[object, ...]]:
        """Подготвя редове за научените категории."""

        rows = []

        method = column.encoding_method

        if method is None:
            return rows

        rows.append(
            (
                "Кодиране",
                column.column_name,
                "Приложен метод",
                ENCODING_METHODS[method],
            )
        )

        feature_display_names = {
            feature.technical_name: feature.display_name
            for feature in column.features
        }

        for category in column.categories:

            if method == "onehot":

                value = feature_display_names.get(
                    category.feature_name,
                    category.feature_name,
                )

                parameter_name = (
                    "Бинарна характеристика"
                )

            elif method == "ordinal":

                value = category.encoded_value

                parameter_name = "Числов код"

            elif method == "frequency":

                value = category.encoded_value

                parameter_name = (
                    "Относителна честота"
                )

            else:

                value = category.encoded_value

                parameter_name = "Кодирана стойност"

            rows.append(
                (
                    "Кодиране",
                    category.category,
                    parameter_name,
                    value,
                )
            )

        unknown_values = {
            "onehot": "Нули във всички бинарни характеристики",
            "ordinal": -1.0,
            "frequency": 0.0,
        }

        rows.append(
            (
                "Кодиране",
                "Непознати категории",
                "Резервна стойност",
                unknown_values.get(method),
            )
        )

        return rows

    def _populate_features(self) -> None:
        """Зарежда генерираните и премахнатите характеристики."""

        report = self._report

        if report is None:
            return

        summary = [
            (
                "Оригинални входни колони: "
                f"{report.input_feature_count}"
            ),
            (
                "Генерирани характеристики: "
                f"{report.generated_feature_count}"
            ),
            (
                "Крайни характеристики: "
                f"{report.final_feature_count}"
            ),
            (
                "Премахнати празни колони: "
                f"{report.removed_empty_count}"
            ),
            (
                "Премахнати постоянни характеристики: "
                f"{report.removed_constant_count}"
            ),
        ]

        self._features_summary_label.setText(
            "\n".join(summary)
        )

        rows = []

        # Запазваме действителния ред на генерираните
        # характеристики от обучения трансформатор.
        for feature in report.generated_features:

            if feature.removed_as_constant:

                status = "Премахната"

                reason = (
                    "Постоянна характеристика в Training"
                )

            else:

                status = "Запазена"

                reason = None

            rows.append(
                (
                    feature.source_column,
                    feature.technical_name,
                    feature.display_name,
                    status,
                    reason,
                )
            )

        # Празните колони са премахнати преди
        # генерирането на трансформирани характеристики.
        for column_name in report.removed_empty_columns:

            rows.append(
                (
                    column_name,
                    None,
                    column_name,
                    "Премахната",
                    "Изцяло празна входна колона в Training",
                )
            )

        self._fill_table(
            self._features_table,
            rows,
        )

    def _populate_warnings(self) -> None:
        """Зарежда диагностичните предупреждения."""

        report = self._report

        if report is None:
            return

        warning_count = len(report.warnings)

        self._warnings_count_label.setText(
            f"Брой предупреждения: {warning_count}"
        )

        if warning_count == 0:

            self._warnings_text.setPlainText(
                "Няма диагностични предупреждения "
                "за предварителната обработка."
            )

            return

        warning_lines = []

        for index, warning in enumerate(
            report.warnings,
            start=1,
        ):

            warning_lines.append(
                f"{index}. {warning}"
            )

        self._warnings_text.setPlainText(
            "\n\n".join(warning_lines)
        )