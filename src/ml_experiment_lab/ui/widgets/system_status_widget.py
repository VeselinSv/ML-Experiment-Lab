
"""Компонент за състоянието на системата."""

from PySide6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)


class SystemStatusWidget(QWidget):
    """Показва състоянието на компонентите на приложението."""

    def __init__(self) -> None:
        super().__init__()

        self.backend_status_value = QLabel("Не е конфигурирана")
        self.ml_engine_status_value = QLabel("Не е инициализирана")
        self.training_status_value = QLabel(
            "Все още няма информация"
        )

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Връзка с услугата:"))
        backend_row.addWidget(self.backend_status_value)

        ml_engine_row = QHBoxLayout()
        ml_engine_row.addWidget(QLabel("Локална среда за машинно обучение:"))
        ml_engine_row.addWidget(self.ml_engine_status_value)

        training_row = QHBoxLayout()
        training_row.addWidget(QLabel("Задачи за обучение:"))
        training_row.addWidget(self.training_status_value)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Състояние на системата"))
        layout.addLayout(backend_row)
        layout.addLayout(ml_engine_row)
        layout.addLayout(training_row)

    @staticmethod
    def _translate_status(status: str) -> str:
        """Превежда стандартните системни състояния."""
        translations = {
            "not configured": "Не е конфигурирана",
            "not configured yet": "Все още не е конфигурирана",
            "configured": "Конфигурирана",
            "connected": "Свързана",
            "disconnected": "Няма връзка",
            "available": "Налична",
            "unavailable": "Недостъпна",
            "not initialized": "Не е инициализирана",
            "initialized": "Инициализирана",
            "ready": "Готова",
            "not ready": "Не е готова",
            "not tracked yet": "Все още няма информация",
            "idle": "Неактивна",
            "running": "В процес на изпълнение",
            "busy": "Заета",
            "error": "Грешка",
            "failed": "Неуспешно",
        }

        return translations.get(status.strip().casefold(), status)

    def update_status(
        self,
        backend_status: str,
        ml_engine_status: str,
        training_status: str,
    ) -> None:
        """Обновява показаните състояния."""

        self.backend_status_value.setText(
            self._translate_status(backend_status)
        )
        self.ml_engine_status_value.setText(
            self._translate_status(ml_engine_status)
        )
        self.training_status_value.setText(
            self._translate_status(training_status)
        )
