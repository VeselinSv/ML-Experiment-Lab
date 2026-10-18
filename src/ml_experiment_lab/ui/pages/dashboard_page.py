
"""Начално табло на приложението."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ml_experiment_lab.ui.widgets.dataset_overview_widget import (
    DatasetOverviewWidget,
)
from ml_experiment_lab.ui.widgets.quick_actions_widget import (
    QuickActionsWidget,
)
from ml_experiment_lab.ui.widgets.recent_experiments_widget import (
    RecentExperimentsWidget,
)
from ml_experiment_lab.ui.widgets.system_status_widget import (
    SystemStatusWidget,
)


class DashboardPage(QWidget):
    """Начално табло с обобщена информация за проекта."""

    def __init__(self) -> None:
        super().__init__()

        # Заглавие и описание.
        title = QLabel("ML Experiment Lab")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel(
            "Среда за работа с данни и експерименти по машинно обучение."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)

        # Съобщение за състоянието на последното действие.
        self.action_status = QLabel("Няма избрано действие.")
        self.action_status.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.action_status.setWordWrap(True)

        # Компонент за бързи действия.
        self.quick_actions = QuickActionsWidget()

        self.quick_actions.new_experiment_requested.connect(
            self.handle_new_experiment
        )

        self.quick_actions.upload_dataset_requested.connect(
            self.handle_upload_dataset
        )

        self.quick_actions.open_model_requested.connect(
            self.handle_open_model
        )

        # Обобщение на заредените набори от данни.
        self.dataset_overview = DatasetOverviewWidget()

        # Последни експерименти.
        self.recent_experiments = RecentExperimentsWidget()

        # Състояние на системата.
        self.system_status = SystemStatusWidget()

        # Основно разположение на началното табло.
        layout = QVBoxLayout(self)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.quick_actions)
        layout.addWidget(self.dataset_overview)
        layout.addWidget(self.recent_experiments)
        layout.addWidget(self.system_status)
        layout.addWidget(self.action_status)

    def handle_new_experiment(self) -> None:
        """Обработва заявка за създаване на нов експеримент."""
        self.action_status.setText(
            "Заявено е създаване на нов експеримент."
        )

    def handle_upload_dataset(self) -> None:
        """Обработва заявка за зареждане на набор от данни."""
        self.action_status.setText(
            "Заявено е зареждане на набор от данни."
        )

    def handle_open_model(self) -> None:
        """Обработва заявка за отваряне на модел."""
        self.action_status.setText(
            "Заявено е отваряне на модел."
        )
