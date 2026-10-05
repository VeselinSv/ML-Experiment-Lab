"""Application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from ml_experiment_lab.ui.main_window import MainWindow
from ml_experiment_lab.config import APP_NAME


def main() -> int:
    """Create the Qt application and start the event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
