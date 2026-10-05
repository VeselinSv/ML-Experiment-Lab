from ml_experiment_lab.config import APP_NAME, APP_VERSION


def test_project_metadata() -> None:
    assert APP_NAME == "ML Experiment Lab"
    assert APP_VERSION == "0.1.0"
