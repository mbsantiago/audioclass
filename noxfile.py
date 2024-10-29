import nox


@nox.session(python=["3.9", "3.10", "3.11", "3.12"], venv_backend="uv")
def test_tflite_only(session):
    session.install(".[tflite]")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", "-m", "not tensorflow", "tests")


@nox.session(python=["3.9", "3.10", "3.11", "3.12"], venv_backend="uv")
def test_tensorflow_only(session):
    session.install(".[tensorflow]")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", "-m", "not tflite", "tests")


@nox.session(python=["3.9", "3.10", "3.11", "3.12"], venv_backend="uv")
def lint(session):
    session.install(".[all]")
    session.install("pyright")
    session.install("ruff")
    session.run("pyright", "src")
    session.run("ruff", "check", "src")
