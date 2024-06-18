from pathlib import Path
import os

import nox


def ensurepath():
    rye_home = os.getenv("RYE_HOME")
    rye_py = Path(rye_home) / "py" if rye_home else Path.home() / ".rye" / "py"

    for py_dir in rye_py.iterdir():
        bin_dir = py_dir / "bin"
        print(bin_dir)
        os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"


ensurepath()


@nox.session(python=["3.9", "3.10", "3.11"], venv_backend="uv")
def test_tflite_only(session):
    session.install(".[tflite]")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", "-m", "tflite")


@nox.session(python=["3.9", "3.10", "3.11"], venv_backend="uv")
def test_tensorflow_only(session):
    session.install(".[tensorflow]")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", "-m", "tensorflow")

@nox.session(python=["3.9", "3.10", "3.11"], venv_backend="uv")
def lint(session):
    session.install(".[all]")
    session.install("pyright")
    session.install("ruff")
    session.run("pyright", "src")
    session.run("ruff", "check", "src")
