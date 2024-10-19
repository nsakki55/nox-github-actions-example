import nox


@nox.session(venv_backend="uv", python=["3.12"], tags=["lint"])
def lint(session):
    session.install("ruff")
    session.run("uv", "run", "ruff", "format")


@nox.session(venv_backend="uv", python=["3.12"], tags=["lint"])
def mypy(session):
    session.install("pyproject.toml")
    session.install("mypy")
    session.run("uv", "run", "mypy", "src")


@nox.session(venv_backend="uv", python=["3.12", "3.11"], tags=["test"])
def test(session):
    session.run("uv", "sync", "--dev")

    if session.posargs:
        test_files = session.posargs
    else:
        test_files = ["tests"]

    session.run("uv", "run", "pytest", *test_files)
