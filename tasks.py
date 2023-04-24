import os

from invoke import task

GIT_HOOKS_LOC = ".hooks"
REQUIREMENTS_DIR = "requirements"
REQUIREMENTS_DEV_DIR = "requirements-dev"


@task
def install_reqs(c):
    c.run(
        f"pip install -r {REQUIREMENTS_DIR}/requirements.txt; "
        f"pip install -r {REQUIREMENTS_DEV_DIR}/requirements-dev.txt"
    )


@task
def pip_sync_reqs(c):
    c.run(
        f"pip-sync {REQUIREMENTS_DIR}/requirements.txt {REQUIREMENTS_DEV_DIR}/requirements-dev.txt"
    )


@task
def pip_compile_reqs(c):
    c.run(
        f"pip-compile --no-emit-index-url --output-file={REQUIREMENTS_DIR}/requirements.txt {REQUIREMENTS_DIR}/requirements.in; "
        f"pip-compile --no-emit-index-url --output-file={REQUIREMENTS_DEV_DIR}/requirements-dev.txt {REQUIREMENTS_DEV_DIR}/requirements-dev.in"
    )


def _generate_isort_cmd(check_only: bool = False) -> str:
    isort_cmd = "isort . "
    isort_ignore_filepath = ".isortignore"
    if os.path.exists(isort_ignore_filepath):
        with open(isort_ignore_filepath, "r") as f:
            isort_ignore_list = f.readlines()

        to_skip_params = " ".join(
            [f"--skip {elt.strip()}" for elt in isort_ignore_list]
        )
        isort_cmd += " " + to_skip_params

    if check_only:
        isort_cmd += "--check"

    return isort_cmd


@task
def format_code(c):
    c.run("black .")
    c.run(_generate_isort_cmd())


# Checks
@task
def check_format(c):
    print("Checking formatting with black")
    c.run("black --check .")
    print("Checking the imports with isort")
    c.run(_generate_isort_cmd(check_only=True))


@task
def check_types(c):
    print("Checking type hints with mypy")
    c.run("mypy .")


@task
def check_code(c):
    check_format(c)
    check_types(c)


# Build
@task
def generate_wheel(c, dist_dir="dist"):
    c.run(f"python -m build --wheel --outdir {dist_dir}")


# git hooks
@task
def install_hooks(c):
    for hook in os.listdir(GIT_HOOKS_LOC):
        print(hook)
        c.run(
            f"rm -f .git/hooks/{hook}; ln -s ../../{GIT_HOOKS_LOC}/{hook} .git/hooks/"
        )
    c.run("chmod a+x .hooks/*")
    c.run("git config --local core.hooksPath .git/hooks")
    print("hooks installed in .git/hooks")
