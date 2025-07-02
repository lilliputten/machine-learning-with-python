import subprocess


def isort():
    print("Running imports sorter...")
    cmd = [
        "isort",
        "--only-modified",
        ".",
    ]
    subprocess.run(cmd)


def lint():
    print("Running pyright linter...")
    cmd = [
        "pyright",
        ".",
    ]
    subprocess.run(cmd)


def format():
    print("Running python linter (blue)...")
    cmd = [
        "black",
        ".",
    ]
    subprocess.run(cmd)


def check_all():
    isort()
    format()
    lint()
