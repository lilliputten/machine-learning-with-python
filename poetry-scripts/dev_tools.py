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
    # env = os.environ.copy()
    # env['PYTHONWARNINGS'] = 'ignore::UserWarning:click'
    subprocess.run(cmd)  # , env=env)


def check_all():
    isort()
    format()
    lint()


# def test():
#     # NOTE: It doesn't work as poetry hasn't invoked it under the venv environment
#     print('Running unittest tests...')
#     cmd = [
#         'python',
#         '-m',
#         'unittest',
#         'discover',
#         '-v',
#         '-f',
#         '-t',
#         '.',
#         '-s',
#         '.',
#         '-p',
#         '*_test.py',
#     ]
#     subprocess.run(cmd)
