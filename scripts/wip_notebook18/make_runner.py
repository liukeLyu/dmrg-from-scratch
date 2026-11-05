"""Run scripts/make_results.py of the worktree as its own process, with the worktree package first on the path.

make_results.py must be the real __main__: on macOS the process pool starts its workers by
re-importing the main module, and they find the worktree package through PYTHONPATH.
"""

import os
import subprocess
import sys

WT = str(__import__("pathlib").Path(__file__).resolve().parents[2])  # the repo root
env = dict(os.environ, PYTHONPATH=WT + "/src", PYTHONUNBUFFERED="1")
sys.exit(subprocess.run([sys.executable, WT + "/scripts/make_results.py", *sys.argv[1:]], env=env, cwd=WT).returncode)
