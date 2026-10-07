import subprocess
import sys

from scripts.evaluate_supervisor import acquire_lock


def test_duplicate_supervisor_is_refused_and_closed_owner_releases_lock(tmp_path):
    program = '''
import sys
from pathlib import Path
from scripts.evaluate_supervisor import acquire_lock
try:
    stream = acquire_lock(Path(sys.argv[1]))
except RuntimeError:
    sys.exit(7)
stream.close()
'''
    owner = acquire_lock(tmp_path)
    try:
        duplicate = subprocess.run([sys.executable, '-c', program, str(tmp_path)])
        assert duplicate.returncode == 7
    finally:
        owner.close()
    successor = subprocess.run([sys.executable, '-c', program, str(tmp_path)])
    assert successor.returncode == 0
