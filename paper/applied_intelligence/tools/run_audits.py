"""Run every CPU audit in order with PYTHONHASHSEED=1; no model is loaded.

Set iteration order changes one unreported descriptive field in replayed_details.json
(precondition_exact_chain of record 5) by 0.3-0.5 points; hash seed 1 reproduces the
archived evidence byte for byte. The audit scripts themselves are frozen by recorded
hashes, so the seed is fixed here rather than inside them.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
AUDITS = [
    ['audit_submission.py'],
    ['audit_original.py'],
    ['audit_strengthening.py', '--existing'],
    ['audit_strengthening.py', '--controlled'],
    ['audit_strengthening.py', '--controller-replay'],
    ['audit_robustness_v2.py', '--omission'],
    ['audit_robustness_v2.py', '--analyze'],
    ['audit_extensions.py'],
]


def main():
    env = dict(os.environ, PYTHONHASHSEED='1', PYTHONUTF8='1')
    for script, *args in AUDITS:
        started = time.time()
        print('RUN', script, *args, flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', str(TOOLS / script), *args], cwd=ROOT, env=env, check=True)
        print(f'DONE {script} {" ".join(args)} {time.time() - started:.0f} s', flush=True)
    print('ALL_AUDITS_PASSED', flush=True)


if __name__ == '__main__':
    main()
