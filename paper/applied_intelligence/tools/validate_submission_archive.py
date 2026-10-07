"""Validate the final archive and replay every reported comparison after extraction."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / 'paper/applied_intelligence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    output = Path(tempfile.mkdtemp(prefix='submission_controlled_', dir=ROOT / 'tmp')).resolve()
    assert output.is_relative_to((ROOT / 'tmp').resolve())
    with zipfile.ZipFile(P / 'ESM_1.zip') as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert (output / name).resolve().is_relative_to(output)
        archive.extractall(output)
    hashes = read(output / 'SHA256SUMS.json')
    for name, expected in hashes.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, name
    names = set(hashes)
    assert 'paper/applied_intelligence/evidence/manual_audit_selection_key.json' not in names
    assert not any(n.endswith('.safetensors') for n in names)
    assert len([n for n in names if n.startswith('results/controlled_repair/fold')]) == 20
    jobs = [
        ('main_cv', ['audit_submission.py'], ['audit.json', 'replayed_details.json']),
        ('original', ['audit_original.py'], ['original_replayed.json', 'original_first_metrics.json']),
        ('existing_decomposition', ['audit_strengthening.py', '--existing'], ['strengthening_existing.json']),
        ('controlled', ['audit_strengthening.py', '--controlled'], ['controlled_audit.json']),
        ('controller', ['audit_strengthening.py', '--controller-replay'], ['controlled_controller_replay.json'])
    ]
    def run(job):
        name, args, expected = job
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            r = subprocess.run([sys.executable, '-X', 'utf8', '-u',
                                str(output / 'paper/applied_intelligence/tools' / args[0]), *args[1:]],
                               cwd=output, env=dict(os.environ, PYTHONUTF8='1'),
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((output / f'{name}.log').read_text(encoding='utf-8')[-7000:])
        for file in expected:
            assert read(output / 'paper/applied_intelligence/evidence' / file) == read(P / 'evidence' / file), (name, file)
        print(f'Extracted archive replay passed: {name}', flush=True)
        return {'analysis': name, 'all_report_fields_equal': True, 'files': expected}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    with zipfile.ZipFile(P / 'Applied_Intelligence_submission_package.zip') as package:
        assert package.testzip() is None
        for name in ('manuscript.tex', 'manuscript.pdf', 'cover_letter.txt', 'submission_notes_zh.txt',
                     'figures/Fig4.pdf', 'ESM_1.zip', 'source_files.zip'):
            assert package.read(name) == (P / name).read_bytes(), name
    with zipfile.ZipFile(P / 'source_files.zip') as source:
        assert source.testzip() is None
        assert source.read('manuscript.tex') == (P / 'manuscript.tex').read_bytes()
        assert {f'figures/Fig{i}.pdf' for i in range(1, 5)} <= set(source.namelist())
    result = {'all_passed': True, 'extracted_to': str(output), 'checksummed_files': len(hashes),
              'analyses': results, 'package_matches_current_files': True,
              'human_selection_key_excluded': True, 'weights_excluded': True}
    (P / 'evidence/package_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
