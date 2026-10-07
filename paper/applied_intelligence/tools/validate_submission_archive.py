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
CHEMU_OK = {'results/chemu/summary.json', 'results/chemu/document_outcomes.csv'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def secret():
    env = ROOT / '.env'
    if not env.exists():
        return None
    for line in env.read_text(encoding='utf-8').splitlines():
        key, _, value = line.partition('=')
        if key.strip() == 'DEEPSEEK_API_KEY' and value.strip():
            return value.strip().strip('"').strip("'").encode()
    return None


def without_chemu_raw(report):
    report = json.loads(json.dumps(report))
    report['external_generation'].pop('chemu_api')
    return report


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
    assert not any(n.endswith(('.safetensors', '.pt', '.bin')) for n in names)
    assert not any(n.startswith('.env') or n.endswith('/.env') for n in names)
    assert len([n for n in names if n.startswith('results/controlled_repair/fold')]) == 20
    chemu = [n for n in names if 'chemu' in n.lower() and not n.endswith('.py')]
    assert set(chemu) == CHEMU_OK, sorted(set(chemu) - CHEMU_OK)
    key = secret()
    if key:
        for name in names:
            assert key not in (output / name).read_bytes(), 'API key found in archive'
    env = dict(os.environ, PYTHONUTF8='1', PYTHONHASHSEED='1')
    jobs = [
        ('main_cv', ['audit_submission.py'], ['audit.json', 'replayed_details.json']),
        ('original', ['audit_original.py'], ['original_replayed.json', 'original_first_metrics.json']),
        ('existing_decomposition', ['audit_strengthening.py', '--existing'], ['strengthening_existing.json']),
        ('controlled', ['audit_strengthening.py', '--controlled'], ['controlled_audit.json']),
        ('controller', ['audit_strengthening.py', '--controller-replay'], ['controlled_controller_replay.json']),
        ('extensions', ['audit_extensions.py'], ['extensions_audit.json']),
    ]

    def run(job):
        name, args, expected = job
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            r = subprocess.run([sys.executable, '-X', 'utf8', '-u',
                                str(output / 'paper/applied_intelligence/tools' / args[0]), *args[1:]],
                               cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((output / f'{name}.log').read_text(encoding='utf-8')[-7000:])
        for file in expected:
            mine = read(output / 'paper/applied_intelligence/evidence' / file)
            saved = read(P / 'evidence' / file)
            if file == 'extensions_audit.json':
                mine, saved = without_chemu_raw(mine), without_chemu_raw(saved)
            assert mine == saved, (name, file)
        print(f'Extracted archive replay passed: {name}', flush=True)
        return {'analysis': name, 'all_report_fields_equal': True, 'files': expected}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    with zipfile.ZipFile(P / 'Applied_Intelligence_submission_package.zip') as package:
        assert package.testzip() is None
        for name in ('manuscript.tex', 'manuscript.bbl', 'manuscript.pdf', 'cover_letter.txt',
                     'submission_notes_zh.txt', 'ESM_1.zip', 'source_files.zip'):
            assert package.read(name) == (P / name).read_bytes(), name
        for name in ('Fig1.pdf', 'Fig2.pdf', 'Fig1.eps', 'Fig2.eps'):
            assert package.read(name) == (P / 'figures' / name).read_bytes(), name
    with zipfile.ZipFile(P / 'source_files.zip') as source:
        assert source.testzip() is None
        assert source.read('manuscript.tex') == (P / 'manuscript.tex').read_bytes()
        assert {'manuscript.bbl', 'refs.bib', 'sn-jnl.cls', 'sn-basic.bst', 'Fig1.pdf', 'Fig2.pdf'} <= set(source.namelist())
    result = {'all_passed': True, 'extracted_to': str(output), 'checksummed_files': len(hashes),
              'analyses': results, 'package_matches_current_files': True,
              'human_selection_key_excluded': True, 'weights_excluded': True,
              'chemu_files_archived': sorted(CHEMU_OK), 'api_key_absent': bool(key)}
    (P / 'evidence/package_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
