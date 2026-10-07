"""Resume and SHA256-verify downloads from ModelScope or a Hugging Face mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--source', choices=['modelscope', 'huggingface'], default='modelscope')
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    dest = ROOT / 'models' / args.repo.split('/')[-1]
    dest.mkdir(parents=True, exist_ok=True)
    if args.source == 'modelscope':
        rev = args.revision or 'master'
        api = f'https://modelscope.cn/api/v1/models/{args.repo}/repo/files?Revision={quote(rev)}&Recursive=true'
        with urlopen(api, timeout=60) as response:
            meta = json.load(response)
        if meta.get('Code') != 200:
            raise RuntimeError(meta)
        entries = [{"name": f['Path'], "size": f['Size'], "sha256": f['Sha256'],
                    "revision": f['Revision']} for f in meta['Data']['Files'] if f['Type'] == 'blob']
        base = f'https://modelscope.cn/models/{args.repo}/resolve'
    else:
        endpoint = args.endpoint.rstrip('/')
        rev = f'/revision/{quote(args.revision)}' if args.revision else ''
        with urlopen(f'{endpoint}/api/models/{args.repo}{rev}?blobs=true', timeout=60) as response:
            meta = json.load(response)
        entries = [{"name": f['rfilename'], "size": f['size'],
                    "sha256": f.get('lfs', {}).get('sha256'), "revision": meta['sha']}
                   for f in meta['siblings']]
        base = f'{endpoint}/{args.repo}/resolve'
    entries = [e for e in entries if e['name'].endswith(('.json', '.safetensors', '.txt'))
               or e['name'] in ('LICENSE', 'README.md')]
    (dest / f'{args.source}_download_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    def fetch(entry):
        path = dest / entry['name']
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + '.part')
        for candidate in (path, part):
            if candidate.exists() and candidate.stat().st_size == entry['size']:
                digest = sha256(candidate)
                if not entry['sha256'] or digest == entry['sha256']:
                    if candidate == part:
                        part.replace(path)
                    print(f"VERIFIED {entry['name']} {entry['size']}", flush=True)
                    return entry | {'actual_sha256': digest}
                candidate.rename(candidate.with_name(candidate.name + f'.invalid-{time.time_ns()}'))
        if part.exists() and part.stat().st_size > entry['size']:
            part.rename(part.with_name(part.name + f'.invalid-{time.time_ns()}'))
        url = f"{base}/{entry['revision']}/{quote(entry['name'])}"
        for attempt in range(8):
            offset = part.stat().st_size if part.exists() else 0
            try:
                request = Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                with urlopen(request, timeout=90) as response:
                    # A server that ignores Range must not corrupt an existing partial.
                    if offset and response.status != 206:
                        offset = 0
                    if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise IOError('Unexpected Content-Range')
                    start = time.monotonic()
                    last = start
                    written = 0
                    with part.open('ab' if offset else 'wb') as stream:
                        while block := response.read(4 * 1024 * 1024):
                            stream.write(block)
                            written += len(block)
                            now = time.monotonic()
                            if now - last >= 10:
                                print(f"DOWNLOAD {entry['name']} {offset + written}/{entry['size']} "
                                      f"{written / (now - start) / 2**20:.2f} MiB/s", flush=True)
                                last = now
                if part.stat().st_size != entry['size']:
                    raise IOError(f"Incomplete file: {part.stat().st_size}/{entry['size']}")
                digest = sha256(part)
                if entry['sha256'] and digest != entry['sha256']:
                    part.rename(part.with_name(part.name + f'.invalid-{time.time_ns()}'))
                    raise IOError('SHA256 mismatch; quarantined download')
                part.replace(path)
                print(f"VERIFIED {entry['name']} {entry['size']}", flush=True)
                return entry | {'actual_sha256': digest}
            except Exception as exc:
                print(f"RETRY {entry['name']} {attempt + 1}: {exc}", flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2 * (attempt + 1), 10))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        files = list(executor.map(fetch, entries))
    manifest = {'repo': args.repo, 'source': args.source, 'files': files, 'verified': True}
    (dest / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('DOWNLOAD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
