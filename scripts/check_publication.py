"""Audit the code/documents-only release without loading data or model weights."""
import ast
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.svg'}
SPECIAL_NAMES = {'LICENSE', '.gitignore', '.gitattributes'}


def main():
    # A Git checkout audits every tracked file, including ignored-but-tracked files.
    # An exported source directory audits all files except local Python caches.
    if (ROOT/'.git').exists():
        names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
        files = [ROOT/n for n in names if n]
    else:
        files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    errors = []
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or (path.suffix not in TEXT_SUFFIXES and path.name not in SPECIAL_NAMES):
            errors.append(f'Not a permitted public text/vector file: {name}'); continue
        if path.stat().st_size > 1_000_000:
            errors.append(f'Unexpectedly large source/document: {name}')
        text = path.read_text(encoding='utf-8-sig')
        # Report file names only, never the potentially sensitive match.
        patterns = [r'[A-Z]:[\\/]+Users[\\/]', r'github_pat_[A-Za-z0-9_]{20,}',
                    r'gh[pousr]_[A-Za-z0-9]{20,}', r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----']
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            errors.append(f'Personal path or credential-like text: {name}')
        if path.suffix == '.py':
            ast.parse(text, filename=name)
        if path.suffix == '.json':
            json.loads(text)
        if path.suffix == '.svg':
            tree = ET.fromstring(text)
            for node in tree.iter():
                if node.tag.split('}')[-1] in ('image', 'script', 'foreignObject'):
                    errors.append(f'Embedded image/script in diagram: {name}')
        if path.suffix == '.md' and name != 'src/README.md':
            # src/README is the immutable historical package guide; models are omitted.
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if target.startswith(('https://', 'http://', '#', 'mailto:')):
                    continue
                target = target.split('#')[0]
                if target and not (path.parent/target).exists():
                    errors.append(f'Broken document link in {name}: {target}')
    manifest = json.loads((ROOT/'src/manifest.json').read_text(encoding='utf-8'))
    preserved = 0
    for name, expected in manifest['files'].items():
        if name.startswith('models/'):
            continue
        actual = hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f'Frozen inference source changed: {name}')
        preserved += 1
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS {len(files)} public files; {preserved} frozen source hashes; syntax, links, scope.')


if __name__ == '__main__':
    main()
