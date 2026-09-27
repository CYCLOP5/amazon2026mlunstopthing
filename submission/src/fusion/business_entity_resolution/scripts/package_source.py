'create a portable source zip; never include data, credentials, caches or weights'
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    names = ['README.md', 'CONTEXT_HANDOFF.md', 'RUNS_HANDOFF.md', '.gitignore', '.amlignore', '.python-version',
             'pyproject.toml', 'uv.lock', 'requirements.txt', 'main.py']
    files = [ROOT/n for n in names]
    project = ROOT/'business_entity_resolution'
    files += [project/n for n in ('README.md','requirements.txt','.amlignore')]
    for folder in ('src','configs','azure','docs','scripts','tests','graph_resolution','innovation','final_hybrid','boosted_hybrid','large_reranker','final_repair','latest_fusion'):
        for path in (project/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.toml','.yml','.yaml','.md','.json'}:
                files.append(path)
    for path in (ROOT/'upstream_neural').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and '.venv' not in path.parts:
            if path.suffix in {'.py','.json','.md','.txt','.toml','.lock'} or path.name in {'.python-version','.gitignore','.amlignore'}:
                files.append(path)
    output = ROOT/'Amazites_updated_code.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path,Path('Amazites_updated_code')/path.relative_to(ROOT))
    print(f'{output}: {len(set(files))} files, {output.stat().st_size:,} bytes')
    print('SHA256',hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
