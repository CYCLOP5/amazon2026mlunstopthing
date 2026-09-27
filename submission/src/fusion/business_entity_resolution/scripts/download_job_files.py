'download only explicitly named azure job output files; credentials stay in memory'
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import time

from azure_artifacts import ACCOUNT, CONTAINER, credential, safe_destination


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or str(path) == '.':
        raise ValueError('Expected a safe, nonempty relative output path')
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output-name', default='out')
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--max-download-mb', type=float, default=0, help='Total selected size cap; zero is unlimited')
    args = parser.parse_args()
    if args.max_download_mb < 0:
        parser.error('--max-download-mb must be nonnegative')
    job = relative_path(args.job)
    output = relative_path(args.output_name)
    if '/' in job or '/' in output:
        parser.error('--job and --output-name must each be one path component')
    files = list(dict.fromkeys(relative_path(value) for value in args.files))
    from azure.core import MatchConditions
    from azure.storage.blob import BlobServiceClient
    service = BlobServiceClient(f'https://{ACCOUNT}.blob.core.windows.net', credential=credential())
    inventory = []
    for relative in files:
        safe_destination(args.destination, relative)
        client = service.get_blob_client(CONTAINER, f'azureml/{job}/{output}/{relative}')
        try:
            props = client.get_blob_properties()
        except Exception as exc:
            raise RuntimeError(f'Cannot inspect selected output {relative} ({type(exc).__name__}); service details omitted') from None
        inventory.append({'relative': relative, 'bytes': props.size, 'etag': str(props.etag)})
        print(f'{relative}: {props.size:,} bytes', flush=True)
    total = sum(item['bytes'] for item in inventory)
    print(f'Selected {len(inventory)} files, {total:,} bytes', flush=True)
    if args.max_download_mb and total > args.max_download_mb * 2**20:
        raise ValueError('Selected files exceed --max-download-mb')
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination/'download_inventory.json').write_text(json.dumps({'job': job, 'output_name': output, 'files': inventory}, indent=2)+'\n')
    if args.inventory_only:
        return
    for item in inventory:
        relative = item['relative']
        dst = safe_destination(args.destination, relative)
        tag = dst.with_name(dst.name+'.etag')
        if dst.is_file() and dst.stat().st_size == item['bytes'] and tag.is_file() and tag.read_text() == item['etag']:
            print(f'{relative}: cached', flush=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        partial = dst.with_name(dst.name+'.partial')
        client = service.get_blob_client(CONTAINER, f'azureml/{job}/{output}/{relative}')
        last_progress = time.monotonic()
        def progress(current, expected):
            nonlocal last_progress
            now = time.monotonic()
            if now-last_progress >= 10 or current == expected:
                print(f'{relative}: {current:,}/{expected:,} bytes', flush=True)
                last_progress = now
        try:
            with partial.open('wb') as stream:
                client.download_blob(etag=item['etag'], match_condition=MatchConditions.IfNotModified,
                                     max_concurrency=2, progress_hook=progress).readinto(stream)
                stream.flush()
                os.fsync(stream.fileno())
            if partial.stat().st_size != item['bytes']:
                raise IOError('Downloaded size differs from inventory')
            current = client.get_blob_properties()
            if str(current.etag) != item['etag'] or current.size != item['bytes']:
                raise IOError('Output changed during download')
            os.replace(partial, dst)
            tag_partial = tag.with_name(tag.name+'.partial')
            tag_partial.write_text(item['etag'])
            os.replace(tag_partial, tag)
        except Exception as exc:
            raise RuntimeError(f'Download failed for {relative} ({type(exc).__name__}); partial retained, service details omitted') from None
        print(f'{relative}: verified', flush=True)
    print('Selected downloads complete.', flush=True)


if __name__ == '__main__':
    main()
