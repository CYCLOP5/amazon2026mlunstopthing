"Inspect/recover surviving artifacts via Azure ML's authorized datastore API"
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from run_stack import TRAIN_RUNS, TEST_RUNS

SUBSCRIPTION = '80a7efc1-d2d5-4992-8ca0-3de6eae0978b'
WORKSPACE = f'/subscriptions/{SUBSCRIPTION}/resourceGroups/vnjhaveri-rg/providers/Microsoft.MachineLearningServices/workspaces/mlworkloads'
ACCOUNT = 'mlworkloads4325016703'
CONTAINER = 'azureml-blobstore-cf30d248-0165-4fbe-9167-f41a17bcfaf4'
DATA = 'LocalUpload/87666a896e8207989c312af14d2bbf86d6b80b6c532449d1ae8f178d617249bf/data/'
MODEL = 'LocalUpload/9e1084f60d9a6b9a4023bb0f69ab568281e2aebb7b2b88bedf9c7a54236751c9/model/'
GATE = 'amazon-ml-2026/shared/gate-1c335af91bfc0b51ca7f9c09af2eca345cdf562e05eb83cd638b01e7307806f4/'
REPORT = 'azureml/jolly_heart_lfccvw0y0k/out/experiments/stack_upgraded_lex/stack/'
CODE_CONTAINER = 'cf30d248-0165-4fbe-9167-f41a17bcfaf4-gb70kzj4c6lmzx7q1papxsvehx'
CODE = 'aml26-gpu-r61a8c15623-code/'


def credential():
    url = 'https://management.azure.com'+WORKSPACE+'/datastores/workspaceblobstore/listSecrets?api-version=2024-10-01'
    proc = subprocess.run(['az','rest','--method','post','--url',url,'--only-show-errors','-o','json'],
                          capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError('Azure ML refused datastore credential access. Check az login and workspace permissions.')
    response = json.loads(proc.stdout)
    secrets = response.get('secrets',response)
    value = secrets.get('key') or secrets.get('accountKey') or secrets.get('sasToken')
    if not value:
        raise RuntimeError('Datastore has no reusable key/SAS; use its configured identity access instead.')
    return value


def assets(groups):

    out = []
    if 'metadata' in groups:
        for blob,name in [(MODEL+'config.json','e5_config.json'), (MODEL+'neural_metadata.json','e5_metadata.json'),
            (DATA+'meta.json','prepared_meta.json'), (GATE+'metadata.json','gate_metadata.json'),
            (REPORT+'report.json','run3_report.json'), (REPORT+'analysis.json','run3_analysis.json')]:
            out.append((CONTAINER,blob,'metadata/'+name,True))
    if 'models' in groups:
        out.extend([(CONTAINER,MODEL,'models/finetuned-e5',False),(CONTAINER,GATE,'models/gate',False)])
    if 'data' in groups:
        out.append((CONTAINER,DATA,'data',False))
    if 'lexical' in groups:
        out.append((CONTAINER,'azureml/clever_neck_n4rvnz0z3m/work/blocking/','lexical-work/blocking',False))
    if 'scores' in groups:
        for split,runs in [('train',TRAIN_RUNS),('test',TEST_RUNS)]:
            for run in runs:
                rel = f'aml26-gpu-{run}/out/{split}/'
                out.append((CONTAINER,'amazon-ml-2026/'+rel,'scores/'+rel,False))
    if 'upstream' in groups:
        out.append((CODE_CONTAINER,CODE,'upstream',False))
    return out


def safe_destination(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Unsafe blob-relative path')
    dst = (root/Path(*path.parts)).resolve()
    if not dst.is_relative_to(root.resolve()):
        raise ValueError('Destination escapes recovery directory')
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',nargs='+',choices=['metadata','models','data','scores','lexical','upstream'],default=['metadata'])
    ap.add_argument('--destination',type=Path,default=Path('artifacts/recovered'))
    ap.add_argument('--download',action='store_true',help='Without this flag, only inspect blob names/sizes')
    ap.add_argument('--max-download-mb',type=float,default=0,help='Optional total-size cap; 0 is unlimited')
    ap.add_argument('--workers',type=int,default=4)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be positive')
    from azure.storage.blob import BlobServiceClient
    service = BlobServiceClient(f'https://{ACCOUNT}.blob.core.windows.net',credential=credential())
    jobs = {}
    for container,prefix,local,exact in assets(set(args.assets)):
        blobs = list(service.get_container_client(container).list_blobs(name_starts_with=prefix))
        if exact:
            blobs = [b for b in blobs if b.name == prefix]
        if not blobs:
            raise FileNotFoundError(f'No surviving blobs under {container}/{prefix}')
        print(f'{local}: {len(blobs):,} blobs, {sum(b.size for b in blobs)/2**20:.2f} MiB',flush=True)
        for blob in blobs:
            relative = local if exact else str(PurePosixPath(local)/blob.name[len(prefix):])
            safe_destination(args.destination,relative)
            jobs[(container,blob.name)] = {'container':container,'name':blob.name,'relative':relative,
                                         'bytes':blob.size,'etag':str(blob.etag)}
    total = sum(j['bytes'] for j in jobs.values())
    args.destination.mkdir(parents=True,exist_ok=True)
    (args.destination/'inventory.json').write_text(json.dumps(list(jobs.values()),indent=2))
    print(f'Total {len(jobs):,} blobs, {total/2**30:.3f} GiB. Inventory: {args.destination}/inventory.json',flush=True)
    if not args.download:
        return
    if args.max_download_mb and total > args.max_download_mb*2**20:
        raise ValueError('Selected artifacts exceed --max-download-mb; inspect inventory or increase the cap')

    def download(job):
        path = safe_destination(args.destination,job['relative'])
        tag = path.with_name(path.name+'.etag')
        if path.is_file() and path.stat().st_size == job['bytes'] and tag.exists() and tag.read_text() == job['etag']:
            return 'cached'
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_name(path.name+'.partial')
        client = service.get_blob_client(job['container'],job['name'])
        from azure.core import MatchConditions
        with temporary.open('wb') as stream:
            client.download_blob(max_concurrency=2,etag=job['etag'],match_condition=MatchConditions.IfNotModified).readinto(stream)
        if temporary.stat().st_size != job['bytes']:
            raise IOError(f'Incomplete download: {job["relative"]}')
        os.replace(temporary,path)
        tag.write_text(job['etag'])
        return 'downloaded'


    values = list(jobs.values())
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0,len(values),args.workers*2):
            for _ in pool.map(download,values[offset:offset+args.workers*2]):
                done += 1
                if done % 100 == 0 or done == len(values):
                    print(f'Recovered {done:,}/{len(values):,} files',flush=True)
    print('Recovery complete. Credentials were not persisted.',flush=True)


if __name__ == '__main__':
    main()
