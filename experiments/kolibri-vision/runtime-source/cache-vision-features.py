"""Bounded pretrained features for targeted alignment; exact broker scope."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
import argparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a'


def control(action):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(90)
        sock.connect('/run/ollama-unify/gpu-negotiator.sock')
        sock.sendall((json.dumps({'action':action,'token':os.environ['OLLAMA_UNIFY_GPU_LEASE']})+'\n').encode())
        response=json.loads(sock.makefile().readline())
        if not response.get('ok'): raise RuntimeError('Broker rejected '+action)
        return response.get('result',response)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ready-file',type=Path,required=True)
    parser.add_argument('--min-tokens',type=int,default=1024)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--cache-name',default='feature-cache-1024')
    parser.add_argument('--data-manifest',type=Path)
    parser.add_argument('--max-cache-bytes',type=int,default=1024*2**20)
    parser.add_argument('--expected-gpus',default=GPU)
    args=parser.parse_args()
    allowed=[GPU,'GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0,'+GPU]
    if args.expected_gpus not in allowed or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.expected_gpus or not os.environ.get('OLLAMA_UNIFY_GPU_LEASE'):
        raise RuntimeError('Exact scoped reservation required')
    spec=importlib.util.spec_from_file_location('vf',ROOT/'runtime-source/vision-features.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not args.cache_name.replace('-','').isalnum() or not 1<=args.min_tokens<=args.max_tokens<=4096:
        raise ValueError('Bounded cache configuration required')
    extractor=module.VisionFeatures(ROOT,gpu=True,min_tokens=args.min_tokens,max_tokens=args.max_tokens)
    args.ready_file.write_text('pretrained vision tower resident\n')
    for _ in range(40):
        lease=control('heartbeat');lease=lease.get('lease',lease)
        if lease.get('state')=='active':break
        time.sleep(.25)
    else:raise RuntimeError('Lease readiness timeout')
    destination=ROOT/args.cache_name
    destination.mkdir(exist_ok=True)
    report={'source_projector_sha256':'abec1b06033105f9729390ce3a01a6bb1f347f0455c7dd8d2c804c5d485b0971',
        'min_image_tokens':args.min_tokens,'max_image_tokens':args.max_tokens,'feature_dtype':'float16',
        'precision_note':'Frozen native F32 features are stored in F16; projection training uses F32.',
        'examples':[],'byte_limit':args.max_cache_bytes,'bytes':0}
    pools=[(args.data_manifest.parent,args.data_manifest)]if args.data_manifest else[(ROOT/p,ROOT/p/'examples.jsonl')for p in ['data-pilot-counted','alignment-pilot']]
    for base,manifest in pools:
        if not manifest.resolve().is_relative_to(ROOT):raise RuntimeError('Data manifest must be run-local')
        lines=manifest.read_text().splitlines()
        if len(lines)>5000 or args.max_cache_bytes>32*2**30:raise RuntimeError('Bounded feature cohort required')
        for line in lines:
            example=json.loads(line)
            output=destination/(example['id'].split(':')[0]+'.npy')
            started=time.monotonic()
            if output.exists():
                features=np.load(output,mmap_mode='r')
            else:
                control('prepare')
                image_path=(base/example['image']).resolve()
                if not image_path.is_relative_to(ROOT):raise RuntimeError('Image must remain run-local')
                with Image.open(image_path) as image:features=extractor.encode(image)
                if features.shape[0]>args.max_tokens*1.2+64 or features.shape[1]!=4608 or not np.isfinite(features).all():
                    raise RuntimeError('Bounded feature contract failed')
                partial=output.with_suffix('.npy.partial')
                with partial.open('wb') as file:np.save(file,features.astype(np.float16))
                partial.replace(output)
                control('ready')
            report['bytes']+=output.stat().st_size
            if report['bytes']>report['byte_limit']:raise RuntimeError('Feature cache storage cap exceeded')
            entry={'id':example['id'],'pool':str(base.relative_to(ROOT)),'split':example['pilot_split'],
                'file':str(output.relative_to(ROOT)),'shape':list(features.shape),
                'seconds':time.monotonic()-started}
            report['examples'].append(entry)
            (destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(entry),flush=True)
    extractor.close()
    print(json.dumps({'completed':len(report['examples']),'bytes':report['bytes']}),flush=True)


if __name__=='__main__':main()
