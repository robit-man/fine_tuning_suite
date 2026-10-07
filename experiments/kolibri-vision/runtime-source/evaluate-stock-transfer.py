"""Stock image evaluation of an experimental pretrained-transfer candidate.

The cohort is small development-only evidence, not a benchmark or publication
gate. All answers and references are retained, with no capability inference from
successful image IO. No publishing code exists here.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import re

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT.parent/'kolibri-1-20261004'
GPU='GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a'
MAIN='fa6ccc6467702ea435876ed0e13f135ec268be8d7818ba66201de44eadb8181f'
DONOR='robit/ornith-1.5:9b'
spec=importlib.util.spec_from_file_location('stock_io',ROOT/'runtime-source/verify-full-stock-pilot.py')
stock=importlib.util.module_from_spec(spec);spec.loader.exec_module(stock)


def normalize(text):
    return ' '.join(re.sub(r'[^\w\s.]',' ',text.lower()).split()).strip('.')


def anls(prediction, reference):
    left,right=normalize(prediction),normalize(reference)
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(a!=b)))
        previous=current
    distance=previous[-1]/max(len(left),len(right),1)
    return 1-distance if distance<.5 else 0.0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--projector',type=Path,required=True)
    parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--skip-donor',action='store_true')
    parser.add_argument('--cohort',choices=['development','independent','natural','independent-all'],default='development')
    args=parser.parse_args()
    if not args.tag.startswith('kolibri-vision-') or not args.run_id.replace('-','').isalnum():
        raise ValueError('Only explicit experimental local tags')
    proof=json.loads(args.proof.read_text())
    if not proof['passed']:raise RuntimeError('Native export verification failed')
    discovery=json.loads(subprocess.run(['docker','gpu','discover'],capture_output=True,text=True,check=True).stdout)
    print(json.dumps({'warnings':discovery['warnings'],'active_leases':discovery['active_leases']}),flush=True)
    if any(GPU in lease['gpu_uuids'] for lease in discovery['active_leases']):
        raise RuntimeError('Image evaluation scope still has external leases')
    with args.projector.open('rb') as file:digest=hashlib.file_digest(file,'sha256').hexdigest()
    if proof.get('projector_sha256')!=digest:raise RuntimeError('Native verification is not bound to this projector')
    export=json.loads(args.projector.with_suffix('.export.json').read_text())
    if digest!=export['sha256']:raise RuntimeError('Candidate digest changed')
    stock.import_projector(args.projector,digest)
    created=stock.api(11436,'/api/create',{'model':args.tag,'stream':False,'requires':'0.35.0',
        'files':{'Kolibri-1-stock-Q4_K_M.gguf':'sha256:'+MAIN,args.projector.name:'sha256:'+digest},
        'template':(RELEASE/'kolibri-ollama-tools.gotmpl').read_text(),
        'info':{'capabilities':['completion','vision','tools','thinking']},
        'parameters':{'num_ctx':8192,'num_predict':128,'temperature':0,'stop':['<|im_end|>']}})
    if created.get('status')!='success':raise RuntimeError('Stock create failed')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(30);sock.connect('/run/ollama-unify/gpu-negotiator.sock')
        sock.sendall((json.dumps({'action':'set_model_gpus','model':args.tag,'gpu_uuids':[GPU]})+'\n').encode())
        if not json.loads(sock.makefile().readline()).get('ok'):raise RuntimeError('Placement failed')
    report={'utc':datetime.now(timezone.utc).isoformat(),'tag':args.tag,'projector_sha256':digest,
        'language_sha256':MAIN,'optimizer_updates':proof['optimizer_updates'],
        'scope':'small '+args.cohort+' cohort only; not a published benchmark',
        'published':False,'examples':[]}
    path=ROOT/'compatibility'/(args.run_id+'-stock-evaluation.json')
    examples=[]
    if args.cohort=='development':
        for pool in ['data-pilot-counted','alignment-pilot']:
            examples.extend((pool,json.loads(line)) for line in (ROOT/pool/'examples.jsonl').read_text().splitlines()
                if json.loads(line)['pilot_split']=='development_holdout')
    if args.cohort in ['independent','independent-all']:
        for entry in json.loads((ROOT/'independent-checks/manifest.json').read_text())['examples']:
            examples.append(('independent-checks',{'id':entry['id'],'config':entry['task'],
                'user':entry['question'],'assistant':entry['answer'],
                'image':str(Path(entry['image']).relative_to('independent-checks')),'expected_short_answer':True}))
    if args.cohort in ['natural','independent-all']:
        for line in (ROOT/'independent-natural-pool/evaluation-only-examples.jsonl').read_text().splitlines():
            entry=json.loads(line)
            if not entry.get('training_excluded'):raise RuntimeError('Natural cohort image is not fenced')
            examples.append(('independent-natural-pool',entry))
    png_directory=ROOT/'compatibility'/'stock-quality-images';png_directory.mkdir(exist_ok=True)
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('projector_sha256')!=digest:raise RuntimeError('Resume projector changed')
        report=previous
    entries={e['id']:e for e in report['examples']}
    for pool,example in examples:
        if example['id'] not in entries:
            entry={'id':example['id'],'config':example['config'],'question':example['user'],
                'reference':example['assistant'],'answers':{}}
            report['examples'].append(entry);entries[example['id']]=entry
    models=[args.tag] if args.skip_donor else [DONOR,args.tag]
    # Batch each model to avoid repeated large decoder reloads on the same GPU.
    for tag in models:
        for pool,example in examples:
            entry=entries[example['id']]
            if tag in entry['answers']:continue
            png=png_directory/(example['id'].split(':')[0]+'.png')
            if not png.exists():
                with Image.open(ROOT/pool/example['image']) as image:image.convert('RGB').save(png)
            if png.stat().st_size>12*2**20:raise RuntimeError('Image payload cap exceeded')
            response=stock.api(11434,'/api/chat',{'model':tag,'think':False,'stream':False,'keep_alive':'5m',
                'messages':[{'role':'user','content':example['user'],
                    'images':[base64.b64encode(png.read_bytes()).decode()]}],
                'options':{'num_gpu':-1,'num_ctx':8192,'num_predict':128,'temperature':0}},
                headers={'X-Ollama-Unify-GPU-UUIDs':GPU,'X-Ollama-Unify-Workload-Class':'background'},timeout=600)
            answer=response['message'].get('content','')
            result={'content':answer,'done':response.get('done'),'reason':response.get('done_reason'),
                'prompt_tokens':response.get('prompt_eval_count'),'output_tokens':response.get('eval_count')}
            if example['config'] in ['docvqa','textvqa','screenqa','chartqa'] or example.get('expected_short_answer'):
                result['anls']=anls(answer,example['assistant'])
                result['normalized_exact']=normalize(answer)==normalize(example['assistant'])
            entry['answers'][tag]=result
            print(json.dumps({'id':entry['id'],'model':tag,'reference':entry['reference'],**result}),flush=True)
            path.write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
