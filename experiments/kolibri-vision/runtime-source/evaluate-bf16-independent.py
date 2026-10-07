"""BF16 precision control for the eight fresh stock-Ollama vision checks."""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TAG='kolibri-vision-aligned:bf16-20261006'
GPU='GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0,GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a'
spec=importlib.util.spec_from_file_location('evaluation',ROOT/'runtime-source/evaluate-stock-transfer.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
api=evaluation.stock.api


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--create-only',action='store_true');args=parser.parse_args()
    export=json.loads((ROOT/'Kolibri-1-vision-aligned-mmproj.export.json').read_text())
    digest=export['sha256']
    if args.create_only:
        license_text=(ROOT.parent/'kolibri-1-20261004/source/LICENSE').read_text()+'\n\n'+(ROOT/'VISION-DONOR-LICENSE.txt').read_text()
        result=api(11436,'/api/create',{'model':TAG,'stream':False,'requires':'0.35.0',
            'files':{'Kolibri-1-stock-BF16-release.gguf':'sha256:1e029937d835d4dd8c2a1ee775464585403b7492b7f5e987e19b231d679fc634',
                'Kolibri-1-vision-aligned-mmproj.gguf':'sha256:'+digest},
            'template':(ROOT.parent/'kolibri-1-20261004/kolibri-ollama-tools.gotmpl').read_text(),
            'license':license_text,'info':{'capabilities':['completion','vision','tools','thinking']},
            'parameters':{'num_ctx':8192,'num_predict':32,'temperature':0,'stop':['<|im_end|>']}})
        if result.get('status')!='success':raise RuntimeError('BF16 stock import failed')
        print(json.dumps({'created':TAG}));return
    discovery=json.loads(subprocess.run(['docker','gpu','discover'],capture_output=True,text=True,check=True).stdout)
    print(json.dumps({'warnings':discovery['warnings'],'active_leases':discovery['active_leases']}),flush=True)
    report={'tag':TAG,'precision':'BF16','projector_sha256':digest,'examples':[],'published':False,
        'scope':'eight fresh evaluation-only engineering checks; no best-in-class claim'}
    path=ROOT/'compatibility/aligned-bf16-independent-evaluation.json'
    for e in json.loads((ROOT/'independent-checks/manifest.json').read_text())['examples']:
        request={'model':TAG,'stream':False,'think':False,'keep_alive':'5m',
            'messages':[{'role':'user','content':e['question'],'images':[base64.b64encode((ROOT/e['image']).read_bytes()).decode()]}],
            'options':{'num_gpu':-1,'num_ctx':8192,'num_predict':16,'temperature':0}}
        response=api(11434,'/api/chat',request,headers={'X-Ollama-Unify-GPU-UUIDs':GPU,'X-Ollama-Unify-Workload-Class':'background'},timeout=900)
        answer=response['message'].get('content','')
        entry={'id':e['id'],'reference':e['answer'],'answer':answer,
            'normalized_exact':evaluation.normalize(answer)==evaluation.normalize(e['answer']),
            'anls':evaluation.anls(answer,e['answer']),'done':response.get('done'),'reason':response.get('done_reason')}
        report['examples'].append(entry);path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
    stdout=ROOT/'compatibility/bf16-vision-stock-cli.stdout';stderr=ROOT/'compatibility/bf16-vision-stock-cli.stderr'
    with stdout.open('w')as out,stderr.open('w')as err:
        command=['ollama','run',TAG,'--think=false','What shape is red? Return only the shape name. '+str(ROOT/'independent-checks/spatial-shape.png')]
        cli=subprocess.run(command,env={**os.environ,'OLLAMA_HOST':'http://127.0.0.1:11434'},stdout=out,stderr=err,timeout=900)
    report['stock_cli_returncode']=cli.returncode
    report['completed']=True;path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
