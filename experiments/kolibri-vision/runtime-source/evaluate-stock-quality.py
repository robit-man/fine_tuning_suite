"""Compare a verified experimental vision GGUF with its donor in stock Ollama.

Image-grounded engineering checks and image-fenced natural/official subsets
retain every answer. Short-answer scoring is descriptive subset evidence, never
the full official benchmark or a best-in-class result. No publication code.
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

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent/'kolibri-1-20261004'
GPU = 'GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a'
MAINS = {'bf16': '1e029937d835d4dd8c2a1ee775464585403b7492b7f5e987e19b231d679fc634',
         'q4_k_m': 'fa6ccc6467702ea435876ed0e13f135ec268be8d7818ba66201de44eadb8181f'}
DONOR = 'robit/ornith-1.5:9b'
spec = importlib.util.spec_from_file_location('previous_evaluation', ROOT/'runtime-source/evaluate-stock-transfer.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
stock = previous.stock


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def answer_for_scoring(answer, config):
    # This official task requests answer+bbox output while its short answer
    # field holds only the answer. Score that field, retain the full output;
    # bbox accuracy is explicitly not claimed by this subset comparison.
    if config=='VQA with position en':
        stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip())
        try:
            value = json.loads(stripped)
            if isinstance(value,dict) and isinstance(value.get('answer'),str):
                return value['answer']
        except ValueError:
            pass
    return answer


def cohort(name):
    examples = []
    if name in ['engineering', 'all']:
        for e in json.loads((ROOT/'independent-checks/manifest.json').read_text())['examples']:
            examples.append({'id': e['id'], 'config': e['task'], 'question': e['question'],
                'references': [e['answer']], 'image': e['image'], 'score_short_answer': True,
                'cohort': 'engineering'})
    if name in ['counterfactual', 'all']:
        path = ROOT/'counterfactual-checks/manifest.json'
        if path.exists():
            for e in json.loads(path.read_text())['examples']:
                examples.append({'id': e['id'], 'config': e['task'], 'question': e['question'],
                    'references': [e['answer']], 'image': e['image'], 'score_short_answer': True,
                    'cohort': 'counterfactual'})
    if name in ['natural', 'all']:
        pool = ROOT/'independent-natural-pool'
        for line in (pool/'evaluation-only-examples.jsonl').read_text().splitlines():
            e = json.loads(line)
            if not e.get('training_excluded'):
                raise RuntimeError('Unfenced natural image')
            examples.append({'id': e['id'], 'config': e['config'], 'question': e['user'],
                'references': [e['assistant']], 'image': str((pool/e['image']).relative_to(ROOT)),
                'score_short_answer': e['config'] in ['docvqa','textvqa','screenqa','chartqa'],
                'cohort': 'natural'})
    if name in ['ocr', 'all']:
        pool = ROOT/'ocrbench-text-evaluation'
        for line in (pool/'examples.jsonl').read_text().splitlines():
            e = json.loads(line)
            if not e.get('training_excluded'):
                raise RuntimeError('Unfenced benchmark image')
            examples.append({'id': e['id'], 'config': e['task_type'], 'question': e['question'],
                'references': e['answers'], 'image': str((pool/e['image']).relative_to(ROOT)),
                'score_short_answer': True, 'cohort': 'ocrbench-text-subset',
                'source_revision': e['revision']})
    if not examples or len(examples)>180:
        raise RuntimeError('Empty or oversized evaluation cohort')
    for e in examples:
        path = (ROOT/e['image']).resolve()
        if not path.is_relative_to(ROOT) or path.stat().st_size>12*2**20:
            raise RuntimeError('Evaluation media is outside the bounded run directory')
        e['media_sha256'] = digest(path)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projector', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--precision', choices=list(MAINS), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--cohort', choices=['engineering','counterfactual','natural','ocr','all'], default='engineering')
    parser.add_argument('--create-only', action='store_true')
    parser.add_argument('--skip-donor', action='store_true')
    parser.add_argument('--num-predict', type=int, default=128)
    parser.add_argument('--gpu', default=GPU)
    args = parser.parse_args()
    selected_gpu=args.gpu
    if not selected_gpu.startswith('GPU-'):
        raise ValueError('An exact stable GPU UUID is required')
    if not args.tag.startswith('kolibri-vision-') or not args.run_id.replace('-','').isalnum():
        raise ValueError('Only our explicit experimental local tags')
    if not 16<=args.num_predict<=256:
        raise ValueError('Bounded answer length required')
    proof = json.loads(args.proof.read_text())
    projector_sha = digest(args.projector)
    export = json.loads(args.projector.with_suffix('.export.json').read_text())
    if not proof['passed'] or proof.get('projector_sha256')!=projector_sha or export['sha256']!=projector_sha:
        raise RuntimeError('Native projection proof is not bound to this candidate')
    if proof.get('checkpoint_sha256')!=export['checkpoint_sha256']:
        raise RuntimeError('Native and export checkpoint identities differ')
    main_sha = MAINS[args.precision]
    if args.create_only:
        stock.import_projector(args.projector, projector_sha)
        license_text = '\n\n'.join(path.read_text() for path in
            [RELEASE/'source/LICENSE', ROOT/'VISION-DONOR-LICENSE.txt', ROOT/'QWEN-VISION-UPSTREAM-LICENSE.txt'])
        result = stock.api(11436, '/api/create', {'model': args.tag, 'stream': False, 'requires': '0.35.0',
            'files': {'Kolibri-1-'+args.precision+'.gguf': 'sha256:'+main_sha,
                      args.projector.name: 'sha256:'+projector_sha},
            'template': (RELEASE/'kolibri-ollama-tools.gotmpl').read_text(), 'license': license_text,
            'info': {'capabilities': ['completion','vision','tools','thinking']},
            'parameters': {'num_ctx':8192,'num_predict':args.num_predict,'temperature':0,'stop':['<|im_end|>']}})
        if result.get('status')!='success':
            raise RuntimeError('Stock experimental create failed')
        print(json.dumps({'created':args.tag,'precision':args.precision,'projector_sha256':projector_sha}))
        return
    discovery = json.loads(subprocess.run(['docker','gpu','discover'],capture_output=True,text=True,check=True).stdout)
    print(json.dumps({'warnings':discovery['warnings'],'active_leases':discovery['active_leases']}),flush=True)
    if selected_gpu not in discovery['selected_gpu_ids'] or discovery.get('gpu_health',{}).get('admission_blocked',True):
        raise RuntimeError('Selected GPU is quarantined or requires recovery')
    if any(selected_gpu in lease['gpu_uuids'] for lease in discovery['active_leases']):
        raise RuntimeError('Evaluation GPU still reserved by an external workload')
    show = stock.api(11436, '/api/show', {'model':args.tag})
    if not {'completion','vision','tools','thinking'}.issubset(show.get('capabilities',[])):
        raise RuntimeError('Experimental packing lacks expected capabilities')
    model_name,model_tag = args.tag.rsplit(':',1)
    manifest_path = Path('/srv/ollama/models/manifests/registry.ollama.ai/library')/model_name/model_tag
    manifest = json.loads(manifest_path.read_text())
    model_layers = [layer['digest'] for layer in manifest['layers'] if
                    layer['mediaType']=='application/vnd.ollama.image.model']
    projector_layers = [layer['digest'] for layer in manifest['layers'] if
                        layer['mediaType']=='application/vnd.ollama.image.projector']
    if model_layers!=['sha256:'+main_sha] or projector_layers!=['sha256:'+projector_sha]:
        raise RuntimeError('Installed tag does not contain the evaluated weights')
    manifest_sha = digest(manifest_path)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(30);sock.connect('/run/ollama-unify/gpu-negotiator.sock')
        sock.sendall((json.dumps({'action':'set_model_gpus','model':args.tag,'gpu_uuids':[selected_gpu]})+'\n').encode())
        if not json.loads(sock.makefile().readline()).get('ok'):
            raise RuntimeError('Scoped model placement failed')
    examples = cohort(args.cohort)
    cohort_sha = hashlib.sha256(json.dumps(examples,sort_keys=True).encode()).hexdigest()
    path = ROOT/'compatibility'/(args.run_id+'-quality-evaluation.json')
    report = {'utc':datetime.now(timezone.utc).isoformat(),'tag':args.tag,'precision':args.precision,
        'language_sha256':main_sha,'projector_sha256':projector_sha,'cohort_sha256':cohort_sha,
        'manifest_sha256':manifest_sha,'gpu_uuids':[selected_gpu],
        'scope':'bounded image-fenced comparison; not a full official benchmark or best-in-class result',
        'scoring_notes':'VQA-with-position scores the answer field only; bbox accuracy is not assessed. Long captions retain raw responses for inspection.',
        'num_predict':args.num_predict,'optimizer_updates':proof['optimizer_updates'],'published':False,'examples':[]}
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k)!=report[k] for k in ['tag','language_sha256','projector_sha256','cohort_sha256','manifest_sha256','num_predict']):
            raise RuntimeError('Resume would mix candidates, media or generation settings')
        report = old
    entries = {e['id']:e for e in report['examples']}
    for e in examples:
        if e['id'] not in entries:
            entries[e['id']] = {**e,'answers':{}}
            report['examples'].append(entries[e['id']])
    for tag in ([args.tag] if args.skip_donor else [DONOR,args.tag]):
        for e in examples:
            entry = entries[e['id']]
            if tag in entry['answers']:
                continue
            response = stock.api(11434,'/api/chat',{'model':tag,'think':False,'stream':False,'keep_alive':'5m',
                'messages':[{'role':'user','content':e['question'],
                    'images':[base64.b64encode((ROOT/e['image']).read_bytes()).decode()]}],
                'options':{'num_gpu':-1,'num_ctx':8192,'num_predict':args.num_predict,'temperature':0}},
                headers={'X-Ollama-Unify-GPU-UUIDs':selected_gpu,'X-Ollama-Unify-Workload-Class':'background'},timeout=900)
            answer = response['message'].get('content','')
            result = {'content':answer,'done':response.get('done'),'reason':response.get('done_reason'),
                'prompt_tokens':response.get('prompt_eval_count'),'output_tokens':response.get('eval_count')}
            if e['score_short_answer']:
                scored = answer_for_scoring(answer,e['config'])
                result['scored_answer'] = scored
                result['normalized_exact'] = any(previous.normalize(scored)==previous.normalize(ref) for ref in e['references'])
                result['anls'] = max(previous.anls(scored,ref) for ref in e['references'])
            entry['answers'][tag] = result
            path.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'id':e['id'],'model':tag,'references':e['references'],**result}),flush=True)
    report['completed'] = True
    summary = {}
    for tag in ([args.tag] if args.skip_donor else [DONOR,args.tag]):
        summary[tag] = {}
        for config in sorted({e['config'] for e in report['examples']}):
            results = [e['answers'][tag] for e in report['examples'] if e['config']==config and e['score_short_answer']]
            if results:
                summary[tag][config] = {'n':len(results),'exact':sum(r['normalized_exact'] for r in results),
                    'mean_anls':sum(r['anls'] for r in results)/len(results)}
    report['summary'] = summary
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed':True,'summary':summary}),flush=True)


if __name__ == '__main__':
    main()
