"""Small, bounded frozen-Kolibri alignment of an existing pretrained projector.

No decoder weights change. Validation images never receive gradient updates.
This produces experimental checkpoints only, never a publication.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import hashlib
import shutil
import time

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT.parent/'kolibri-1-20261004'
GPUS=['GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0','GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a']


def load_module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'runtime-source'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ready-file',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--learning-rate',type=float,default=2e-5)
    parser.add_argument('--feature-cache',type=Path,default=ROOT/'feature-cache-1024/manifest.json')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'transferred-projector.pt')
    parser.add_argument('--data-manifest',type=Path)
    parser.add_argument('--run-id',default='targeted')
    parser.add_argument('--max-validation-per-config',type=int,default=0)
    parser.add_argument('--existing-active-lease',action='store_true')
    parser.add_argument('--checkpoint-every',type=int,default=50)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--expected-gpus',default=','.join(GPUS))
    parser.add_argument('--cpu-layers',type=int,default=0)
    parser.add_argument('--cpu-threads',type=int,default=16)
    parser.add_argument('--selection-metric',choices=['token','macro'],default='token')
    args=parser.parse_args()
    expected_gpus=args.expected_gpus.split(',')
    if (not expected_gpus or len(expected_gpus)>2 or len(set(expected_gpus))!=len(expected_gpus) or
        any(not g.startswith('GPU-') for g in expected_gpus) or
        os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')!=expected_gpus or not os.environ.get('OLLAMA_UNIFY_GPU_LEASE')):
        raise RuntimeError('Exact declared UUID-scoped broker reservation required')
    if not 0<=args.cpu_layers<50 or not 1<=args.cpu_threads<=16:
        raise ValueError('Invalid bounded CPU offload')
    if not 1<=args.epochs<=10:raise ValueError('Bounded alignment epochs required')
    if not 1<=args.checkpoint_every<=250:raise ValueError('Bounded checkpoint interval required')
    if not args.run_id.replace('-','').isalnum():raise ValueError('Invalid run identifier')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(6100626)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    qualification=load_module('qualification','qualify-full-training.py')
    control=qualification.control
    module=load_module('frozen_decoder','frozen-decoder.py')
    initial=torch.load(args.checkpoint,weights_only=True,map_location='cpu')
    resume=initial.get('training_state') if args.resume else None
    if args.resume and not resume:raise RuntimeError('Checkpoint has no exact training-resume state')
    if not initial.get('pretrained_transfer'):raise RuntimeError('Existing pretrained initialization required')
    started=time.monotonic()
    def progress(layer,device):
        if layer%8==0:print(json.dumps({'loaded_layer':layer,'device':device}),flush=True)
    decoder=module.FrozenAFMoE(str(RELEASE/'Kolibri-1-stock-BF16-release.gguf'),
        devices=tuple('cuda:'+str(n) for n in range(len(expected_gpus))),
        cpu_layers=args.cpu_layers,checkpoint_layers=True,grouped_experts=True,
        cpu_embedding=True,cpu_head=True,load_progress=progress)
    bridge=torch.nn.Linear(4608,2560,device='cuda:0',dtype=torch.float32)
    bridge.load_state_dict(initial['state_dict'])
    optimizer=torch.optim.AdamW(bridge.parameters(),lr=args.learning_rate,weight_decay=0)
    for parameter in bridge.parameters():
        optimizer.state[parameter]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(parameter),
            'exp_avg_sq':torch.zeros_like(parameter)}
    if resume:
        optimizer.load_state_dict(initial['optimizer_state'])
    for device in decoder.devices:torch.cuda.synchronize(device)
    args.ready_file.write_text('frozen BF16 decoder and transferred bridge resident\n')
    if args.existing_active_lease:
        control('ready')
    for _ in range(40):
        lease=control('heartbeat');lease=lease.get('lease',lease)
        if lease.get('state')=='active':break
        time.sleep(.25)
    else:raise RuntimeError('Supervisor readiness timeout')
    control('prepare')
    inputs=json.loads((ROOT/'compatibility/full-bf16-stock-input.json').read_text())
    references=[ROOT/'compatibility/stock-gpu-layer-capture/logits.f32',
        ROOT/'compatibility/full-bf16-stock.logits.bin',ROOT/'compatibility/stock-layer-capture/logits.f32']
    with torch.no_grad():
        current=decoder(decoder.embeddings(torch.tensor(inputs['tokens'])),output_positions=[len(inputs['tokens'])-1])[0].cpu()
    parity=[]
    for reference in references:
        expected=torch.from_numpy(np.fromfile(reference,dtype=np.float32))
        rmse=float((current-expected).square().mean().sqrt())
        p=expected.log_softmax(-1)
        kl=float((p.exp()*(p-current.log_softmax(-1))).sum())
        parity.append({'reference':str(reference),'rmse':rmse,'kl':kl,
            'passed':rmse<=.05 and kl<=.0005 and int(current.argmax())==int(expected.argmax())})
    if not any(p['passed'] for p in parity):raise RuntimeError('Native decoder parity failed; no training')
    control('ready')
    tokenizer=Tokenizer.from_file(str(RELEASE/'source/tokenizer.json'))
    cache=json.loads(args.feature_cache.read_text())
    feature_paths={e['id']:ROOT/e['file'] for e in cache['examples']}
    examples=[]
    if args.data_manifest:
        examples=[json.loads(line)for line in args.data_manifest.read_text().splitlines()]
    else:
        for pool in ['data-pilot-counted','alignment-pilot']:
            examples.extend(json.loads(line) for line in (ROOT/pool/'examples.jsonl').read_text().splitlines())
    data_sha=hashlib.sha256(json.dumps(examples,sort_keys=True).encode()).hexdigest()
    cache_sha=hashlib.sha256(args.feature_cache.read_bytes()).hexdigest()
    if resume and any(resume.get(k)!=v for k,v in {
        'run_id':args.run_id,'data_sha256':data_sha,'feature_cache_sha256':cache_sha,
        'epochs':args.epochs,'learning_rate':args.learning_rate,
        'max_validation_per_config':args.max_validation_per_config,
        'selection_metric':args.selection_metric,'cpu_layers':args.cpu_layers,
        'gpu_uuids':expected_gpus}.items()):
        raise RuntimeError('Resume would mix data, cache, run identity or optimizer settings')
    if len(examples)>5000 or len(examples)*args.epochs>20000:raise RuntimeError('Alignment budget exceeded')
    training=[e for e in examples if e['pilot_split']=='training']
    validation=[e for e in examples if e['pilot_split']=='development_holdout']
    validation_total=len(validation)
    if args.max_validation_per_config:
        selected=[];counts={}
        for e in validation:
            count=counts.get(e['config'],0)
            if count<args.max_validation_per_config:
                selected.append(e);counts[e['config']]=count+1
        validation=selected
    if {e['id'].split(':')[0] for e in training}&{e['id'].split(':')[0] for e in validation}:
        raise RuntimeError('Image split overlap')
    # Calibrate only magnitude using training images/questions. Qwen's learned
    # image embeddings have a larger scale than Kolibri's MuP input space.
    if initial.get('optimizer_updates',0)>0:
        calibration=initial['projection_calibration']
    else:
        control('prepare')
        image_energy=text_energy=image_elements=text_elements=0
        with torch.no_grad():
            for example in training:
                features=torch.from_numpy(np.array(np.load(feature_paths[example['id']]),dtype=np.float32)).to('cuda:0')
                projection=bridge(features)
                image_energy+=float(projection.square().sum());image_elements+=projection.numel()
                ids=tokenizer.encode(example['user'],add_special_tokens=False).ids
                words=decoder.embeddings(torch.tensor(ids))
                text_energy+=float(words.square().sum());text_elements+=words.numel()
            scale=math.sqrt((text_energy/text_elements)/(image_energy/image_elements))
            bridge.weight.mul_(scale);bridge.bias.mul_(scale)
            del features,projection,words
        calibration={'source_image_rms':math.sqrt(image_energy/image_elements),
            'target_text_rms':math.sqrt(text_energy/text_elements),'scale':scale,'training_images_only':True}
        control('ready')
    print(json.dumps({'projection_calibration':calibration}),flush=True)
    def forward(example, blank=False):
        features=torch.from_numpy(np.array(np.load(feature_paths[example['id']]),dtype=np.float32)).to('cuda:0')
        if blank:features.zero_()
        prefix="<|im_start|>system\n# Reasoning effort\n\nReasoning is disabled. Proceed straight to answering according to the user's instructions.<|im_end|>\n<|im_start|>user\n<|vision_start|>"
        suffix='<|vision_end|>'+example['user'].replace('<image>','')+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        ids0=tokenizer.encode(prefix,add_special_tokens=False).ids
        ids1=tokenizer.encode(suffix,add_special_tokens=False).ids
        answer=tokenizer.encode(example['assistant']+'<|im_end|>',add_special_tokens=False).ids[:128]
        before=decoder.embeddings(torch.tensor(ids0))
        after=decoder.embeddings(torch.tensor(ids1+answer[:-1]))
        projection=bridge(features)
        inputs=torch.cat([before,projection,after],0)
        answer_start=len(ids0)+len(features)+len(ids1)-1
        logits=decoder(inputs,output_positions=list(range(answer_start,answer_start+len(answer))))
        loss=torch.nn.functional.cross_entropy(logits.float(),torch.tensor(answer,device=logits.device))
        return loss,len(answer),float(projection.detach().square().mean().sqrt())
    report={'utc':datetime.now(timezone.utc).isoformat(),'scope':'bounded pretrained-projection alignment; small development cohort only',
        'pretrained_transfer':initial['pretrained_transfer'],'projection_calibration':calibration,
        'feature_cache':str(args.feature_cache),'training_examples':len(training),
        'validation_examples':len(validation),'full_validation_partition_examples':validation_total,
        'epochs':args.epochs,'learning_rate':args.learning_rate,'warm_start_checkpoint':str(args.checkpoint),
        'language_weights_changed':False,'parity':parity,'steps':[],'validations':[],
        'gpu_uuids':expected_gpus,'cpu_layers':args.cpu_layers,'cpu_threads':args.cpu_threads,
        'selection_metric':args.selection_metric,
        'published':False,'optimizer_updates':initial.get('optimizer_updates',0),'stage_optimizer_updates':0}
    path=ROOT/'compatibility'/(args.run_id+'-alignment-report.json')
    best_loss=float('inf')
    active_epoch=0;active_order=[];active_cursor=0
    if resume:
        report=initial['training_report']
        report.setdefault('resumes',[]).append({'utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint':str(args.checkpoint),'optimizer_updates':report['optimizer_updates']})
        best_loss=resume['best_validation_loss']
        active_epoch=resume['epoch'];active_order=resume['order'];active_cursor=resume['cursor']
        if len(active_order)!=len(training) or sorted(active_order)!=list(range(len(training))) or not 0<=active_cursor<=len(training):
            raise RuntimeError('Invalid saved epoch order or cursor')
    def save(name):
        checkpoint=ROOT/name;partial=checkpoint.with_suffix('.pt.partial')
        if shutil.disk_usage(ROOT).free<64*2**30+256*2**20:
            raise RuntimeError('Checkpoint would violate shared-disk headroom')
        def cpu_state(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu()
            if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu_state(v) for v in value]
            return value
        value={'state_dict':{k:v.detach().cpu() for k,v in bridge.state_dict().items()},
            'optimizer_updates':report['optimizer_updates'],'pretrained_transfer':initial['pretrained_transfer'],
            'projection_calibration':calibration,
            'scope':report['scope'],'validation':report['validations'][-1] if report['validations'] else None}
        if active_epoch:
            value.update({'optimizer_state':cpu_state(optimizer.state_dict()),'training_report':report,
                'training_state':{'run_id':args.run_id,'data_sha256':data_sha,'feature_cache_sha256':cache_sha,
                    'epochs':args.epochs,'learning_rate':args.learning_rate,'epoch':active_epoch,
                    'max_validation_per_config':args.max_validation_per_config,
                    'selection_metric':args.selection_metric,'cpu_layers':args.cpu_layers,
                    'gpu_uuids':expected_gpus,
                    'order':active_order,'cursor':active_cursor,'best_validation_loss':best_loss,
                    'rng_state':torch.random.get_rng_state(),
                    'epoch_validation_complete':bool(report['validations'] and
                        report['validations'][-1]['epoch']==active_epoch and
                        report['validations'][-1]['optimizer_updates']==report['optimizer_updates'])}})
        torch.save(value,partial)
        if partial.stat().st_size>256*2**20:
            raise RuntimeError('Checkpoint exceeds its explicit artifact bound')
        partial.replace(checkpoint)
    def evaluate(epoch):
        nonlocal best_loss
        values=[]
        with torch.no_grad():
            for example in validation:
                control('prepare')
                validation_started=time.monotonic()
                loss,tokens,rms=forward(example)
                blank_loss,_,_=forward(example,blank=True)
                values.append({'id':example['id'],'config':example['config'],'loss':float(loss),
                    'blank_loss':float(blank_loss),'answer_tokens':tokens,'image_embedding_rms':rms,
                    'seconds':time.monotonic()-validation_started})
                control('ready')
                report['validation_progress']={'epoch':epoch,'completed_examples':len(values),'total_examples':len(validation)}
                path.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({'phase':'validation',**report['validation_progress'],
                    'seconds':values[-1]['seconds']}),flush=True)
        mean_loss=sum(e['loss']*e['answer_tokens'] for e in values)/sum(e['answer_tokens'] for e in values)
        blank_loss=sum(e['blank_loss']*e['answer_tokens'] for e in values)/sum(e['answer_tokens'] for e in values)
        configurations={e['config'] for e in values}
        macro_loss=sum(sum(e['loss'] for e in values if e['config']==name)/
            sum(e['config']==name for e in values) for name in configurations)/len(configurations)
        result={'epoch':epoch,'optimizer_updates':report['optimizer_updates'],'mean_token_loss':mean_loss,
            'mean_config_loss':macro_loss,
            'blank_mean_token_loss':blank_loss,'examples':values}
        report['validations'].append(result)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='examples'}),flush=True)
        criterion=mean_loss if args.selection_metric=='token' else macro_loss
        if criterion<best_loss:
            best_loss=criterion;save('aligned-best-projector.pt'if args.run_id=='targeted'else args.run_id+'-best-projector.pt')
    if not resume:evaluate(0)
    else:torch.random.set_rng_state(resume['rng_state'])
    start_epoch=resume['epoch'] if resume else 1
    if resume and resume['epoch_validation_complete']:start_epoch+=1
    for epoch in range(start_epoch,args.epochs+1):
        if resume and epoch==resume['epoch']:
            order=resume['order'];cursor=resume['cursor']
        else:order=torch.randperm(len(training)).tolist();cursor=0
        active_epoch=epoch;active_order=order;active_cursor=cursor
        for cursor in range(cursor,len(order)):
            index=order[cursor]
            example=training[index]
            control('prepare')
            step_started=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss,tokens,rms=forward(example)
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.0))
            if not torch.isfinite(loss) or not math.isfinite(norm) or norm<=0:raise RuntimeError('Invalid image gradient')
            optimizer.step()
            report['optimizer_updates']+=1
            report['stage_optimizer_updates']+=1
            entry={'epoch':epoch,'update':report['optimizer_updates'],'id':example['id'],
                'loss':float(loss.detach()),'gradient_norm':norm,'answer_tokens':tokens,
                'image_embedding_rms':rms,'seconds':time.monotonic()-step_started}
            report['steps'].append(entry)
            active_cursor=cursor+1
            print(json.dumps(entry),flush=True)
            path.write_text(json.dumps(report,indent=2)+'\n')
            if report['stage_optimizer_updates']%args.checkpoint_every==0:
                save(args.run_id+'-latest-projector.pt')
            control('ready')
        evaluate(epoch)
        save('aligned-latest-projector.pt'if args.run_id=='targeted'else args.run_id+'-latest-projector.pt')
    report['seconds']=time.monotonic()-started
    report['completed']=True
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed':True,'optimizer_updates':report['optimizer_updates'],
        'seconds':report['seconds'],'best_validation_loss':best_loss}),flush=True)


if __name__=='__main__':main()
