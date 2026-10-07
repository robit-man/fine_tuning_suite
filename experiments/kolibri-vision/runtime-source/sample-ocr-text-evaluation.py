"""Stratify a bounded OCRBench-v2 text subset, never used for training."""
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path

from huggingface_hub import HfFileSystem
from PIL import Image, ImageOps
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'ocrbench-text-evaluation'
spec = importlib.util.spec_from_file_location('sampling',ROOT/'runtime-source/sample-training-data.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)
TARGETS = {'text recognition en':20, 'fine-grained text recognition en':10,
    'VQA with position en':10, 'cognition VQA en':10, 'diagram QA en':10, 'reasoning VQA en':10}


def short_answers(answers):
    return isinstance(answers,list) and answers and all(isinstance(a,str) and
        1<=len(a)<=80 and not any(c in a for c in '[]{}') for a in answers)


def main():
    if (OUT/'examples.jsonl').exists():
        raise RuntimeError('Choose a fresh output to preserve earlier evaluation media')
    OUT.mkdir(exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    info = json.loads((ROOT/'data-sources/lmms-lab--OCRBench-v2-benchmark-info.json').read_text())
    inventory = json.loads((ROOT/'compatibility/ocrbench-task-group-inventory.json').read_text())
    used = [json.loads(line) for line in (ROOT/'broader-alignment-data/examples.jsonl').read_text().splitlines()]
    hashes = {e['image_sha256'] for e in used}
    dhashes = [int(e['image_dhash'],16) for e in used]
    budget = sampling.Budget(384*2**20,64*2**20,OUT)
    fs = HfFileSystem()
    examples = []
    seen = set()
    counts = Counter()
    rejects = Counter()
    fetched = set()
    exhausted = False
    for kind,target in TARGETS.items():
        # Prefer groups with enough valid short answers and bounded image size.
        # This is a convenience subset, not an official benchmark distribution.
        groups = [g for g in inventory['groups'] if g['short_answer_types'].get(kind,0)]
        groups.sort(key=lambda g:(g['short_answer_types'].get(kind,0)<target,
                                  g['image_column_compressed_bytes'],g['shard'],g['rowgroup']))
        for group in groups:
            if counts[kind]>=target or exhausted:
                break
            key = (group['shard'],group['rowgroup'])
            if key in fetched:
                continue
            if group['image_column_compressed_bytes']>128*2**20:
                rejects['oversized_rowgroup']+=1
                continue
            try:
                with sampling.bounded_file(fs,f"datasets/{info['id']}@{info['sha']}/{key[0]}",budget) as file:
                    parquet = pq.ParquetFile(file)
                    table = parquet.read_row_group(key[1],use_threads=False)
                fetched.add(key)
                for row in table.to_pylist():
                    row_kind = row.get('type')
                    if row_kind not in TARGETS or counts[row_kind]>=TARGETS[row_kind]:
                        continue
                    if not short_answers(row.get('answers')):
                        rejects['non_short_text_answers']+=1
                        continue
                    payload = row.get('image',{}).get('bytes')
                    if not payload or len(payload)>32*2**20:
                        rejects['image_payload_cap']+=1
                        continue
                    with Image.open(io.BytesIO(payload)) as image:
                        if image.width*image.height>32000000:
                            rejects['image_pixel_cap']+=1
                            continue
                        image = ImageOps.exif_transpose(image).convert('RGB')
                        sha,perceptual,size = sampling.image_identity(image)
                        if sha in hashes or sha in seen:
                            rejects['exact_used_image']+=1
                            continue
                        if min((int(perceptual,16)^h).bit_count() for h in dhashes)<=2:
                            rejects['near_used_image']+=1
                            continue
                        buffer = io.BytesIO()
                        image.save(buffer,format='PNG')
                    path = OUT/'images'/(sha+'.png')
                    sampling.Budget.write(budget,path,buffer.getvalue())
                    seen.add(sha)
                    counts[row_kind]+=1
                    examples.append({'id':sha+':ocrbench-v2','image':str(path.relative_to(OUT)),
                        'image_sha256':sha,'image_dhash':perceptual,'size':list(size),
                        'question':row['question'],'answers':row['answers'],'task_type':row_kind,
                        'dataset_name':row.get('dataset_name'),'source_id':row.get('id'),
                        'repo':info['id'],'revision':info['sha'],'shard':key[0],'rowgroup':key[1],
                        'training_excluded':True})
                    (OUT/'examples.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in examples))
                print(json.dumps({'kind':kind,'counts':dict(counts),'transfer_bytes':budget.transferred}),flush=True)
            except RuntimeError:
                rejects['bounded_budget_or_decode_error']+=1
                if budget.transferred>=budget.transfer_limit-128*2**20:
                    exhausted = True
                    break
            except Exception as error:
                rejects[type(error).__name__]+=1
    report = {'repo':info['id'],'revision':info['sha'],'examples':len(examples),'targets':TARGETS,
        'task_counts':dict(counts),'passed_sampling_targets':all(counts[k]>=n for k,n in TARGETS.items()),
        'scope':'image-fenced English short-answer OCRBench-v2 convenience subset; not an official full score',
        'training_examples':0,'used_image_fence_count':len(hashes),'rejects':dict(rejects),
        'transfer_bytes_upper_bound':budget.transferred,'transfer_limit':budget.transfer_limit,
        'artifact_bytes':budget.output,'artifact_limit':budget.output_limit,
        'fetched_groups':[list(key) for key in sorted(fetched)]}
    (OUT/'sampling-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
