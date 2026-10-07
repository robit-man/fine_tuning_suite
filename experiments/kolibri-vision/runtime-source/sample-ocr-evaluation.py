"""Bounded OCRBench-v2 short-answer subset, evaluation only and image-fenced.

This does not implement or claim the full official benchmark score. It retains
task types, raw answers, image identities, and the pinned public revision.
"""
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path

from huggingface_hub import HfFileSystem
from PIL import Image,ImageOps
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ocrbench-evaluation'
spec=importlib.util.spec_from_file_location('sampling',ROOT/'runtime-source/sample-training-data.py')
sampling=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampling)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    info=json.loads((ROOT/'data-sources/lmms-lab--OCRBench-v2-benchmark-info.json').read_text())
    used=[json.loads(line)for line in (ROOT/'broader-alignment-data/examples.jsonl').read_text().splitlines()]
    hashes={e['image_sha256']for e in used}
    dhashes=[int(e['image_dhash'],16)for e in used]
    budget=sampling.Budget(768*2**20,128*2**20,OUT)
    fs=HfFileSystem();entries=[];rejects=Counter();types=Counter();seen=set()
    shards=[s['rfilename']for s in info['siblings']if s['rfilename'].endswith('.parquet')]
    for shard in shards:
        if len(entries)>=60:break
        try:
            with sampling.bounded_file(fs,f"datasets/{info['id']}@{info['sha']}/{shard}",budget)as file:
                parquet=pq.ParquetFile(file)
                for group in range(parquet.metadata.num_row_groups):
                    if len(entries)>=60:break
                    meta=parquet.metadata.row_group(group)
                    compressed=sum(meta.column(i).total_compressed_size for i in range(meta.num_columns))
                    if compressed>128*2**20:rejects['oversized_rowgroup']+=1;continue
                    table=parquet.read_row_group(group,use_threads=False)
                    for row in table.to_pylist():
                        if len(entries)>=60:break
                        answers=row.get('answers')
                        if not isinstance(answers,list)or not answers or any(not isinstance(a,str)or not 1<=len(a)<=80 for a in answers):
                            rejects['non_short_text_answers']+=1;continue
                        if any(any(c in a for c in ['[',']','{','}'])for a in answers):
                            rejects['structured_answer_task']+=1;continue
                        kind=row.get('type','unknown')
                        if types[kind]>=15:rejects['task_type_cap']+=1;continue
                        payload=row.get('image',{}).get('bytes')
                        if not payload or len(payload)>32*2**20:rejects['image_payload_cap']+=1;continue
                        with Image.open(io.BytesIO(payload))as image:
                            if image.width*image.height>32000000:rejects['image_pixel_cap']+=1;continue
                            image=ImageOps.exif_transpose(image).convert('RGB')
                            digest,perceptual,size=sampling.image_identity(image)
                            if digest in hashes or digest in seen:
                                rejects['exact_used_image']+=1;continue
                            if min((int(perceptual,16)^h).bit_count()for h in dhashes)<=2:
                                rejects['near_used_image']+=1;continue
                            buffer=io.BytesIO();image.save(buffer,format='PNG')
                        path=OUT/'images'/(digest+'.png')
                        sampling.Budget.write(budget,path,buffer.getvalue())
                        entry={'id':digest+':ocrbench-v2','image':str(path.relative_to(OUT)),
                            'image_sha256':digest,'image_dhash':perceptual,'size':list(size),
                            'question':row['question'],'answers':answers,'task_type':kind,
                            'dataset_name':row.get('dataset_name'),'source_id':row.get('id'),
                            'repo':info['id'],'revision':info['sha'],'shard':shard,'rowgroup':group,
                            'training_excluded':True}
                        entries.append(entry);seen.add(digest);types[kind]+=1
                        (OUT/'examples.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in entries))
                        print(json.dumps({'accepted':len(entries),'task_type':kind,'transfer_upper_bound':budget.transferred}),flush=True)
        except Exception as error:
            rejects[type(error).__name__]+=1
            # Signed remote URLs/credentials are deliberately never printed.
            if budget.transferred>=budget.transfer_limit-128*2**20:break
    report={'scope':'image-fenced short-answer subset; not the full official OCRBench-v2 score',
        'repo':info['id'],'revision':info['sha'],'examples':len(entries),'task_counts':dict(types),
        'used_image_fence_count':len(hashes),'training_examples':0,'rejects':dict(rejects),
        'transfer_bytes_upper_bound':budget.transferred,'transfer_limit':budget.transfer_limit,
        'artifact_bytes':budget.output,'artifact_limit':budget.output_limit}
    (OUT/'sampling-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
