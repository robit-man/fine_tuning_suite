"""Keep new natural images outside every used alignment/development image."""
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
used=[]
for pool in ['data-pilot-counted','alignment-pilot']:
    used.extend(json.loads(line) for line in (ROOT/pool/'examples.jsonl').read_text().splitlines())
hashes={e['image_sha256'] for e in used}
perceptual=[int(e['image_dhash'],16) for e in used]
source=ROOT/'independent-natural-pool'
retained=[];rejects=Counter()
for line in (source/'examples.jsonl').read_text().splitlines():
    entry=json.loads(line)
    if entry['image_sha256'] in hashes:
        rejects['exact_used_image']+=1;continue
    if min((int(entry['image_dhash'],16)^h).bit_count() for h in perceptual)<=2:
        rejects['near_used_image_dhash_hamming_le_2']+=1;continue
    entry['sampling_split']=entry['pilot_split']
    entry['pilot_split']='independent_evaluation_only'
    entry['training_excluded']=True
    retained.append(entry)
output=source/'evaluation-only-examples.jsonl'
output.write_text(''.join(json.dumps(e)+'\n' for e in retained))
report={'scope':'new images excluded from this run\'s alignment and development images; upstream pretraining contamination unknown',
    'used_image_fence_count':len(hashes),'near_duplicate_hamming_threshold':2,
    'retained':len(retained),'configs':dict(Counter(e['config'] for e in retained)),
    'rejected':dict(rejects),'training_examples':0,'manifest':str(output)}
(source/'evaluation-fence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
