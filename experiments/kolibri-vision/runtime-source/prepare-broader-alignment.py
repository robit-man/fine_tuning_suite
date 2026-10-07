"""Bounded alignment cohort reusing pretrained weights, with explicit fencing.

Natural images come from the pinned sampled public source. Synthetic training
labels are generated from the actual pixels and excluded from evaluation IDs.
No evaluation image enters training, including prior development examples.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import importlib.util

from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'broader-alignment-data'
OUT.mkdir(exist_ok=True)
(OUT/'images').mkdir(exist_ok=True)
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
MONO=Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
font=lambda size,mono=False:ImageFont.truetype(str(MONO if mono else FONT),size)
rng=random.Random(610062617)
spec=importlib.util.spec_from_file_location('image_identity',ROOT/'runtime-source/sample-training-data.py')
canonical=importlib.util.module_from_spec(spec);spec.loader.exec_module(canonical)


def identity(image):
    digest,perceptual,_=canonical.image_identity(image)
    return digest,perceptual


def main():
    protected=[]
    for pool,filename in [('data-pilot-counted','examples.jsonl'),('alignment-pilot','examples.jsonl'),
        ('independent-natural-pool','evaluation-only-examples.jsonl')]:
        for line in (ROOT/pool/filename).read_text().splitlines():
            e=json.loads(line)
            if e['pilot_split']!='training':protected.append(e)
    fence={e['image_sha256']for e in protected}
    dhashes=[int(e['image_dhash'],16)for e in protected]
    for e in json.loads((ROOT/'independent-checks/manifest.json').read_text())['examples']:
        with Image.open(ROOT/e['image'])as image:digest,dhash=identity(image)
        fence.add(digest);dhashes.append(int(dhash,16))
    examples=[];seen=set();rejects=Counter()
    for line in (ROOT/'targeted-natural-1000/examples.jsonl').read_text().splitlines():
        e=json.loads(line)
        if e['image_sha256']in seen:rejects['duplicate_image']+=1;continue
        if e['pilot_split']=='training' and (e['image_sha256']in fence or min((int(e['image_dhash'],16)^h).bit_count()for h in dhashes)<=2):
            e['pilot_split']='development_holdout';e['fenced_from_training']=True
        e['image']=str(Path('../targeted-natural-1000')/e['image'])
        examples.append(e);seen.add(e['image_sha256'])
    for line in (ROOT/'alignment-pilot/examples.jsonl').read_text().splitlines():
        e=json.loads(line)
        if e['image_sha256']in seen:continue
        e['image']=str(Path('../alignment-pilot')/e['image'])
        examples.append(e);seen.add(e['image_sha256'])
    synthetic_limit=128*2**20
    written=0
    for n in range(500):
        task=['invoice','paragraph','table','ui','chart'][n%5]
        width,height=rng.choice([(1100,850),(1200,950),(1400,1000)])
        image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image)
        left=rng.randint(40,95);top=rng.randint(40,85)
        if task=='invoice':
            identifier=rng.choice(['INV','REC','DOC','PAY'])+'-'+str(rng.randint(1000,9999))
            subtotal=rng.randint(1500,40000);shipping=rng.randint(100,4000);total=subtotal+shipping
            money=lambda cents:f'${cents/100:.2f}'
            draw.text((left,top),'Transaction summary',font=font(42),fill='black')
            draw.text((left,top+110),'Reference: '+identifier,font=font(rng.randint(28,38),True),fill='black')
            draw.text((left,top+230),'Subtotal: '+money(subtotal),font=font(32),fill='black')
            draw.text((left,top+305),'Shipping: '+money(shipping),font=font(32),fill='black')
            draw.text((left,top+425),'TOTAL: '+money(total),font=font(42),fill='black')
            if rng.random()<.5:question,answer='What is the transaction reference? Return only the identifier.',identifier
            else:question,answer='What is the final total including shipping? Return only the dollar amount.',money(total)
        elif task=='paragraph':
            alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            code=''.join(rng.choice(alphabet)for _ in range(4))+'-'+''.join(rng.choice(alphabet)for _ in range(4))
            draw.text((left,top),'Maintenance appointment',font=font(40),fill='black')
            lines=['Your appointment is confirmed for next week.',
                'The access code for this visit is '+code+'.',
                'Keep the access code ready for the technician.',
                'This notice expires after the appointment.']
            for i,line in enumerate(lines):draw.text((left,top+140+i*70),line,font=font(rng.randint(25,30)),fill='black')
            question,answer='Read the access code. Return only the code.',code
        elif task=='table':
            names=rng.sample(['Falcon','Otter','Lynx','Badger','Robin','Raven','Finch','Fox'],3)
            counts=rng.sample(range(10,95),3);target=rng.randrange(3)
            draw.text((left,top),'Warehouse inventory',font=font(42),fill='black')
            for i,row in enumerate([['Item','Count','Warehouse']]+[[name,str(count),site]for name,count,site in zip(names,counts,['East','West','North'])]):
                y=top+140+i*110
                for x,text in zip([left,left+450,left+700],row):draw.text((x,y),text,font=font(32),fill='black')
                draw.line((left,y+70,width-60,y+70),fill='#a0a0a0',width=2)
            question,answer=f'How many {names[target]} items are there? Return only the number.',str(counts[target])
        elif task=='ui':
            label=rng.choice(['Refresh','Continue','Export','Upload','Download','Connect','Validate','Publish','Schedule','Confirm'])
            draw.rectangle((0,0,width,100),fill='#17243d')
            draw.text((left,28),'Application preferences',font=font(36),fill='white')
            draw.text((left,200),'Action required',font=font(38),fill='black')
            draw.text((left,280),'Choose the next action below.',font=font(28),fill='black')
            x=left if rng.random()<.5 else width-420
            draw.rounded_rectangle((x,520,x+350,625),radius=12,fill='#1666ba')
            draw.text((x+25,550),label,font=font(33),fill='white')
            draw.rounded_rectangle((width-x-350,680,width-x,780),radius=12,fill='#dddddd')
            draw.text((width-x-325,705),'Cancel',font=font(30),fill='black')
            question,answer='What does the blue action button say? Return only its label.',label
        else:
            labels=rng.sample(['Alpha','Delta','Echo','Foxtrot','Golf','Hotel','India','Juliet'],3)
            heights=rng.sample(range(130,380,20),3)
            draw.text((left,top),'Production report',font=font(42),fill='black')
            baseline=height-160
            draw.line((left+40,top+100,left+40,baseline),fill='black',width=3)
            draw.line((left+40,baseline,width-50,baseline),fill='black',width=3)
            for i,(label,h)in enumerate(zip(labels,heights)):
                x=left+130+i*280
                draw.rectangle((x,baseline-h,x+130,baseline),fill='#367bb8')
                draw.text((x-10,baseline+35),label,font=font(27),fill='black')
            question,answer='Which category has the tallest bar? Return only its name.',labels[heights.index(max(heights))]
        digest,dhash=identity(image)
        if digest in seen or digest in fence:raise RuntimeError('Synthetic duplicate/heldout collision')
        file=OUT/'images'/(digest+'.png');image.save(file)
        written+=file.stat().st_size
        if written>synthetic_limit:raise RuntimeError('Synthetic artifact cap exceeded')
        examples.append({'id':digest+':synthetic-'+task,'config':'synthetic-'+task,
            'image':str(file.relative_to(OUT)),'image_sha256':digest,'image_dhash':dhash,
            'user':question,'assistant':answer,'pilot_split':'development_holdout'if (n//5)%5==4 else'training',
            'source':'deterministic generated pixels with exact labels','generator_seed':610062617})
        seen.add(digest)
    train={e['image_sha256']for e in examples if e['pilot_split']=='training'}
    validation={e['image_sha256']for e in examples if e['pilot_split']=='development_holdout'}
    if train&validation or train&fence:raise RuntimeError('Training/evaluation overlap')
    if len(train)>3000 or len(validation)>1000:raise RuntimeError('Bounded cohort size exceeded')
    (OUT/'examples.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in examples))
    report={'scope':'targeted pretrained projector alignment, no vision tower or language retraining',
        'examples':len(examples),'training_examples':len(train),'validation_examples':len(validation),
        'protected_evaluation_images':len(fence),'protected_evaluation_overlap':0,
        'source_counts':dict(Counter(e['config']for e in examples)),
        'synthetic_bytes':written,'synthetic_byte_limit':synthetic_limit,'rejects':dict(rejects)}
    (OUT/'preparation-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
