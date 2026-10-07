"""Fresh paired image checks: identical questions, changed visual answers."""
import importlib.util
import io
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'counterfactual-checks'
spec = importlib.util.spec_from_file_location('sampling',ROOT/'runtime-source/sample-training-data.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = lambda size:ImageFont.truetype(FONT,size)


def main():
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Preserve existing fresh test evidence')
    OUT.mkdir(exist_ok=True)
    budget = sampling.Budget(0,8*2**20,OUT)
    used = [json.loads(line) for line in (ROOT/'broader-alignment-data/examples.jsonl').read_text().splitlines()]
    hashes = {e['image_sha256'] for e in used}
    perceptuals = [int(e['image_dhash'],16) for e in used]
    examples = []
    seen = set()
    def save(name,image,question,answer,task,pair):
        sha,perceptual,size = sampling.image_identity(image)
        if sha in hashes:
            raise RuntimeError('Fresh check overlaps alignment images')
        distance = min((int(perceptual,16)^h).bit_count() for h in perceptuals)
        if distance<=2:
            raise RuntimeError('Fresh check is perceptually too close to an alignment image')
        path = OUT/(name+'.png')
        buffer = io.BytesIO()
        image.save(buffer,format='PNG')
        sampling.Budget.write(budget,path,buffer.getvalue())
        examples.append({'id':'counterfactual-'+name,'image':str(path.relative_to(ROOT)),
            'question':question,'answer':answer,'task':task,'pair':pair,
            'image_sha256':sha,'image_dhash':perceptual,'nearest_alignment_dhash_distance':distance,
            'size':list(size),'training_excluded':True})
        seen.add(sha)
    for variant,(reference,total,code) in enumerate([
        ('VX-8426','$268.91','L6P9-W3R7'),('JT-3071','$93.48','B2M8-C5H4')]):
        image = Image.new('RGB',(1280,880),'#f0f4ed')
        draw = ImageDraw.Draw(image)
        draw.rectangle((35,35,1245,845),fill='white',outline='#426b40',width=5)
        draw.rectangle((35,35,1245,145),fill='#426b40')
        draw.text((85,60),'FIELDWORK SERVICES',font=font(44),fill='white')
        for y,text,size in [(230,'Invoice reference: '+reference,40),
            (355,'Amount due: '+total,44),(480,'Access code: '+code,38),
            (620,'Payment is due within fourteen days.',30)]:
            draw.text((85,y),text,font=font(size),fill='black')
        for field,question,answer in [('reference','Read the invoice reference. Return only the reference.',reference),
            ('total','Read the amount due. Return only the dollar amount.',total),
            ('code','Read the access code. Return only the code.',code)]:
            save(f'invoice-{variant}-{field}',image,question,answer,'counterfactual_ocr','invoice-'+field)
    for variant,(label,other) in enumerate([('Export','Archive'),('Restore','Refresh')]):
        image = Image.new('RGB',(1100,760),'#faf4e8')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20,20,1080,115),fill='#5c4234')
        draw.text((60,42),'Collection management',font=font(40),fill='white')
        draw.text((70,175),'Choose an action for this collection.',font=font(32),fill='black')
        draw.rounded_rectangle((100,320,470,430),radius=12,fill='#165fbc')
        draw.text((180,350),label,font=font(40),fill='white')
        draw.rounded_rectangle((640,320,1010,430),radius=12,fill='#d6cec2')
        draw.text((700,350),other,font=font(40),fill='black')
        draw.rectangle((70,535,1030,650),outline='#806b51',width=3)
        draw.text((100,565),'12 items selected',font=font(34),fill='black')
        save(f'ui-{variant}',image,'What does the blue button say? Return only its label.',
             label,'counterfactual_screenshot','ui-label')
    for variant in range(2):
        image = Image.new('RGB',(1000,740),'#f7f1e7')
        draw = ImageDraw.Draw(image)
        draw.rectangle((40,60,960,680),outline='#75634d',width=4)
        draw.text((300,95),'Colored objects',font=font(38),fill='black')
        if variant==0:
            draw.rectangle((590,280,770,460),fill='#d52525')
            draw.ellipse((170,280,350,460),fill='#1853ca')
            red,blue,position = 'square','circle','left'
        else:
            draw.polygon([(180,460),(360,460),(270,275)],fill='#d52525')
            draw.rectangle((600,280,780,460),fill='#1853ca')
            red,blue,position = 'triangle','square','right'
        for field,question,answer in [('red','What shape is red? Return only the shape name.',red),
            ('blue','What shape is blue? Return only the shape name.',blue),
            ('position','Is the blue object on the left or right? Return only left or right.',position)]:
            save(f'shapes-{variant}-{field}',image,question,answer,'counterfactual_spatial','shapes-'+field)
    for variant,heights in enumerate([(375,170,265),(180,295,405)]):
        image = Image.new('RGB',(1040,780),'#f5f8ff')
        draw = ImageDraw.Draw(image)
        draw.text((200,40),'Service tickets by team',font=font(40),fill='black')
        draw.line((125,160,125,610),fill='black',width=4)
        draw.line((125,610,930,610),fill='black',width=4)
        labels = ['Delta','Echo','Foxtrot']
        for x,height,label in zip([210,460,710],heights,labels):
            draw.rectangle((x,610-height,x+120,610),fill='#5c7ac3')
            draw.text((x-12,645),label,font=font(30),fill='black')
        save(f'chart-{variant}',image,'Which team has the tallest bar? Return only its name.',
             labels[heights.index(max(heights))],'counterfactual_chart','chart-tallest')
    report = {'scope':'fresh paired engineering checks; not a public benchmark',
        'training_excluded':True,'alignment_images_fenced':len(hashes),'unique_images':len(seen),
        'checks':len(examples),'artifact_bytes':budget.output,'artifact_limit':budget.output_limit,
        'examples':examples}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'}))


if __name__=='__main__':
    main()
