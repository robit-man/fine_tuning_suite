"""Fresh image-grounded checks excluded from every alignment update.

These are small engineering checks, not benchmark or best-in-class evidence.
Different hidden identifiers and visual positions rule out memorized answers.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'independent-checks'
OUT.mkdir(exist_ok=True)
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font=lambda size:ImageFont.truetype(str(FONT),size)
records=[]


def save(name,image,question,answer,task):
    path=OUT/(name+'.png')
    image.save(path)
    records.append({'id':name,'image':str(path.relative_to(ROOT)),'question':question,
        'answer':answer,'task':task,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'split':'independent_evaluation_only'})


image=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(image)
draw.text((70,60),'NORTHBRIDGE SUPPLIES',font=font(48),fill='black')
draw.text((70,155),'INVOICE KBR-5937',font=font(42),fill='black')
draw.text((70,245),'Issued: 2026-10-06',font=font(28),fill='black')
for y,text in [(340,'Item                     Quantity           Amount'),(415,'Desk lamp                    3               $75.00'),(480,'Cable organizer              6               $24.50'),(580,'Subtotal                                  $99.50'),(635,'Shipping                                  $12.75'),(710,'TOTAL                                    $112.25')]:
    draw.text((70,y),text,font=font(28),fill='black')
save('invoice-id',image,'What is the invoice identifier? Return only the identifier.','KBR-5937','document_ocr')
save('invoice-total',image,'What is the total including shipping? Return only the dollar amount.','$112.25','document_ocr')

image=Image.new('RGB',(1100,700),'white');draw=ImageDraw.Draw(image)
draw.text((55,50),'Storage inventory',font=font(44),fill='black')
xs=[55,460,750];ys=[150,260,370,480]
rows=[['Item','Count','Location'],['Osprey','47','West'],['Kestrel','19','North'],['Heron','83','East']]
for y,row in zip(ys,rows):
    for x,text in zip(xs,row):draw.text((x,y),text,font=font(34),fill='black')
    draw.line((45,y+68,1030,y+68),fill='#8899aa',width=2)
save('table-lookup',image,'How many Osprey items are in the inventory? Return only the number.','47','table_reading')

image=Image.new('RGB',(1100,700),'white');draw=ImageDraw.Draw(image)
draw.text((55,60),'Service access notice',font=font(40),fill='black')
paragraph=['Please keep this notice for your scheduled maintenance visit.',
    'The technician must quote the access code N7Q4-Z8M2.',
    'The code expires after the visit. Do not reuse an expired code.']
for i,text in enumerate(paragraph):draw.text((55,175+i*70),text,font=font(25),fill='black')
save('paragraph-code',image,'Read the access code from the notice. Return only the code.','N7Q4-Z8M2','document_ocr')

image=Image.new('RGB',(1100,720),'#edf1f7');draw=ImageDraw.Draw(image)
draw.rectangle((0,0,1100,95),fill='#17243d');draw.text((40,22),'Workspace settings',font=font(40),fill='white')
draw.text((55,145),'Cloud synchronization',font=font(36),fill='black')
draw.text((55,230),'Last sync: 09:42',font=font(28),fill='#334455')
draw.rounded_rectangle((650,530,1030,635),radius=14,fill='#1666ba')
draw.text((693,555),'Synchronize',font=font(36),fill='white')
draw.rounded_rectangle((55,530,360,635),radius=14,fill='#cccccc')
draw.text((130,555),'Cancel',font=font(36),fill='black')
save('ui-button',image,'What does the blue button say? Return only its label.','Synchronize','screenshot_ocr')

image=Image.new('RGB',(900,700),'white');draw=ImageDraw.Draw(image)
draw.text((60,35),'Units sold',font=font(42),fill='black')
draw.line((100,120,100,560),fill='black',width=4);draw.line((100,560,840,560),fill='black',width=4)
for x,height,label in [(180,190,'Alpha'),(400,365,'Bravo'),(620,275,'Charlie')]:
    draw.rectangle((x,560-height,x+120,560),fill='#367bb8')
    draw.text((x-4,590),label,font=font(27),fill='black')
save('bar-chart',image,'Which category has the tallest bar? Return only its name.','Bravo','chart_understanding')

image=Image.new('RGB',(1000,700),'white');draw=ImageDraw.Draw(image)
draw.ellipse((100,220,280,400),fill='#da2626')
draw.rectangle((620,220,800,400),fill='#1d55ca')
draw.text((90,475),'Left object',font=font(32),fill='black')
draw.text((585,475),'Right object',font=font(32),fill='black')
save('spatial-shape',image,'What shape is red? Return only the shape name.','circle','spatial_reasoning')
save('spatial-position',image,'Is the blue object on the left or right? Return only left or right.','right','spatial_reasoning')

(OUT/'manifest.json').write_text(json.dumps({'scope':'fresh evaluation-only engineering images; not a published benchmark',
    'training_excluded':True,'examples':records},indent=2)+'\n')
print(json.dumps({'independent_images':len(records),'output':str(OUT)}))
