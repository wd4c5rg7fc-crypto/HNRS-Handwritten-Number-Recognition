"""Read-only experiment: adaptive CNN shifts vs working V4 baseline.
Run from the HNRS_Combined_Project root. No project files are modified.
"""
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MEMBER = ROOT / 'member_sources' / 'member_2'
sys.path.insert(0, str(MEMBER))
from Segmentation import segment_digits
from division_grouping import merge_division_parts
from connected_digit_splitter import split_connected_digits
from operator_recognition import recognise_operator

EXPECTED = {
    1:'12-5+8',2:'15+23',3:'8+3*4',4:'26-3*(17-4)',5:'18-9',
    6:'36/(3+3)',7:'24/6',8:'7*(9-4)',9:'7+5',10:'(8+3)*4',
    11:'6*4',12:'50-27',13:'15+(6*3)-4',14:'100/(5*4)',
    15:'4*(16+9)',16:'9-(2+3)',17:'(10+8)/(7-3)',18:'72/8+6',
    19:'9-(2+3)',20:'48/6+7*2',21:'35-18',22:'15+3',23:'2*3'
}

def preprocess(character):
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None: return None
    x,y,w,h = cv2.boundingRect(points)
    scale = 20.0/max(w,h)
    nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
    resized = cv2.resize(binary[y:y+h,x:x+w],(nw,nh),interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28,28),dtype=np.uint8)
    ox,oy = (28-nw)//2,(28-nh)//2
    canvas[oy:oy+nh,ox:ox+nw] = resized
    m = cv2.moments(canvas)
    if m['m00']:
        cx,cy = m['m10']/m['m00'],m['m01']/m['m00']
        dx,dy = round(13.5-cx)+1,round(13.5-cy)
        canvas = cv2.warpAffine(canvas,np.float32([[1,0,dx],[0,1,dy]]),(28,28))
    return canvas.astype(np.float32)/255.0

def shifted(image, dx):
    return cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,0]]),(28,28),
                          flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def cnn_predict(image,model):
    return model.predict(image.reshape(1,28,28,1),verbose=0)[0]

def recognise(chars,flags,model,adaptive=False,threshold=.80):
    symbols=[]; decisions=[]
    for idx,(char,from_split) in enumerate(zip(chars,flags)):
        symbol = None if from_split else recognise_operator(char)
        if symbol is not None:
            symbols.append(symbol)
            continue
        processed = preprocess(char)
        if processed is None:
            raise ValueError(f'Empty component at position {idx}')
        probs = cnn_predict(processed,model)
        base_digit = int(np.argmax(probs))
        best_digit,best_conf,best_shift = base_digit,float(probs[base_digit]),0
        if adaptive and best_conf < threshold:
            for dx in (-1,-2):
                p = cnn_predict(shifted(processed,dx),model)
                d = int(np.argmax(p)); conf=float(p[d])
                if conf > best_conf:
                    best_digit,best_conf,best_shift=d,conf,dx
        if adaptive and best_shift:
            decisions.append(f'pos{idx}:{base_digit}->{best_digit} shift{best_shift}')
        symbols.append(str(best_digit if adaptive else base_digit))
    return ''.join(symbols),'; '.join(decisions)

def main():
    model_path=ROOT/'models'/'best_cnn_model.keras'
    image_dir=MEMBER/'testimages'
    for path in (model_path,image_dir):
        if not path.exists(): raise FileNotFoundError(path)
    model=load_model(str(model_path))
    paths=sorted((p for p in image_dir.glob('testcase*.*') if p.suffix.lower() in ('.jpg','.jpeg','.png')),
                 key=lambda p:int(''.join(c for c in p.stem if c.isdigit()) or 0))
    print(f'Found {len(paths)} images. Adaptive shift experiment (threshold 80%).')
    print(f'{"Test":<6}{"Expected":<22}{"Baseline":<22}{"Adaptive":<22}Change')
    rows=[]
    for path in paths:
        num=int(''.join(c for c in path.stem if c.isdigit()))
        try:
            image=cv2.imread(str(path))
            if image is None: raise ValueError('Unreadable image')
            chars,boxes=segment_digits(image,method='improved',group_division=True)
            chars,boxes=merge_division_parts(chars,boxes)
            chars,boxes,flags=split_connected_digits(chars,boxes,return_split_flags=True)
            baseline,_=recognise(chars,flags,model,False)
            adaptive,decisions=recognise(chars,flags,model,True)
            exp=EXPECTED.get(num)
            rows.append(dict(testcase=num,expected=exp or '',baseline=baseline,adaptive=adaptive,
                             baseline_correct='' if exp is None else str(baseline==exp),
                             adaptive_correct='' if exp is None else str(adaptive==exp),
                             changes=decisions,error=''))
            print(f'{num:<6}{(exp or "unknown"):<22}{baseline:<22}{adaptive:<22}{decisions}')
        except Exception as exc:
            rows.append(dict(testcase=num,expected=EXPECTED.get(num,''),baseline='',adaptive='',
                             baseline_correct='',adaptive_correct='',changes='',error=str(exc)))
            print(f'{num:<6}ERROR: {exc}')
    labelled=[r for r in rows if r['expected'] and not r['error']]
    b=sum(r['baseline_correct']=='True' for r in labelled)
    a=sum(r['adaptive_correct']=='True' for r in labelled)
    gains=[r['testcase'] for r in labelled if r['baseline_correct']=='False' and r['adaptive_correct']=='True']
    regress=[r['testcase'] for r in labelled if r['baseline_correct']=='True' and r['adaptive_correct']=='False']
    print(f'\nBaseline: {b}/{len(labelled)}; Adaptive: {a}/{len(labelled)}')
    print('Newly correct:',gains,'| Regressions:',regress)
    out=ROOT/'test_results'/'adaptive_digit_shift_v4.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['testcase'])
        writer.writeheader();writer.writerows(rows)
    print('CSV:',out)

if __name__=='__main__': main()
