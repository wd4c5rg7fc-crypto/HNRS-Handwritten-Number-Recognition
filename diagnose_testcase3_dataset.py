"""Inspect operator/digit shape measurements across HNRS test images.
Read-only: saves a CSV in test_results, does not modify recognition code.
Run from project root: python diagnose_testcase3_dataset.py
"""
import csv
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'member_sources' / 'member_2'))
from Segmentation import segment_digits
from division_grouping import merge_division_parts
from connected_digit_splitter import split_connected_digits
from operator_recognition import recognise_operator, _asterisk, _has_large_enclosed_loop

EXPECTED = {1:'12-5+8',2:'15+23',3:'8+3*4',4:'26-3*(17-4)',5:'18-9',6:'36/(3+3)',7:'24/6',8:'7*(9-4)',9:'7+5',10:'(8+3)*4',11:'6*4',12:'50-27',13:'15+(6*3)-4',14:'100/(5*4)',15:'4*(16+9)',16:'9-(2+3)',17:'(10+8)/(7-3)',18:'72/8+6',19:'9-(2+3)',20:'48/6+7*2',21:'35-18',22:'15+3',23:'2*3'}

def metrics(character):
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None: return None
    x,y,w,h = cv2.boundingRect(points)
    crop = binary[y:y+h,x:x+w]
    resized = cv2.resize(crop, (40,40), interpolation=cv2.INTER_NEAREST)>0
    corners = [resized[:10,:10],resized[:10,-10:],resized[-10:,:10],resized[-10:,-10:]]
    n,_,_,_ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    return dict(ratio=round(w/h,3), components=n-1,
        horizontal=round(float(np.mean(resized[17:23,:])),3),
        vertical=round(float(np.mean(resized[:,17:23])),3),
        corners=round(float(np.mean([np.mean(a) for a in corners])),3),
        diagonal1=round(float(np.mean(np.diag(resized))),3),
        diagonal2=round(float(np.mean(np.diag(np.fliplr(resized)))),3),
        strongest_column=round(float(np.max(np.mean(resized,axis=0))),3),
        asterisk=_asterisk(crop), loop=_has_large_enclosed_loop(crop))

def main():
    images = ROOT/'member_sources'/'member_2'/'testimages'
    if not images.is_dir(): raise FileNotFoundError(images)
    rows=[]
    for path in sorted(images.glob('testcase*.*'),key=lambda p:int(''.join(c for c in p.stem if c.isdigit()) or 0)):
        if path.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
        number=int(''.join(c for c in path.stem if c.isdigit()))
        image=cv2.imread(str(path))
        if image is None:continue
        chars,boxes=segment_digits(image,method='improved',group_division=True)
        chars,boxes=merge_division_parts(chars,boxes)
        chars,boxes,flags=split_connected_digits(chars,boxes,return_split_flags=True)
        expected=EXPECTED.get(number,'')
        aligned=len(expected)==len(chars)
        for i,(char,flag) in enumerate(zip(chars,flags)):
            data=metrics(char)
            if data is None:continue
            row=dict(testcase=number,position=i+1,expected=expected[i] if aligned else '?',
                aligned=aligned,recognised= 'CNN_ONLY' if flag else (recognise_operator(char) or 'CNN'),split_piece=flag,**data)
            rows.append(row)
            if number in (3,8,11,22,23):
                print(f"Test {number:2d} pos {i+1:2d}: expected={row['expected']:>2} recognised={row['recognised']:<8} ratio={data['ratio']:.3f} horiz={data['horizontal']:.3f} vert={data['vertical']:.3f} corners={data['corners']:.3f} diag1={data['diagonal1']:.3f} diag2={data['diagonal2']:.3f} components={data['components']}")
    out=ROOT/'test_results'/'operator_shape_dataset.csv'
    out.parent.mkdir(exist_ok=True,parents=True)
    if rows:
        with out.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(f'\nSaved {len(rows)} character measurements to: {out}')
    print('NOTE: expected=? means segmentation count differs from expected text; do not treat as labelled.')

if __name__=='__main__':main()
