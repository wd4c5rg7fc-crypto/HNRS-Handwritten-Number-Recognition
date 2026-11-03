"""Small rule-level checks; does not require trained models or test images."""
import numpy as np
from unittest.mock import patch
import HNRS_app_v72_parenthesis_pair as app

def check(legacy, candidate, opening, closing, expected, split=None):
    n=len(legacy)
    probs=np.zeros((n,16),dtype=float)
    for i,c in enumerate(candidate):
        probs[i,app.MATH_CLASSES.index(c)]=0.99
    probs[0,app.MATH_CLASSES.index('(')]=opening
    probs[6,app.MATH_CLASSES.index(')')]=closing
    with patch.object(app,'open_paren_geometry_v46',return_value=False), \
         patch.object(app,'narrow_open_paren_geometry_v52',return_value=False), \
         patch.object(app,'closing_paren_geometry_v47',return_value=False):
        result,notes=app.fuse_valid_parenthesis_disagreements_v72(
            legacy,list(candidate),[np.zeros((40,20),dtype=np.uint8)]*n,
            probs,split or [False]*n,[None]*n)
    assert result==expected,(result,expected,notes)
    print('PASS:',legacy,'=>',result,notes[0])

check('636+247/6','(36+24)/6',0.891,0.993,'(36+24)/6')
check('636+247/6','(36+24)/6',0.891,0.60,'636+247/6')
check('636+247/6','(36+24)/6',0.891,0.993,'636+247/6',split=[False]*6+[True]+[False]*2)
check('636+247/6','(36+24)/8',0.891,0.993,'636+247/6')
print('All V7.2 rule-level checks passed.')
