import pickle
import numpy as np

fname = 'CE_UNSAT.pkl'
with open(fname,'rb') as f:
    ctr_exmpls = pickle.load(f)

fname = 'relus.pkl'
with open(fname,'rb') as f:
    crrct_assgnmnt = pickle.load(f)



violations_idx = []
for i,ce in enumerate(ctr_exmpls):
    sum = 0
    sz = len(ce)
    for idx in ce:
        sum += int(crrct_assgnmnt[idx])
    if(sum == sz):
        violations_idx.append(i)
print(violations_idx)
