import pickle
import numpy as np

fname = 'CE_UNSAT.pkl'
with open(fname,'rb') as f:
    ctr_exmpls = pickle.load(f)

fname = 'relus.pkl'
with open(fname,'rb') as f:
    crrct_assgnmnt = pickle.load(f)

fname = 'SAT_found.pkl'
with open(fname,'rb') as f:
    sat = pickle.load(f)

violations_idx = []
for i,ce in enumerate(ctr_exmpls):
    sum = 0
    sz = len(ce)
    for idx in ce:
        sum += int(crrct_assgnmnt[idx])
    if(sum == sz):
        violations_idx.append(i)

for idx,elem in enumerate(sat):
    match = True
    for i in range(len(elem)):
        if(elem[i] != crrct_assgnmnt[i]):
            match = False
    if(match):
        print(idx,'Match')

        


# print(violations_idx)
# print(len(ctr_exmpls))
