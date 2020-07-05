import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import numpy as np

#df2 = pd.read_csv(r"D:\Yucaipa_work\golf_course_3.csv")
fn1 = r"D:\Yucaipa_work\kh_yuc_creek.csv"
#fn2 = r"D:\Yucaipa_work\golf_course_2.csv"
#fn3 = r"D:\Yucaipa_work\Park_Ponds.csv"

mffn = r"C:\work\Slave\exp6\yucaipa.nam"
df = pd.read_csv(fn1)
#df2 = pd.read_csv(fn2)
#df3 = pd.read_csv(fn3)
#df = pd.concat([df1,df2,df3])
rrows = df['HRU_ROW'] - 1
ccols = df['HRU_COL'] - 1

mf = flopy.modflow.Modflow.load(os.path.basename(mffn),model_ws= os.path.dirname(mffn), load_only=['DIS', 'BAS6', 'UPW', 'HFB6'])

rech_ = []

for rr_cc in zip(rrows.values, ccols.values):
    rech_.append([0, rr_cc[0], rr_cc[1], 0.6*20172.6*9/31 ]) # 0.15*150*3.28*150*3.28
stress_period_data = {}

for sp in range(707,815): #add recharge from 2005
    stress_period_data[sp] = rech_

well = flopy.modflow.mfwel.ModflowWel(mf, ipakcb=1, stress_period_data=stress_period_data, dtype=None, extension='wel')

hk = mf.upw.hk.array.copy()
for i in range(4):
    vals = hk[i,rrows, ccols]
    vals = vals * 2.0
    hk[i, rrows, ccols] = vals
mf.upw.hk = hk

sy = mf.upw.sy.array
sy[sy>0.18] = 0.18
mf.upw.sy = sy
xx = 1



pass