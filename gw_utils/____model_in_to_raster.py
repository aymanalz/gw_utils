import sys, os
sys.path.insert(0,r"D:\Models\Yucaipa\codes")

import gw_utils
import datetime
import numpy as np
import flopy

mfname = r"D:\Models\Yucaipa\Yuc\pilot_KF_and_pest\model\model\tran_mf3d\yucaipa.nam"
mf = flopy.modflow.Modflow.load(mfname, model_ws = os.path.dirname(mfname) ,   load_only=['DIS', 'BAS6', 'UPW'])

k1 = mf.upw.hk.array[1,:,:]
k2 = mf.upw.hk.array[2,:,:]
k3 = mf.upw.hk.array[3,:,:]

for i in range(3):
    kk = np.log10(mf.upw.hk.array[i+1, :, :])
    np.savetxt('K_{}.txt'.format(i+1), kk)

for i in range(3):
    ss = np.log10(mf.upw.ss.array[i+1, :, :])
    np.savetxt('Ss_{}.txt'.format(i+1), ss)

for i in range(3):
    sy = (mf.upw.sy.array[i+1, :, :])
    np.savetxt('Sy_{}.txt'.format(i+1), sy)
xxx= 1