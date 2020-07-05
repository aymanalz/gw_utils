import os, sys

import numpy as np

sys.path.insert(0,r"D:\codes")
print(sys.path)

import gw_utils
import flopy

mf_tr = r"C:\work\Slave\model_1_no_run\model\tran_mf3d\yucaipa.nam"
mf_ss_ws = r"D:\Yucaipa_work\yuc_ss"

obs_info = np.load(r"D:\Yucaipa_work\obs_multi_layers_2_1_20.npy", allow_pickle= True).all()

mf_tr = flopy.modflow.Modflow.load(mf_tr)

mf_ss = gw_utils.tr2ss.main_ss_to_tarn(mf_tran=mf_tr, ss_folder = mf_ss_ws,
                               ss_name = 'yucaipa_ss', stress_start_end = [277,815],
                               obs_dict = obs_info['hoblist'])

xx = 1