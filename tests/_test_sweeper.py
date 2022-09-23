import os, sys
import sweeper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

slave_dir = r".\template_workerB"
pyexe = sys.executable
cmd = pyexe + " bad_land_highK.py"

N = 15

argu_value = np.linspace(start= -3  , stop= 0, num=N)
#argu_value = np.logspace(-5, 0, N)

sweeper.start_slaves(slave_dir,"pestpp",'pstname', N, slave_root= r".\workers_poolB", port=5, run_cmd=cmd,
                     args = argu_value, output_file= None , output_folder= None, cleanup=False)
if 0:
    alldf = []
    for i in range(15):
        folder = r"C:\work\san_antonio\results"
        fn = os.path.join(folder, '{}_Exp1.csv'.format(i))
        df = pd.read_csv(fn)
        alldf.append(df)
    alldf = pd.concat(alldf)
    alldf['par'] = argu_value
    alldf = alldf.reset_index()

    alldf.to_csv(r".\results\jh_coeff.csv")