import os, sys
import numpy as np
import pandas as pd



def find_unconverged_cells(listfile):
    ReadCellsFlg = 0
    with open(listfile, 'r') as file:
        data = file.readline()


        if " Residual-Control   Outer-Iter.   Inner-Iter." in data:
            ReadCellsFlg = 1

        if " NO OUTPUT CONTROL FOR STRESS PERIOD" in data:
            ReadCellsFlg = 0
            parts = data.strip().split()
            sp = int(parts[6])
            ts = int(parts[9])





if __name__ == "__main__":
    listfile = r"D:\Yucaipa_work\Yuc4\yucaipa.list"
    find_unconverged_cells(listfile)



