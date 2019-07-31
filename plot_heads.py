import os, sys
import matplotlib.pyplot as plt
import flopy
import general_util
import hob_util
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
import datetime
import calendar

def xldate_to_datetime(start_date, tdelta):
    temp = start_date
    delta = datetime.timedelta(days=int(tdelta))
    return temp + delta

def plot_all_heads(mfname, start_date, end_date):
    """

    :param mfname: modflow name file
    :return:
    """

    # Read modflow basic information
    mf = flopy.modflow.Modflow.load(mfname, load_only= ['DIS', 'BAS6'])

    # get all files
    mf_files = general_util.get_mf_files(mfname)

    # get needed files
    hob_in_file = mf_files['HOB'][1]
    hds_file = mf_files['hds'][1]
    for pkg in mf_files.keys():
        if '.HOB.out' in mf_files[pkg][1]:
            hob_out_file = mf_files[pkg][1]

    # Generate a dataframe for observations
    hob_df = hob_util.in_hob_to_df(mfname)

    #Load hds
    hds_obj = flopy.utils.HeadFile(hds_file)

    #get well names
    well_names = []
    well_ids = []
    for well_name in hob_df['name'].values:
        try:
            well_name, timid = well_name.split('.')
        except:
            timid = 0
        well_names.append(well_name)
        well_ids.append(int(well_name.split('_')[1]))
    hob_df['id']  = well_ids

    # plot
    well_groups = hob_df.groupby('id')
    with PdfPages('mod_3.pdf') as pdf:
        for well in well_groups:
            print(well[0])
            col = well[1]['col'].values[0] - 1
            row = well[1]['row'].values[0] - 1

            if well[1]['mlay'].values[0] is None:
                layers = well[1]['layer'].values[0]
            else:
                layers = well[1]['mlay'].values[0]


            #todo: deal with not multi-data
            for i, lay in enumerate(layers.keys()):
                head_ts = hds_obj.get_ts((lay, row, col))
                if i == 0:
                    head = head_ts[:,1] * layers[lay]
                else:
                    head = head + head_ts[:,1] * layers[lay]
            dates = []
            for tdelta in head_ts[:, 0]:
                curr_date = xldate_to_datetime(start_date, tdelta)
                if calendar.isleap(curr_date.year):
                    curr_date = curr_date.year + curr_date.month/12.0 + curr_date.day/366.0
                else:
                    curr_date = curr_date.year + curr_date.month / 12.0 + curr_date.day / 365.0
                dates.append(curr_date)

            plt.plot(dates, head)

            # add obs
            dates = []
            head = well[1]['head'].values
            for tdelta in well[1]['totim']:
                curr_date = xldate_to_datetime(start_date, tdelta)
                if calendar.isleap(curr_date.year):
                    curr_date = curr_date.year + curr_date.month/12.0 + curr_date.day/366.0
                else:
                    curr_date = curr_date.year + curr_date.month / 12.0 + curr_date.day / 365.0
                dates.append(curr_date)
            plt.scatter(dates, head, edgecolors= 'r')
            plt.ylim([max(head) + 100, min(head)-100])
            plt.title(well[0])
            pdf.savefig()
            plt.close()
            x = 1







            pass



    pass