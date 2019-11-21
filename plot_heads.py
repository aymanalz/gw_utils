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
import pandas as pd

def xldate_to_datetime(start_date, tdelta):
    temp = start_date
    delta = datetime.timedelta(days=int(tdelta))
    return temp + delta

def add_simulated_all_layers(ax, row, col, layers, hds_obj, start_date):
    # only get the date
    head_ts = hds_obj.get_ts((0, row, col))
    dates = []
    for tdelta in head_ts[:, 0]:
        curr_date = xldate_to_datetime(start_date, tdelta)
        if calendar.isleap(curr_date.year):
            curr_date = curr_date.year + curr_date.month / 12.0 + curr_date.day / 366.0
        else:
            curr_date = curr_date.year + curr_date.month / 12.0 + curr_date.day / 365.0
        dates.append(curr_date)

    for ilay in range(hds_obj.model.nlay):
        head_ts = hds_obj.get_ts((ilay, row, col))
        ib  = hds_obj.model.bas6.ibound.array[ilay, row, col]
        head = head_ts[:, 1]
        if ib == 1:
            label = "Layer {}".format(ilay)
            ax.plot(dates, head, label= label, linewidth=0.8, alpha = 1.0, zorder=1)






def add_simulated_multi_layer_heads(ax, row, col, layers, hds_obj, start_date):
    """

    Parameters
    ----------
    ax
    row
    col
    layers
    hds_obj
    start_date

    Returns
    -------

    """
    for i, lay in enumerate(layers.keys()):
        head_ts = hds_obj.get_ts((lay, row, col))
        if i == 0:
            head = head_ts[:, 1] * layers[lay]
        else:
            head = head + head_ts[:, 1] * layers[lay]
    dates = []
    for tdelta in head_ts[:, 0]:
        curr_date = xldate_to_datetime(start_date, tdelta)
        if calendar.isleap(curr_date.year):
            curr_date = curr_date.year + curr_date.month / 12.0 + curr_date.day / 366.0
        else:
            curr_date = curr_date.year + curr_date.month / 12.0 + curr_date.day / 365.0
        dates.append(curr_date)

    ax.plot(dates, head, label = 'Simulated Water table', linewidth = 2.0)


def plot_all_heads(mfname, start_date, end_date, pdf_file = 'all_water_levels4.pdf',
                   add_water_table = True, x_limit = [], obs_name_file = None):
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

    # read_hob_out
    hobout_df = None
    for file in mf_files.keys():
        fn = mf_files[file][1]
        basename = os.path.basename(fn)
        if ".hob.out" in basename:
            hobout_df = pd.read_csv(fn, delim_whitespace=True)

    if not(hobout_df is None):
        wellnms = []
        timids = []
        for irec, rec in hobout_df.iterrows():
            well_name = rec['OBSERVATION NAME']
            if '.' in well_name:
                well_name, timid = well_name.split('.')
            else:
                well_name = well_name
                timid = 0
            wellnms.append(well_name)
            timids.append(timid)
        hobout_df['timid'] = timids
        hobout_df['Wellname'] = wellnms

    #Load hds
    hds_obj = flopy.utils.HeadFile(hds_file)
    hds_obj.model = mf

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

    # get local names
    local_names = pd.read_csv(obs_name_file)


    well_groups = hob_df.groupby('id')
    with PdfPages(pdf_file) as pdf:
        for well in well_groups:
            fig, ax = plt.subplots(1)


            #plt.style.use('bmh')
            plt.style.use('seaborn-deep')
            #ax.set_facecolor('lightgreen')
            well_base_name = well[1]['Basename'].values[0]
            print(well[0])
            is_multilayers = False
            col = well[1]['col'].values[0] - 1
            row = well[1]['row'].values[0] - 1

            if well[1]['mlay'].values[0] is None:
                layers = well[1]['layer'].values[0]
            else:
                is_multilayers = True
                layers = well[1]['mlay'].values[0]


            #todo: deal with not multi-data
            if 0:
                add_simulated_multi_layer_heads(ax, row, col, layers, hds_obj, start_date)

            # add all active layers
            add_simulated_all_layers(ax, row, col, layers, hds_obj, start_date)


            # add obs
            dates = []
            head1 = well[1]['head'].values
            for tdelta in well[1]['totim']:
                curr_date = xldate_to_datetime(start_date, tdelta)
                if calendar.isleap(curr_date.year):
                    curr_date = curr_date.year + curr_date.month/12.0 + curr_date.day/366.0
                else:
                    curr_date = curr_date.year + curr_date.month / 12.0 + curr_date.day / 365.0
                dates.append(curr_date)
            ax.scatter(dates, head1, marker='.', label = 'OBSERVED VALUE', facecolors='none', edgecolors='r', s = 14, zorder=2)

            # add hob simulated
            curr_hob_out = hobout_df[hobout_df['Wellname'] ==  well_base_name]
            head = curr_hob_out['SIMULATED EQUIVALENT'].values
            ax.scatter(dates, head, marker=".", label='SIMULATED EQUIVALENT',  facecolors='none', edgecolors='b', s=14, linewidths = 1, alpha = 0.8, zorder= 3)


            plt.ylim([min(head1)-100, max(head1) + 100])
            locName = local_names.loc[local_names['ID']==well[0], 'Name']
            try:
                tit = "Well ID:{}" \
                      "\n{}".format(well[0], locName.values[0])
            except:
                tit =  "Well ID:{}".format(well[0])
            plt.title(tit)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel(" Head (ft)", fontsize=14)
            plt.tight_layout()
            plt.legend()
            plt.xlim(x_limit)
            plt.grid(alpha=0.5, linestyle='--')
            pdf.savefig()

            plt.close()
            x = 1







            pass



    pass