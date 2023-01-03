import os, sys
import numpy as np
import pandas as pd
import pyemu

def write_input_files(pst,pst_path='.'):
    """write parameter values to model input files

       Args:
           pst (`pyemu.Pst`): a Pst instance
           pst_path (`str`): the path to where the control file and template
               files reside.  Default is '.'.

       Note:

           This function uses template files with the current parameter \
           values (stored in `pst.parameter_data.parval1`).

           This function uses multiprocessing - one process per template file

           This is a simple implementation of what PEST does.  It does not
           handle all the special cases, just a basic function...user beware

       """
    par = pst.parameter_data
    par.loc[:, "parval1_trans"] = (par.parval1 * par.scale) + par.offset
    pairs = np.array(list(zip(pst.template_files, pst.input_files)))
    num_tpl = len(pairs)
    chunk_len = 50
    num_chunk_floor = num_tpl // chunk_len
    main_chunks = pairs[:num_chunk_floor * chunk_len].reshape(
        [-1, chunk_len, 2]).tolist()  # the list of files broken down into chunks
    remainder = pairs[num_chunk_floor * chunk_len:].tolist()  # remaining files


    pyemu.pst_utils.write_to_template(pst.parameter_data.parval1_trans,
                                      os.path.join(pst_path,tpl_file),
                                      os.path.join(pst_path,in_file))

def get_best_pars(pstMaster):

    # find file with best parms
    best_parfile = os.path.splitext(pstMaster.filename)[0] + '.bpa'
    bpar_df = pd.read_csv(best_parfile, skiprows=1, delim_whitespace= True,
                names=['parnm', 'parval', 'offs', 'scale'],
                header=None)
    bpar_df = bpar_df.set_index(['parnm'])
    return bpar_df

def get_ipar(pstMaster, ipar):

    # get parameters of iteration ipar
    ipar_file = os.path.splitext(pstMaster.filename)[0] + '.ipar'

    bpar_df = pd.read_csv(ipar_file)
    bpar_df = bpar_df[bpar_df['iteration'] == ipar]
    del (bpar_df['iteration'])
    bpar_df = bpar_df.T
    bpar_df['parnm'] = bpar_df.index.values
    bpar_df['parval'] = bpar_df[ipar].values
    bpar_df['offs'] = 0
    bpar_df['scale'] = 1
    return bpar_df

def run_model_using_best_par(Masterfile, Slavefile ):



    pstMaster = pyemu.Pst(Masterfile)
    pstSlave = pyemu.Pst(Slavefile)

    bpar = get_best_pars(pstMaster)

    par = pstSlave.parameter_data
    par['parval1'] = bpar['parval']


    par.loc[:, "parval1_trans"] = (par.parval1 * par.scale) + par.offset
    pairs = []
    for irow, row_record in pstSlave.model_input_data.iterrows():
        pp = (row_record['pest_file'], row_record['model_file'])
        pairs.append(pp)

    pairs = np.array(pairs)
    num_tpl = len(pairs)
    slaveFolder = os.path.dirname(pstSlave.filename)
    for tp_in in pairs:
        pyemu.pst_utils.write_to_template(parvals = par["parval1_trans"],
                                          tpl_file = os.path.join(slaveFolder, tp_in[0]),
                                          in_file = os.path.join(slaveFolder, tp_in[1]))

def run_model_using_ipar(Masterfile, Slavefile, ipar):



    pstMaster = pyemu.Pst(Masterfile)
    pstSlave = pyemu.Pst(Slavefile)

    bpar = get_ipar(pstMaster, ipar)

    par = pstSlave.parameter_data
    par['parval1'] = bpar['parval']


    par.loc[:, "parval1_trans"] = (par.parval1 * par.scale) + par.offset
    pairs = []
    for irow, row_record in pstSlave.model_input_data.iterrows():
        pp = (row_record['pest_file'], row_record['model_file'])
        pairs.append(pp)

    pairs = np.array(pairs)
    num_tpl = len(pairs)
    slaveFolder = os.path.dirname(pstSlave.filename)
    for tp_in in pairs:
        pyemu.pst_utils.write_to_template(parvals = par["parval1_trans"],
                                          tpl_file = os.path.join(slaveFolder, tp_in[0]),
                                          in_file = os.path.join(slaveFolder, tp_in[1]))

def update_pst_using_best_run(Masterfile, Slavefile, new_pst_file ):


    pstMaster = pyemu.Pst(Masterfile)
    pstSlave = pyemu.Pst(Slavefile)

    bpar = get_best_pars(pstMaster)

    par = pstSlave.parameter_data
    par['parval1'] = bpar['parval']

    pstSlave.parameter_data = par

    pstSlave.write(new_filename=new_pst_file)

def run_model_using_realization_id(Masterfile, Slavefile, param_ensemble, realization = 1):

    pass






if __name__ == "__main__":
    Masterfile = r"D:\Models\RussianRiver\ayman\RR_ies\model_main\GSFLOW\worker_dir_ies\pest\tr_mf.pst"
    Slavefile = r"D:\Models\RussianRiver\ayman\RR_ies\model\GSFLOW\worker_dir_ies\pest\tr_mf.pst"
    ens_file = r""
    run_model_using_realization_id(Masterfile=Masterfile,
                                   Slavefile=Slavefile,
                                   param_ensemble=ens_file)
    # run_model_using_best_par(Masterfile = Masterfile,
    #                          Slavefile = Slavefile )



