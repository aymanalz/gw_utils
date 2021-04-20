import configparser
import copy
import os
import shutil
import sys

"""
    - General functions to produce and run all files needed to run a job on HTcondor
    - Input is a configuration file, that contains information to run the job.
    - A folder "Home Folder" will be generated, which contains all data that is
        needed to run an HTconodr job. In particular:
        --Home_folder
        |-- condor submission file.
        |-- worker batch file to unzip zipped worker folder, and cd worker folder, and run a command.
        |-- unzip.exe
        |-- zipped worker folder
            |-- command (maybe bat file) to run the model on the remote computer.
            |-- python scripts, modflow/gsflow/pest/...etc
            
    - The model folder consists of a  "Python Environment", model data, and python lib.
    - The worker consists of a command that run an execuatble and all other data needed for this run to succeed.
    - Usually the worker is zipped to speed up files transfere and when finished is unzipped.
    - In the "Condor Info" section, all condor data is listed.
     
"""


def config_to_class(config):
    """
    Convert a config object to a class

    """

    class Config_class:
        def __init__(self):
            pass

    config_main = Config_class()
    for section in config.sections():
        options = config[section].keys()
        config_secondary = Config_class()
        for option in options:
            value = config[section][option]
            setattr(config_secondary, option, value)
        setattr(config_main, section, config_secondary)
    return config_main


def class_to_config(cconfig):
    config = configparser.ConfigParser()
    sections = cconfig.__dict__.keys()
    for section in sections:
        section_class = getattr(cconfig, section)
        options = section_class.__dict__.keys()

        for option in options:
            value = getattr(section_class, option)
            if not (section in config.sections()):
                config.add_section(section)
            config.set(section=section, option=option, value=value)
    return config


def initialize_config_file(workspace=r".", config_file_name="pest_work.init"):
    """

    Parameters
    ----------
    config_file_name

    Returns
    -------

    """
    if not (os.path.isdir(workspace)):
        os.mkdir(workspace)

    config = configparser.ConfigParser()

    config['General'] = {'config_file_name': str(config_file_name),
                         'home_folder': workspace,
                         }

    config['Master_Node_Info'] = {'computer_address': "",
                                  'port_number': 4075}

    config['Worker_Info'] = {'worker_dir': 'model', 'dataset_dir': "",
                             'python_env': "",
                             'model_run_command': "",

                             'other_files': '',
                             'other_folders': '',
                             'worker_batch_file': "run_worker.bat"
                             }

    config['PEST_Info'] = {'master_folder': "",
                           'pst_file': "",
                           'pest_exe': ""}

    submit_info = """
    notification = never
    universe = vanilla
    log = log/Yuc.$(Cluster).log
    output = log\worker_$(Cluster)_$(Process).out
    error = log\worker_$(Cluster)_$(Process).err
    stream_output = true
    stream_error = true
    executable = worker_pest.bat
    requirements = (PoolName == "1601") || (PoolName == "1401")
    arguments =  igswcawwas1701 4075 yuc_prms.pst
    request_memory =2500
    request_disk = 10000 
    rank = MIPS
    should_transfer_files = YES
    transfer_input_files = model.zip, unzip.exe, worker_pest.bat
    queue 8
    """
    conda_dict = {'submit_file': ""}
    submit_info = submit_info.split('\n')
    for line in submit_info:
        if "=" in line:
            if "==" in line:
                line = line.replace("==", "@@")
                parts = line.split("=")
                parts[1] = parts[1].replace("@@", "==")
            else:
                parts = line.split("=")

            conda_dict[parts[0].strip()] = parts[1].strip()

        if "queue" in line:
            parts = line.split()
            conda_dict[parts[0].strip()] = parts[1].strip()

    config['Condor_Info'] = conda_dict

    with open(os.path.join(workspace, config_file_name), 'w') as configfile:
        config.write(configfile)

    config_class = config_to_class(config)
    return config_class


def write_config_file(config, fname=None):
    if not (isinstance(config, configparser.ConfigParser)):
        _config = class_to_config(config)

    if not (fname is None):
        _config.set('General', 'config_file_name', os.path.basename(fname))
        _config.set('General', 'home_folder', os.path.dirname(os.path.abspath(fname)))
    else:
        fn = _config.get('General', 'config_file_name')
        folder = _config.get('General', 'home_folder')
        fname = os.path.join(folder, fn)

    with open(fname, 'w') as configfile:
        _config.write(configfile)


def read_config_file(fname):
    config = configparser.ConfigParser()
    config.read(fname)
    return config


def write_work_bat_file(_config):
    if isinstance(_config, configparser.ConfigParser):
        config = config_to_class(_config)
    else:
        config = copy.deepcopy(_config)

    worker_folder = os.path.join(config.General.home_folder, config.Worker_Info.worker_dir)
    bat_fname = config.Worker_Info.worker_batch_file
    zipped_worker_dir = config.Worker_Info.worker_dir
    content = """dir
    ipconfig
    :: catch the arguments
    set masterIP=%1
    set masterPORT=%2
    set casename=%3
    
    :: unzipping python and putting in the path
    :: unzipping model
    unzip {}.zip >nul
    cd {}
    :: starting remote exe
    {}   
    """.format(zipped_worker_dir, zipped_worker_dir, config.Worker_Info.model_run_command)

    fidw = open(os.path.join(config.General.home_folder, bat_fname), 'w')
    content = content.split("\n")
    for line in content:
        line = line.strip()
        line = "\n" + line
        fidw.write(line)
    cmd = config.Worker_Info.model_run_command
    cmd = "\n" + cmd
    fidw.write(line)
    fidw.close()


def assemble_worker(_config):
    """
    - This function collect files and folders needed to run a worker. Data needed are extracted from
    "Worker_Info" section in the config file.
    - Other files usually needed is unzip.exe and run_worker.bat
    - config: is a config class or config obj that is generated from a configparser object

    """
    if isinstance(_config, configparser.ConfigParser):
        config = config_to_class(_config)
    else:
        config = copy.deepcopy(_config)

    home_dir = config.General.home_folder
    worker_folder = os.path.join(config.General.home_folder, config.Worker_Info.worker_dir)
    worker_info = config.Worker_Info

    # dataset
    if os.path.isdir(worker_info.dataset_folder):
        src = worker_info.dataset_folder
        dst = os.path.join(home_dir, worker_folder, os.path.basename(src))
        print("... copying dataset folder {} ".format(src))
        shutil.copytree(src, dst)
    else:
        print("--- Model dataset, {}, is not a valid directory".format(worker_info.dataset_folder))

    # python env
    if os.path.isdir(worker_info.python_env):
        src = worker_info.python_env
        dst = os.path.join(home_dir, worker_folder, os.path.basename(src))
        print("... copying Python env folder {} ".format(src))
        shutil.copytree(src, dst)
    else:
        print("--- Python Environment, {}, is not a valid directory".format(worker_info.dataset_folder))

    # other_folders
    other_folders = worker_info.other_folders.strip().split(",")
    for folder in other_folders:
        if os.path.isdir(folder):
            src = folder
            dst = os.path.join(home_dir, worker_folder, os.path.basename(src))
            print("... copying other folders {} ".format(src))
            shutil.copytree(src, dst)
        else:
            print("--- Other folder, {}, is not a valid directory".format(folder))

    # other files
    other_files = worker_info.other_files.strip().split(",")
    for file in other_files:
        if os.path.isfile(file):
            src = file
            dst = os.path.join(home_dir, worker_folder, os.path.basename(src))
            print("... copying other files {} ".format(src))
            shutil.copy(src, dst)
        else:
            print("--- Other file, {}, is not a valid directory".format(folder))

    # write worker runner batch file
    write_work_bat_file(config)

    # zipp worker folder
    output_filename = os.path.join(config.General.home_folder,
                      os.path.basename(config.Worker_Info.worker_dir))

    print("... zipping the worker dir")
    shutil.make_archive(output_filename, 'zip', home_dir, config.Worker_Info.worker_dir )

def generate_pest_master_folder(config):
    pass


def generate_condor_submit_file(config):
    pass


def assemble_condor_job(config):
    pass


def assemble__run_job(config):
    pass
