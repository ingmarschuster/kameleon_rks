import os
from os.path import expanduser
from time import strftime, gmtime, sleep

import numpy as np
import pandas as pd
import hashlib

from kameleon_rks.tools.log import Log

logger = Log.get_logger()

def _create_dir_if_not_exist(fname):
    # create result dir if wanted
    if os.sep in fname:
        try:
            directory = os.sep.join(fname.split(os.sep)[:-1])
            os.makedirs(directory)
        except OSError:
            pass

def store_samples(samples, fname = expanduser("~") + os.sep  + "results.txt", **kwargs):
    # add filename if only path is given
    if fname[-1] == os.sep:
        fname += "results.txt"
    
    _create_dir_if_not_exist(fname)
    
    # very crude protection against conflicting access from parallel processes
    write_success = False
    while not write_success:
        try:
            # append to file
            f_handle = file(fname, 'a')
            np.savetxt(f_handle, samples)
            f_handle.close()
            write_success = True
        except IOError:
            print("IOError writing to %s ... trying again in 1s." % fname)
            sleep(1)

def store_results(fname = expanduser("~") + os.sep  + "results.txt", **kwargs):
    # add filename if only path is given
    if fname[-1] == os.sep:
        fname += "results.txt"
    
    _create_dir_if_not_exist(fname)
    
    # use current time as index for the dataframe
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    new_df = pd.DataFrame([[kwargs[k] for k in kwargs.keys()]], index=[current_time],columns=kwargs.keys())
    
    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col=0)
        df = df.append(new_df)
    else:
        df = new_df

    # very crude protection against conflicting access from parallel processes
    write_success = False
    while not write_success:
        try:
            df.to_csv(fname)
            write_success = True
        except IOError:
            print("IOError writing to csv ... trying again in 1s.")
            sleep(1)
        
def assert_file_has_sha1sum(fname, sha1_reference):
    sha1 = sha1sum(fname)
    if not sha1 == sha1_reference:
        raise RuntimeError("File %s has sha1sum %s which is different from the provided reference %s" % \
                           (fname, sha1, sha1_reference))

def sha1sum(fname, blocksize=65536):
    """
    Computes sha1sum of the given file. Same as the unix command line hash.
    
    Returns: string with the hex-formatted sha1sum hash
    """
    hasher = hashlib.sha1()
    with open(fname, 'rb') as afile:
        logger.debug("Hashing %s" % fname)
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
    return hasher.hexdigest()
