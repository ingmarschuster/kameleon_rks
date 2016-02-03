import hashlib

from kameleon_rks.tools.log import Log

logger = Log.get_logger()

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
        logger.debug("Hasing %s" % fname)
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
    return hasher.hexdigest()
