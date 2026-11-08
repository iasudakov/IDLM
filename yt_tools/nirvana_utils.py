# Nirvana dependencies
try:
    import nirvana_dl
    from distutils.dir_util import copy_tree, remove_tree
    import os
except ImportError:
    nirvana_dl = None
    
    
    
def clean_snapshot_path():
    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Clean the previous state in {snapshot_path}")
        remove_tree(snapshot_path)


def copy_snapshot_to_out(out):
    """ The preempted run transfers its "state" to the restarted run through "snapshot path".
        "state" is a tar-archive that contains all files put into "snapshot path" by the preempted run.
        This function moves the files in the "state" archive to you local "out" dir.
    """
    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy the previous state from {snapshot_path} to {out}")
        copy_tree(snapshot_path, out)
    
        # os.system(f"tar -xf {out}/state -C {out}/")
        # ^-- This is not needed for the recent PythonDL blocks. 
        # The state should be already unpacked in snapshot_path
    

def copy_out_to_snapshot(out, dump=True):
    """ This function copies all files in the local "out" directory to "snapshot path".
        dump: If True, put these files into tar-archive "state" and 
              send it to the Python DL output.  
    """
    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy {out} to the snapshot path: {snapshot_path}")

        # Delete previous state to avoid memory explosion
        copy_tree(out, snapshot_path)

        if dump:
            # Make it visible in the Python DL output
            nirvana_dl.snapshot.dump_snapshot(snapshot_path)
    else:
        print(f"Nirvana DL is not found. Checkpoint is not transfered to the snapshot path.")


def copy_out_to_output_data(out):
    """ This function copies all files in the local "out" directory to "output_data path".
    """
    if nirvana_dl:
        data_path = nirvana_dl.output_data_path()
        print(f"Copy {out} to the output_data path: {data_path}")

        # Delete previous state to avoid memory explosion
        copy_tree(out, data_path)
    else:
        print(f"Nirvana DL is not found. Data is not transfered to the output_data path.")


def copy_logs_to_logs_path(out):
    """ This function copies all files in the local "out" directory to "logs path".
    """
    if nirvana_dl:
        logs_path = nirvana_dl.logs_path()
        print(f"Copy {out} to the logs path: {logs_path}")

        # Delete previous state to avoid memory explosion
        copy_tree(out, logs_path)
    else:
        print(f"Nirvana DL is not found. Logs are not transfered to the logs path.")