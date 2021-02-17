import os
from os.path import join as opj
from shutil import copyfile, SameFileError
import git  # pip install gitpython
from sqlalchemy import create_engine
import numpy as np
import subprocess
from importlib.machinery import SourceFileLoader


def load_configs(configs_path, assign_name='cfg'):
    """See: https://stackoverflow.com/questions/67631/how-to-import-a- ...
     ... module-given-the-full-path"""
    return SourceFileLoader(assign_name, configs_path).load_module()


def save_configs(configs_path, results_path, warn=True):
    """ save a copy of config file and last commit hash for reproducibility
    see: https://stackoverflow.com/questions/14989858/ ...
    get-the-current-git-hash-in-a-python-script
    """
    savename = opj(results_path, os.path.basename(configs_path))
    if warn and os.path.exists(savename):
        input(
            f"This will OVERWRITE: {savename}\n"
            "Are you SURE you want to continue?? (press Ctrl+C to abort)"
        )
    try:
        copyfile(configs_path, savename)
    except SameFileError:
        pass
    repo = git.Repo(search_parent_directories=True)
    with open(opj(results_path, "last_commit_hash.txt"), 'w') as f:
        f.write(repo.head.object.hexsha)


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def ordered_vals_from_ordered_dict(d):
    vs = []
    for v in d.values():
        if v not in vs:
            vs.append(v)
    return vs


def connect_to_sqlite(db_path: str):
    sql_engine = create_engine('sqlite:///' + db_path, echo=False)
    return sql_engine.connect()


def maybe_mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def isGPUDevice():
    """Determine if device is an NVIDIA GPU device"""
    return os.system("nvidia-smi") == 0


def AllocateGPU(
        N_GPUs=1, GPUs_to_use=None, TOTAL_GPUS=4,
        verbose=True, N_trials=0):
    """Restrict GPU use to a set number or name.
    Args:
        N_GPUs - int, number of GPUs to restrict to.
        GPUs_to_use - optional, list of int ID's of GPUs to use.
                    if none, this will fetch GPU's with lowest
                    memory consumption
        verbose - bool, print to screen?
    """
    # only restrict if not a GPU machine or already restricted
    isGPU = isGPUDevice()

    assert TOTAL_GPUS == 4, 'Only 4-GPU machines supported for now.'

    try:
        AlreadyRestricted = os.environ["CUDA_VISIBLE_DEVICES"] is not None
    except KeyError:
        AlreadyRestricted = False

    if isGPU and (not AlreadyRestricted):
        try:
            if GPUs_to_use is None:

                if verbose:
                    print("Restricting GPU use to {} GPUs ...".format(N_GPUs))

                # If you did not specify what GPU to use, this will just
                # fetch the GPUs with lowest memory consumption.

                # Get processes from nvidia-smi command
                gpuprocesses = str(
                    subprocess.check_output("nvidia-smi", shell=True)) \
                    .split('\\n')
                # Parse out numbers, representing GPU no, PID and memory use
                start = 24
                gpuprocesses = gpuprocesses[start:len(gpuprocesses) - 2]
                gpuprocesses = [j.split('MiB')[0] for i, j in
                                enumerate(gpuprocesses)]

                # Add "fake" zero-memory processes to represent all GPUs
                extrapids = np.zeros([TOTAL_GPUS, 3])
                extrapids[:, 0] = np.arange(TOTAL_GPUS)

                PIDs = []
                for p in range(len(gpuprocesses)):
                    pid = [int(s) for s in gpuprocesses[p].split() if
                           s.isdigit()]
                    if len(pid) > 0:
                        PIDs.append(pid)
                # PIDs.pop(0)
                PIDs = np.array(PIDs)

                if len(PIDs) > 0:
                    PIDs = np.concatenate((PIDs, extrapids), axis=0)
                else:
                    PIDs = extrapids

                # Get GPUs memory consumption
                memorycons = np.zeros([TOTAL_GPUS, 2])
                for gpuidx in range(TOTAL_GPUS):
                    thisgpuidx = 1 * np.array(PIDs[:, 0] == gpuidx)
                    thisgpu = PIDs[thisgpuidx == 1, :]
                    memorycons[gpuidx, 0] = gpuidx
                    memorycons[gpuidx, 1] = np.sum(thisgpu[:, 2])

                # sort and get GPU's with lowest consumption
                memorycons = memorycons[memorycons[:, 1].argsort()]
                GPUs_to_use = list(np.int32(memorycons[0:N_GPUs, 0]))

            # Now restrict use to available GPUs
            gpus_list = GPUs_to_use.copy()
            GPUs_to_use = ",".join([str(j) for j in GPUs_to_use])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = GPUs_to_use
            if verbose:
                print("Restricted GPU use to GPUs: " + GPUs_to_use)
            return gpus_list

        except ValueError:
            if N_trials < 2:
                if verbose:
                    print("Got value error, trying again ...")
                N = N_trials + 1
                AllocateGPU(N_GPUs=N_GPUs, N_trials=N)
            else:
                raise ValueError(
                    "Something is wrong, tried too many times and failed.")

    else:
        if verbose:
            if isGPU:
                print("No GPU allocation done.")
            if AlreadyRestricted:
                print("GPU devices already allocated.")


def Merge_dict_with_default(
        dict_given: dict, dict_default: dict, keys_Needed: list = None):
    """Sets default values of dict keys not given"""

    keys_default = list(dict_default.keys())
    keys_given = list(dict_given.keys())

    # Optional: force user to unput some keys (eg. those without defaults)
    if keys_Needed is not None:
        for j in keys_Needed:
            if j not in keys_given:
                raise KeyError("Please provide the following key: " + j)

    keys_Notgiven = [j for j in keys_default if j not in keys_given]

    for j in keys_Notgiven:
        dict_given[j] = dict_default[j]

    return dict_given


def file_len(fname: str):
    """
    Given a filename, get number of lines it has efficiently. See:
    https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    """
    try:
        p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        return int(result.strip().split()[0])

    except FileNotFoundError:
        # on windows systems where subprocess and file paths are weird
        with open(fname) as fp:
            count = 0
            for _ in fp:
                count += 1
        return count


def kill_all_nvidia_processes():
    """
    Force kills all NVIDIA processes, even if they don't
    show up in NVIDIA_SMI (common when tensorflow gets crazy
    and you kill the kernel or it dies).
    """

    input("Killing all gpu processes .. continue?" +
          "Press any button to continue, or Ctrl+C to quit ...")

    # get gpu processes -- note that this
    # gets processes even if they don't show up
    # in the nvidia-smi command (which happens
    # often with tensorflow)
    gpuprocesses = str(subprocess.check_output(
        "fuser -v /dev/nvidia*", shell=True)).split('\\n')

    # preprocess process list
    gpuprocesses = gpuprocesses[0].split(" ")[1:]
    if "'" in gpuprocesses[-1]:
        gpuprocesses[-1] = gpuprocesses[-1].split("'")[0]

    # put into string form
    gpuprocesses_str = '{'
    for pr in gpuprocesses:
        gpuprocesses_str += str(pr) + ','
    gpuprocesses_str += '}'

    # now kill
    kill_command = "kill -9 %s" % (gpuprocesses_str)
    os.system(kill_command)

    print("killed the following processes: " + gpuprocesses_str)


if __name__ == '__main__':
    AllocateGPU(N_GPUs=1, TOTAL_GPUS=8)
