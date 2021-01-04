import subprocess


def run_bash_cmd(cmd):
    print('Running command', cmd)
    process = subprocess.Popen(cmd.split(' '),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


def merge_dicts(dict_1, dict_2):
    dict_1.update(dict_2)
    return dict_1

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)