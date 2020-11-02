import subprocess


def run_bash_cmd(cmd):
    print('Running command', cmd)
    process = subprocess.Popen(cmd.split(' '),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)