import os
import sys

dup_list = sys.argv[1]

with open(dup_list, 'r') as f:
    lines = f.readlines()
    for line in lines:
        dups = line.split()
        print('Removing duplicates of {}'.format(dups[0]))
        for dup in dups[1:]:
            cmd = 'rm {}'.format(dup)
            os.system(cmd)
