import os
import sys
import numpy
from matplotlib import pyplot

LOG_FILENAME = 'log.txt'
TRAIN_LOSS_KEYWORD = '| Training loss: '
VAL_LOSS_KEYWORD = '| Validation loss: '
ITER_INDEX = 4
LOSS_INDEX = -1


def parse_log(filepath):
    with open(filepath, 'r') as f:
        train_curve = []
        val_curve = []
        line = f.readline()
        while line:
            if TRAIN_LOSS_KEYWORD in line or VAL_LOSS_KEYWORD in line:
                tokens = line.split()
                loss = [int(tokens[ITER_INDEX]), float(tokens[LOSS_INDEX])]
                if TRAIN_LOSS_KEYWORD in line:
                    train_curve.append(loss)
                else:
                    val_curve.append(loss)
            line = f.readline()
    return train_curve, val_curve

root_dir = sys.argv[1].rstrip(os.sep)
keyword = sys.argv[2] if len(sys.argv) > 2 else None

groups = [x for x in os.listdir(root_dir) if os.path.isdir(x) and (True if keyword is None else (keyword in x))]

for group in groups:
    log_path = os.sep.join([root_dir, group, LOG_FILENAME])
    train_loss, val_loss = parse_log(log_path)
    train_loss = numpy.array(train_loss)
    val_loss = numpy.array(val_loss)
    pyplot.figure('Train/Test Loss Curves')
    pyplot.plot(train_loss[:, 0], train_loss[:, 1], label=group)
    pyplot.plot(val_loss[:, 0], val_loss[:, 1], '--', label=group)

pyplot.legend(loc='top right')
pyplot.show()
