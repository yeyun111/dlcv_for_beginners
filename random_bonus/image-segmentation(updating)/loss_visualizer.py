import os
import sys
import numpy
from matplotlib import pyplot

LOG_FILENAME = 'log.txt'
TRAIN_LOSS_KEYWORD = '| Training loss: '
VAL_LOSS_KEYWORD = '| Validation loss: '
ITER_INDEX = 4
LOSS_INDEX = -1
MIOU_INDEX = -5
MPA_INDEX = -9


def parse_log(filepath):
    with open(filepath, 'r') as f:
        train_curve = []
        val_curve = []
        line = f.readline()
        while line:
            if TRAIN_LOSS_KEYWORD in line or VAL_LOSS_KEYWORD in line:
                tokens = line.split()
                measure = [int(tokens[ITER_INDEX]), float(tokens[LOSS_INDEX])]
                if TRAIN_LOSS_KEYWORD in line:
                    train_curve.append(measure)
                else:
                    measure.extend([float(tokens[MPA_INDEX]), float(tokens[MIOU_INDEX])])
                    val_curve.append(measure)

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
    pyplot.figure('mPA/mIOU Curves')
    pyplot.plot(val_loss[:, 0], val_loss[:, 2], label='{}-mPA'.format(group))
    pyplot.plot(val_loss[:, 0], val_loss[:, 3], '--', label='{}-mIOU'.format(group))

pyplot.figure('Train/Test Loss Curves')
pyplot.legend(loc='upper right')
pyplot.figure('mPA/mIOU Curves')
pyplot.legend(loc='lower right')
pyplot.show()
