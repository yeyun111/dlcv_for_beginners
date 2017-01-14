import os
import sys
import cv2
from collect_data import SUPPORTED_FORMATS

input_path = sys.argv[1]

for root, dirs, files in os.walk(input_path):
    for filename in files:
        ext = filename[filename.rfind('.')+1:].lower()
        if ext not in SUPPORTED_FORMATS:
            continue
        filepath = os.sep.join([root, filename])
        if cv2.imread(filepath) is None:
            os.system('rm {}'.format(filepath))
            print('{} is not a valid image file. Deleted!'.format(filepath))
