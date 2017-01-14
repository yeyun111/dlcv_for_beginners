import os
import cv2
import sys

input_path = sys.argv[1].rstrip(os.sep)
target_short_edge = int(sys.argv[2])

for root, dirs, files in os.walk(input_path):
    print('scanning {} ...'.format(root))
    for filename in files:
        filepath = os.sep.join([root, filename])

        img = cv2.imread(filepath)
        h, w = img.shape[:2]
        short_edge = min(w, h)

        if short_edge > target_short_edge:
            scale = float(target_short_edge) / float(short_edge)
            new_w = int(round(w*scale))
            new_h = int(round(h*scale))
            print('Down sampling {} from {}x{} to {}x{} ...'.format(
                filepath, w, h, new_w, new_h
            ))
            img = cv2.resize(img, (new_w, new_h))
            cv2.imwrite(filepath, img)

print('Done!')
