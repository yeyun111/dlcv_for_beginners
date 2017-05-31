import os
import cv2
import sys

folders = sys.argv[1:-1]
length = int(sys.argv[-1])

cnt = 0
for folder in folders:
    print('scanning {} ...'.format(folder))
    folder = folder.rstrip('/')
    files = os.listdir(folder)

    for img_file in files:
        filepath = '{}/{}'.format(folder, img_file)
        try:
            img = cv2.imread(filepath)
            h, w, c = img.shape
        except:
            print('problematic file:', filepath)
            continue

        if img is None:
            print('problematic file:', filepath)
            continue
        elif h == length and w == length:
            continue
        else:
            if h > w:
                dl = int((h-w)/2)
                if dl > 0:
                    img = img[dl:-dl, ...]
            else:
                dl = int((w-h)/2)
                if dl > 0:
                    img = img[:, dl:-dl, ...]
            img = cv2.resize(img, (length, length))
            cv2.imwrite(filepath, img)

        cnt += 1
        if cnt % 100 == 0:
            print('{} images processed!'.format(cnt))

print('Done!')

