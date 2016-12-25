import os
import cv2
from tkFileDialog import askdirectory
from tkMessageBox import askyesno

WINDOW_NAME = 'Simple Bounding Box Labeling Tool'
FPS = 24
SUPPOTED_FORMATS = ['jpg', 'jpeg', 'png']
DEFAULT_COLOR = {'Object': (255, 0, 0)}
COLOR_GRAY = (192, 192, 192)
BAR_HEIGHT = 16

KEY_UP = 65362
KEY_DOWN = 65364
KEY_LEFT = 65361
KEY_RIGHT = 65363
KEY_ESC = 27
KEY_DELETE = 65535
KEY_EMPTY = 0

get_bbox_name = '{}.bbox'.format


class SimpleBBoxLabeling:

    def __init__(self, data_dir, fps=FPS, window_name=None):
        self._data_dir = data_dir
        self.fps = fps
        self.window_name = window_name if window_name else WINDOW_NAME

        self._pt0 = None
        self._pt1 = None
        self._drawing = False
        self._cur_label = None
        self._bboxes = []

        label_path = '{}.labels'.format(self._data_dir)
        self.label_colors = DEFAULT_COLOR if not os.path.exists(label_path) else self.load_labels(label_path)

        imagefiles = [x for x in os.listdir(self._data_dir) if x[x.rfind('.') + 1:].lower() in SUPPOTED_FORMATS]
        labeled = [x for x in imagefiles if os.path.exists(get_bbox_name(x))]
        to_be_labeled = [x for x in imagefiles if x not in labeled]

        self._filelist = labeled + to_be_labeled
        self._index = len(labeled)
        if self._index > len(self._filelist) - 1:
            self._index = len(self._filelist) - 1

    def _mouse_ops(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._pt0 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._pt1 = (x, y)
            self._bboxes.append((self._cur_label, (self._pt0, self._pt1)))

        elif event == cv2.EVENT_MOUSEMOVE:
            self._pt1 = (x, y)

        elif event == cv2.EVENT_RBUTTONUP:
            if self._bboxes:
                self._bboxes.pop()

    def _clean_bbox(self):
        self._pt0 = None
        self._pt1 = None
        self._drawing = False
        self._bboxes = []

    def _draw_bbox(self, img):

        h, w = img.shape[:2]
        canvas = cv2.copyMakeBorder(img, 0, BAR_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=COLOR_GRAY)

        label_msg = '{}: {}, {}'.format(self._cur_label, self._pt0, self._pt1) \
            if self._drawing \
            else 'Current label: {}'.format(self._cur_label)
        msg = '{}/{}: {} | {}'.format(self._index + 1, len(self._filelist), self._filelist[self._index], label_msg)

        cv2.putText(canvas, msg, (1, h+12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
        for label, (bpt0, bpt1) in self._bboxes:
            label_color = self.label_colors[label] if label in self.label_colors else COLOR_GRAY
            cv2.rectangle(canvas, bpt0, bpt1, label_color, thickness=2)
            cv2.putText(canvas, label, (bpt0[0]+3, bpt0[1]+15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, label_color, 2)
        if self._drawing:
            label_color = self.label_colors[self._cur_label] if self._cur_label in self.label_colors else COLOR_GRAY
            if self._pt1[0] >= self._pt0[0] and self._pt1[1] >= self._pt0[1]:
                cv2.rectangle(canvas, self._pt0, self._pt1, label_color, thickness=2)
            cv2.putText(canvas, self._cur_label, (self._pt0[0] + 3, self._pt0[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, label_color, 2)
        return canvas

    @staticmethod
    def export_bbox(filepath, bboxes):
        if bboxes:
            with open(filepath, 'w') as f:
                for bbox in bboxes:
                    line = repr(bbox) + '\n'
                    f.write(line)
        elif os.path.exists(filepath):
            os.remove(filepath)

    @staticmethod
    def load_bbox(filepath):
        bboxes = []
        with open(filepath, 'r') as f:
            line = f.readline().rstrip()
            while line:
                bboxes.append(eval(line))
                line = f.readline().rstrip()
        return bboxes

    @staticmethod
    def load_labels(filepath):
        label_colors = {}
        with open(filepath, 'r') as f:
            line = f.readline().rstrip()
            while line:
                label, color = eval(line)
                label_colors[label] = color
                line = f.readline().rstrip()
        return label_colors

    @staticmethod
    def load_sample(filepath):
        img = cv2.imread(filepath)
        bbox_filepath = get_bbox_name(filepath)
        bboxes = []
        if os.path.exists(bbox_filepath):
            bboxes = SimpleBBoxLabeling.load_bbox(bbox_filepath)
        return img, bboxes

    def _export_n_clean_bbox(self):
        bbox_filepath = os.sep.join([self._data_dir, get_bbox_name(self._filelist[self._index])])
        self.export_bbox(bbox_filepath, self._bboxes)
        self._clean_bbox()

    def _delete_current_sample(self):
        filename = self._filelist[self._index]
        filepath = os.sep.join([self._data_dir, filename])
        if os.path.exists(filepath):
            os.remove(filepath)
        filepath = get_bbox_name(filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        self._filelist.pop(self._index)
        print('{} is deleted!'.format(filename))

    def start(self):

        last_filename = ''
        label_index = 0
        labels = self.label_colors.keys()
        n_labels = len(labels)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_ops)
        key = KEY_EMPTY
        delay = int(1000 / FPS)

        while key != KEY_ESC:

            if key == KEY_UP:
                if label_index == 0:
                    pass
                else:
                    label_index -= 1

            elif key == KEY_DOWN:
                if label_index == n_labels - 1:
                    pass
                else:
                    label_index += 1

            elif key == KEY_LEFT:
                if self._index > 0:
                    self._export_n_clean_bbox()

                self._index -= 1
                if self._index < 0:
                    self._index = 0

            elif key == KEY_RIGHT:
                if self._index < len(self._filelist) - 1:
                    self._export_n_clean_bbox()

                self._index += 1
                if self._index > len(self._filelist) - 1:
                    self._index = len(self._filelist) - 1

            elif key == KEY_DELETE:
                if askyesno('Delete Sample', 'Are you sure?'):
                    self._delete_current_sample()
                    key = KEY_EMPTY
                    continue

            filename = self._filelist[self._index]
            if filename != last_filename:
                filepath = os.sep.join([self._data_dir, filename])
                img, self._bboxes = self.load_sample(filepath)

            self._cur_label = labels[label_index]

            canvas = self._draw_bbox(img)
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(delay)

            last_filename = filename

        print('Finished!')

        cv2.destroyAllWindows()
        self.export_bbox(os.sep.join([self._data_dir, get_bbox_name(filename)]), self._bboxes)

        print('Labels updated!')

if __name__ == '__main__':
    dir_with_images = askdirectory(title='Where are the images?')
    labeling_task = SimpleBBoxLabeling(dir_with_images)
    labeling_task.start()
