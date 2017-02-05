import os
import sys
import xml.etree.ElementTree as ET
#import xml.dom.minidom as minidom
import cv2
from bbox_labeling import SimpleBBoxLabeling

input_dir = sys.argv[1].rstrip(os.sep)

bbox_filenames = [x for x in os.listdir(input_dir) if x.endswith('.bbox')]

for bbox_filename in bbox_filenames:
    bbox_filepath = os.sep.join([input_dir, bbox_filename])
    jpg_filepath = bbox_filepath[:-5]
    if not os.path.exists(jpg_filepath):
        print('Something is wrong with {}!'.format(bbox_filepath))
        break

    root = ET.Element('annotation')

    filename = ET.SubElement(root, 'filename')
    jpg_filename = jpg_filepath.split(os.sep)[-1]
    filename.text = jpg_filename

    img = cv2.imread(jpg_filepath)
    h, w, c = img.shape
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(c)

    bboxes = SimpleBBoxLabeling.load_bbox(bbox_filepath)
    for obj_name, coord in bboxes:
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = obj_name
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymin = ET.SubElement(bndbox, 'ymin')
        ymax = ET.SubElement(bndbox, 'ymax')
        (left, top), (right, bottom) = coord
        xmin.text = str(left)
        xmax.text = str(right)
        ymin.text = str(top)
        ymax.text = str(bottom)

    xml_filepath = jpg_filepath[:jpg_filepath.rfind('.')] + '.xml'
    with open(xml_filepath, 'w') as f:
        anno_xmlstr = ET.tostring(root)

        # In case a nicely formatted xml is needed
        # uncomment the following 2 lines and minidom import
        #anno_xml = minidom.parseString(anno_xmlstr)
        #anno_xmlstr = anno_xml.toprettyxml()
        f.write(anno_xmlstr)
