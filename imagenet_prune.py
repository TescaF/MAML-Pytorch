import pdb
import  os.path
from os import path
import xml.etree.ElementTree as ET
import json

if __name__ == '__main__':
    img_dir = "/home/tesca/data/imagenet/n03481172/"
    box_dir = "/home/tesca/data/imagenet/n03481172/Annotation/n03481172/"
    prefix = "n03481172_"
    max_num = 35834
    dims = dict()

    for i in range(max_num + 1):
        fn = box_dir + prefix + str(i) + ".xml"
        img_fn = img_dir + prefix + str(i) + ".JPEG"
        if not (path.exists(fn) and path.exists(img_fn)):
            continue
        tree = ET.parse(fn)
        root = tree.getroot()
        objs = root.findall('object')
        if len(objs) == 1:
            box = objs[0].find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            dims[i] = [xmax - xmin, ymax - ymin]
    with open(box_dir + 'obj_dims.json', 'w') as fp:
        json.dump(dims, fp)

