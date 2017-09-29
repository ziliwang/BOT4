from PIL import Image
import os
import pickle
import cv2
import numpy as np
import click
from sklearn.externals import joblib


#  global
TMP_FOLDER = os.path.join('train_cal', 'tmp')
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)
OUT_FOLDER = False
HEIGH = 0
WIDTH = 0
BOUND = 2
TRAINSETNAME = '.traindata.sav'
INDEX_OUT = {}
COM_OUT = {}
MAP_INDEX_KEY = {}


def process_image(_file):
    img = Image.open(_file)
    if HEIGH + WIDTH == 0:
        raise ValueError('illegal height and width')
    img.thumbnail([2 * HEIGH, HEIGH + BOUND * 2])  # 缩小至指定高度
    img = img.crop((img.size[0] - BOUND - WIDTH,
                    BOUND,
                    img.size[0] - BOUND,
                    img.size[1] - BOUND))
    tmp_out = os.path.join(TMP_FOLDER, 'tmp.jpg')
    img.save(tmp_out)
    if os.path.exists(tmp_out):
        img = cv2.imread(tmp_out)
        return np.abs(img.astype(np.int32) - 255) / 255
    else:
        raise FileExistsError(tmp_out)


def check_format(_type, header):
    global MAP_INDEX_KEY
    if _type == 'index':
        global INDEX_OUT
        check = INDEX_OUT
    elif _type == 'com':
        global COM_OUT
        check = COM_OUT
    else:
        raise ValueError('illegal arguments')
    if len(MAP_INDEX_KEY.keys()) > 0:
        for i, v in enumerate(header):
            if i == 0:
                if len(check.keys()) < len(MAP_INDEX_KEY.keys()):
                    check['x'] = []
                continue
            if MAP_INDEX_KEY[i] != v:
                return False
            if len(check.keys()) < len(MAP_INDEX_KEY.keys()) + 1:
                check[v] = []
    else:
        for i, v in enumerate(header):
            if i == 0:
                check['x'] = []
                continue
            MAP_INDEX_KEY[i] = v
            check[v] = []
    return check


def one_type_process(lab_file, image_dir, _type):
    with open(lab_file, encoding='utf-8') as f:
        d = [i.split('\t') for i in f.read().split('\n') if i]
    # check OUT
    out = check_format(_type=_type, header=d[0])
    if not out:
        raise ValueError('Ununiform format')
    for items in d[1:]:
        if items
        # matrix
        if items[0].endswith('.jpg'):
            image_file = os.path.join(image_dir, items[0])
        else:
            image_file = os.path.join(image_dir, items[0] + '.jpg')
        if os.path.exists(image_file):
            out['x'].append(process_image(image_file))
            for i, item in enumerate(items[1:]):
                out[MAP_INDEX_KEY[i + 1]].append(item)
        else:
            print('{} not exists'.format(items[0]))


def process(folder):
    # check
    if not set(['labs.txt', 'index_labs.txt']).issubset(set(os.listdir(folder))):
        raise(ValueError('illegal folder'))
    # index
    one_type_process(lab_file=os.path.join(folder, 'index_labs.txt'),
                     image_dir=os.path.join(folder, 'index_pic'),
                     _type='index')
    # gegu
    one_type_process(lab_file=os.path.join(folder, 'labs.txt'),
                     image_dir=os.path.join(folder, 'pic'),
                     _type='com')


@click.command()
@click.option('--data', help='train data folder')
@click.option('--height', help='image height')
@click.option('--width', help='image width')
def main(data, height, width):
    global HEIGH
    global WIDTH
    global OUT_FOLDER
    HEIGH = int(height)
    WIDTH = int(width)
    OUT_FOLDER = os.path.join('train_cal', '{}_{}'.format(HEIGH, WIDTH))
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    scan_dirs = data.split(',')
    for i in scan_dirs:
        process(i)
    # save
    joblib.dump(COM_OUT, os.path.join(OUT_FOLDER, 'com_train.sav'))
    joblib.dump(INDEX_OUT, os.path.join(OUT_FOLDER, 'index_train.sav'))


if __name__ == "__main__":
    main()
