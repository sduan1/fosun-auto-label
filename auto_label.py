import argparse
import os
import numpy as np
import base64
import labelme
import torch
import torch.nn as nn
import cv2
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.nn.functional import conv2d
import json


parser = argparse.ArgumentParser(description='Auto labeler')
parser.add_argument('--rootdir', type=str,
                    help='an integer for the accumulator')

args = parser.parse_args()
ROOT_DIR = args.rootdir


def calc_dist(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2

    return np.sqrt((x1-x2)**2+(y1-y2)**2)


def valid_point(candidates, p, thresh=20):
    for c in candidates:
        if calc_dist(c, p) <= thresh:
            return False
    return True


def filter_candidates(candidates, res_len=3):
    labels = []
    i = 0
    while len(labels) != res_len:
        if valid_point(labels, candidates[i]):
            labels.append(candidates[i])
        i += 1
    return labels


def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def assgin_labels_to_points(points):
    ret = {}
    ys = [p[1] for p in points]
    top_idx = np.argmin(ys)
    ret['top'] = points[top_idx]
    _ = points.pop(top_idx)

    xs = [p[0] for p in points]
    right_idx = np.argmax(xs)

    ret['right'] = points[right_idx]
    _ = points.pop(right_idx)

    ret['left'] = points[0]
    return ret


def to_json(points, image_name, image):
    (h, w, c) = image.shape
    ret = {
        "version": "5.0.1",
        "flags": {},
    }
    shapes = []
    labeled_points = assgin_labels_to_points(points)
    for k, p in labeled_points.items():
        shape_data = {
            "label": k,
            "points": list(p),
            "ground_id": None,
            "shape_type": "point",
            "flags": {}
        }
        shapes.append(shape_data)
    ret['shapes'] = shapes
    ret['imagePath'] = image_name
    data = labelme.LabelFile.load_image_file(f'{ROOT_DIR}/{image_name}')
    image_data = base64.b64encode(data).decode('utf-8')
    ret['imageData'] = image_data
    ret['imageHeight'] = h
    ret['imageWidth'] = w

    return ret


def label_cross(img, cuda=True, center_crop_size=400, num_of_candidates=10):
    kernel = cv2.imread('cross.png')
    kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY).astype(np.int16)
    kernel = (255 - kernel)*100 - 2000
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.cuda() if cuda else kernel

    image = cv2.imread(f'{ROOT_DIR}/{img}')
    (h, w, c) = image.shape
    orig_img = cv2.imread(f'{ROOT_DIR}/{img}')
    image = image[:, :, 1]
    image = center_crop(image, (center_crop_size, center_crop_size))

    image = torch.from_numpy(image)
    image = image.float().unsqueeze(0).unsqueeze(0)
    image = image.cuda() if cuda else image

    output = conv2d(image, kernel.unsqueeze(
        0).unsqueeze(0).cuda(), padding='same')
    output = output.cpu() if cuda else output
    output_numpy = output.numpy()
    output_numpy = np.clip(output_numpy, 0, None)

    output_numpy = (output_numpy-np.min(output_numpy)) / \
        (np.max(output_numpy)-np.min(output_numpy))
    output_numpy *= 255

    output_numpy = output_numpy[0, 0, :, :]
    v, i = torch.topk(torch.from_numpy(
        output_numpy).flatten(), num_of_candidates)
    indexes = np.array(np.unravel_index(i.numpy(), output_numpy.shape)).T
    candidates = indexes

    numpy_points = filter_candidates(candidates)

    h_offset = (h-400)/2
    w_offset = (w-400)/2
    labelme_points = [(x[1]+w_offset, x[0]+h_offset) for x in numpy_points]
    print(labelme_points)
    json_data = to_json(labelme_points, img, orig_img)
    with open(f'{ROOT_DIR}/{img[0:-4]}.json', 'w') as f:
        json.dump(json_data, f)


def label_phil(img):
    print('not yet implemented')
    pass


def main():

    print('Scanning label directory...')
    rootdir_files = os.listdir(ROOT_DIR)
    rootdir_len = len(rootdir_files)

    print(f'Directory files: {rootdir_len}')
    json_files = [x for x in rootdir_files if '.json' in x]
    print(f'finished labels: {len(json_files)}')
    image_files = [x for x in rootdir_files if '.jpg' in x]
    print(f'Todo: {len(image_files)} files')
    assert (len(json_files) + len(image_files) ==
            rootdir_len), 'Directory contains files other than \'jpg\' or \'json\''

    for i in image_files:
        image_name = i[0:-4]
        if '_' in image_name:
            try:
                label_cross(i)
            except Exception as e:
                print(f'WARNING: label of {i} failed:\n {e}')
        else:
            label_phil(i)


if __name__ == '__main__':
    main()
