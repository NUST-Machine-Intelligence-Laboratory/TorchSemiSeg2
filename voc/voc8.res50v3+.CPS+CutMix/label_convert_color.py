
from __future__ import print_function
import os
import sys
import numpy as np
from skimage.io import imread, imsave
import glob


def pascal_palette():
    palette = {(0, 0, 0): 0,
               (128, 0, 0): 1,
               (0, 128, 0): 2,
               (128, 128, 0): 3,
               (0, 0, 128): 4,
               (128, 0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64, 0, 0): 8,
               (192, 0, 0): 9,
               (64, 128, 0): 10,
               (192, 128, 0): 11,
               (64, 0, 128): 12,
               (192, 0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0, 64, 0): 16,
               (128, 64, 0): 17,
               (0, 192, 0): 18,
               (128, 192, 0): 19,
               (0, 64, 128): 20}

    return palette


def convert_from_color_segmentation(seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = pascal_palette()

    for c, i in palette.items():
        color_seg[ seg == i,:] = c
        
    # color_seg = color_seg[..., ::-1]

    return color_seg


def main():
    ##
    ext = '.png'
    ##
    # path, txt_file, path_converted = process_arguments(sys.argv)
    path = '/data/TorchSemiSeg2/DATA/pascal_voc/val/label/*.png'					# 前面转换后的语义图片
    path_converted = '/home/chenrui/code/TorchSemiSeg2/exp.voc/voc8.res50v3+.CPS+CutMix/color_label/'				# 着色后的图片的保存位置

    # Create dir for converted labels
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    f = glob.glob(path)
    for img_name in f:
        img_base_name = os.path.basename(img_name)
        img = imread(img_name)

        if (len(img.shape) == 2):
            img = convert_from_color_segmentation(img)
            imsave(os.path.join(path_converted, img_base_name), img)
        else:
            print(img_name + " is not composed of three dimensions, therefore "
                                "shouldn't be processed by this script.\n"
                                "Exiting.", file=sys.stderr)
            exit()


def process_arguments(argv):
    if len(argv) != 4:
        help()

    path = argv[1]
    list_file = argv[2]
    new_path = argv[3]

    return path, list_file, new_path


def help():
    print('Usage: python convert_labels.py PATH LIST_FILE NEW_PATH\n'
          'PATH points to directory with segmentation image labels.\n'
          'LIST_FILE denotes text file containing names of images in PATH.\n'
          'Names do not include extension of images.\n'
          'NEW_PATH points to directory where converted labels will be stored.'
          , file=sys.stderr)
    exit()


if __name__ == '__main__':
    main()
