import argparse
import math
from distutils.util import strtobool
from typing import List
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('-i', '--input_dir', default='data/parrington', type=str, help='Folder of input images.')
    parser.add_argument('-p', '--plot', default='False', type=str, help='Whether to plot result or not.')
    return parser

def imgImportFromPil(img_path: str):
    pil_img = Image.open(img_path).convert("RGB")
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    exif = pil_img.getexif()
    return cv_img

if __name__ == '__main__':
    total_time = 0
    start = time.time()
    parser = get_parser()
    args = parser.parse_args()
    
    img_contents = []
    '''
    [
        {
            "filepath": FILEPATH,
            "data": OPENCV_IMG,
            "offset": {"x": int, "y": int}, # 1 means shift left or top, -1 means shift right or down
            "brightness": INT
        }
    ]
    '''

    for file in os.listdir(args.input_dir):
        file_lower = file.lower()
        if file_lower.endswith('.jpg') or file_lower.endswith('.png'):
            img_filePath = os.path.join(args.input_dir, file)
            img = imgImportFromPil(img_filePath)

            img_contents.append({
                'filepath': img_filePath,
                'data': img,
                "offset": {"x": 0, "y": 0},
            })