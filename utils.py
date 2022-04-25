import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import exifread

# Focal length: 2600 for Elephant mountain
#               2400 for Taipei101
#                705 for parrington

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('-i', '--input_dir', default='test3', type=str, help='Folder of input images.')
    parser.add_argument('-p', '--plot', default='False', type=str, help='Whether to plot result or not.')
    parser.add_argument('-r', '--match_ratio', default=0.8, type=float, help='Ratio for keypoint matching.')
    parser.add_argument('-f', '--focal_length', default=0, type=float, help='focal length of image.')
    parser.add_argument('-d', '--degree', default=0, type=float, help='rotation of image.')
    return parser

def imgImportFromPil(img_path: str):
    pil_img = Image.open(img_path).convert("RGB")
    pil_img = pil_img.resize([1000, 1500])
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    exif = pil_img.getexif()
    return cv_img

def getExifFromPath(img_path, exif=None):
    if exif is None:
        exif = {}
    with open(img_path, 'rb') as file:
        tags = exifread.process_file(file, details=False)
        # for key, val in tags.items():
        #     print(key, val)
        if 'EXIF ExposureTime' in tags:
            exif['exposure_time'] = eval(str(tags['EXIF ExposureTime']))
        if 'EXIF ISOSpeedRatings' in tags:
            exif['iso'] = eval(str(tags['EXIF ISOSpeedRatings']))
        if 'EXIF FocalLength' in tags:
            exif['focal_len'] = eval(str(tags['EXIF FocalLength']))
        
    return exif

def imshow_plt(img, color_bar=False):
    plt.figure(figsize=(10, 8))
    if len(img.shape) > 2:
        plt_img = img[:, :, ::-1]
    else:
        plt_img = img
    plt.imshow(plt_img)
    if color_bar:
        plt.colorbar()
    plt.show()
    return
    
def imshows_plt(imgs):
    fig = plt.figure(figsize=(20, 20))
    for i, img in enumerate(imgs):
        fig.add_subplot(1, len(imgs), i + 1)
        plt_img = img
        plt.imshow(plt_img)
    plt.colorbar()
    plt.show()
    return
    
def list_plt(list):
    index_list = [i for i in range(len(list))]
    plt.bar(index_list, list)
    plt.show()

def plot_matches(matches, total_img, offset):
    match_img = total_img.copy()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0, 0], matches[:, 0, 1], 'xr')
    ax.plot(matches[:, 1, 0] + offset, matches[:, 1, 1], 'xr')
     
    ax.plot([matches[:, 0, 0], matches[:, 1, 0] + offset], [matches[:, 0, 1], matches[:, 1, 1]],
            'r', linewidth=0.5)

    plt.show()