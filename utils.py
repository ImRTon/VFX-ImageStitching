import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def imshow_plt(img):
    plt.figure(figsize=(10, 8))
    if len(img.shape) > 2:
        plt_img = img[:, :, ::-1]
    else:
        plt_img = img
    plt.imshow(plt_img)
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