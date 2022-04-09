import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('-i', '--input_dir', default='test', type=str, help='Folder of input images.')
    parser.add_argument('-p', '--plot', default='False', type=str, help='Whether to plot result or not.')
    return parser

def imgImportFromPil(img_path: str):
    pil_img = Image.open(img_path).convert("RGB")
    pil_img = pil_img.resize([1500, 1000])
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    exif = pil_img.getexif()
    return cv_img

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