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
import utils
import SIFT
import imageStitching
from scipy import spatial
import imutils

def GetKeyPointAndDescriptor(img_content):
    DoGOctaves, GaussianOctaves = SIFT.to_gaussian_list(img_content['data'])
    magOctaves, sitaOctaves = SIFT.get_gradients(GaussianOctaves)

    idv_keypoints = []
    keypoint_dict = {}
    discard_pnts_count = 0

    for octave_idx, octave in enumerate(DoGOctaves):
        keypoints = SIFT.find_extrema_in_DoG(DoGOctaves[octave_idx])
        for keypoint in keypoints:
            var = SIFT.findScaleSpaceExtrema(DoGOctaves[octave_idx], keypoint[0], keypoint[1], keypoint[2])
            if var is not None:
                row, col, sigma_idx, extremum = var
                if (row, col) not in keypoint_dict:
                    sigma = 1.6
                    keypoint_size = sigma * (2 ** ((sigma_idx + extremum[2]) / 3.0)) * (2 ** (octave_idx + 1))
                    keypnt = SIFT.KeyPoint((col + extremum[0]) * (2 ** octave_idx), (row + extremum[1]) * (2 ** octave_idx), sigma_idx=sigma_idx, size=keypoint_size)
                    keypnt.octave = octave_idx
                    idv_keypoints.extend(SIFT.get_orientation(keypnt, sigma_idx, magOctaves, sitaOctaves))
                    # keypoint_dict[(row, col)] = 1
            else:
                discard_pnts_count += 1

    print("Discarded points :", discard_pnts_count)

    idv_keypoints = SIFT.remove_duplicate_keypnts(idv_keypoints)
    for idv_keypoint in idv_keypoints:
        idv_keypoint.pt[0] /= 2.0
        idv_keypoint.pt[1] /= 2.0
        idv_keypoint.size /= 2.0

    descrptrs = SIFT.get_descriptors(idv_keypoints, magOctaves, sitaOctaves)
    cv_keypoints = [kpnt.to_CV_keypoint() for kpnt in idv_keypoints]
    
    # 嘗試利用opencv算出我們的keypoint descriptor
    #sift = cv2.xfeatures2d.SIFT_create()
    #kps, descrptrs = sift.compute(img_content['data'], cv_keypoints)

    img_content['keypoints'] = idv_keypoints
    img_content['descriptors'] = descrptrs
    #utils.imshow_plt(descrptrs)
    
    # im_key = cv2.drawKeypoints(img_content['data'], cv_keypoints, np.array([]), (255, 0, 0))
    # im_key = cv2.drawKeypoints(imgs, cv_keypoints, np.array([]), (255, 0, 0))

    #utils.imshow_plt(im_key)

def cylindricalWarp(img, f):
    h, w = img.shape[:2]
    result = np.zeros((h, w, 3))
    x_c = w / 2
    y_c = h / 2

    for y in range(h):
        for x in range(w):
            x_b = f * math.atan((x - x_c) / f) + x_c
            y_b = f * (y - y_c) / math.sqrt((x - x_c) * (x - x_c) + f * f) + y_c
            result[round(y_b), round(x_b), :] = img[y, x, :]
    return result


if __name__ == '__main__':
    total_time = 0
    start = time.time()
    parser = utils.get_parser()
    args = parser.parse_args()
    
    img_contents = []
    '''
    [
        {
            "filepath": FILEPATH,
            "data": OPENCV_IMG,
            "offset": {"x": int, "y": int}, # 1 means shift left or top, -1 means shift right or down
            "exif": dict
        }
    ]
    '''

    for file in os.listdir(args.input_dir):
        file_lower = file.lower()
        if file_lower.endswith('.jpg') or file_lower.endswith('.png'):
            img_filePath = os.path.join(args.input_dir, file)
            img = utils.imgImportFromPil(img_filePath)

            if not args.degree == 0:
                (h, w) = img.shape[:2]
                (cX, cY) = (w // 2, h // 2)
                # rotate our image by -90 degrees around the image
                img = imutils.rotate_bound(img, args.degree)
                img = cv2.resize(img, (w, h))
                #cv2.imshow("Rotated by %s Degrees" % str(args.degree), img)
                #cv2.waitKey(0)

            img_contents.append({
                'filepath': img_filePath,
                'data': img,
                'keypoints': None,
                'descriptors': None,
                'exif': utils.getExifFromPath(img_filePath)
            })

    # Map to cylinder
    for i in range(len(img_contents)):
        h, w = img_contents[i]['data'].shape[:2]
        if 'focal_len' in img_contents[i]['exif']:
            f = img_contents[i]['exif']['focal_len'] * 100
        else:
            f = args.focal_length
        result = cylindricalWarp(img_contents[i]['data'], f)
        result = imageStitching.removeBlackBorderLR(result)
        img_contents[i]['data'] = result.copy().astype(np.uint8)
        #plt.imshow(img_contents[i]['data'])
        #plt.show()

    # 計算鄰近圖之間的最佳offset
    offsets = []
    GetKeyPointAndDescriptor(img_contents[0])
    previousKps = img_contents[0]['keypoints']
    previousDscrts = img_contents[0]['descriptors']

    for i in range(1, len(img_contents)):
        leftImg = img_contents[i - 1]
        rightImg = img_contents[i]

        keypointPairs = []
        
        GetKeyPointAndDescriptor(rightImg)
        kps1 = previousKps
        kps2 = rightImg['keypoints']
        dscrts1 = previousDscrts
        dscrts2 = rightImg['descriptors']
        
        print("keypoint matching")

        # kd-tree
        tree = spatial.KDTree(dscrts2)

        # Keypoint match
        progress = tqdm(total=len(kps1))
        for j in range(len(kps1)):
            firstKP = kps1[j].pt
            targetDescriptor = dscrts1[j]
            
            # query from kd-tree
            distance, resultIdx = tree.query(targetDescriptor, 2)
            
            if distance[0] / distance[1] <= args.match_ratio:
                secondKP = kps2[resultIdx[0]].pt
                keypointPairs.append([firstKP, secondKP])
            
            progress.update(1)
        
        progress.close()
        print("match count:" + str(len(keypointPairs)))
        keypointPairs = np.array(keypointPairs)
        total_img = np.concatenate((leftImg['data'], rightImg['data']), axis=1)
        # Good matches
        if args.plot == 'True':
            utils.plot_matches(keypointPairs, total_img, leftImg['data'].shape[1])

        bestTranslate = imageStitching.compute_best_Translate(keypointPairs)
        offsets.append(bestTranslate)

        previousKps = kps2
        previousDscrts = dscrts2

    # Image stitching
    leftImg = img_contents[0]
    offsets = np.array(offsets)
    offset = np.zeros((2))

    (x, y) = np.sum(offsets, axis=0)
    alpha = y / x
    for i in range(1, len(img_contents)):
        print("stitching %d" % i)
        offset = offset + offsets[i - 1]
        rightImg = img_contents[i]
        result = imageStitching.warp(leftImg['data'], rightImg['data'], offset).astype(np.uint8)
        leftImg['data'] = result
        plt.imsave(os.path.join(args.input_dir, "result%s.jpg" % str(i)), result)

    # 處理每次Stitching產生的Y軸下移問題
    no_drift_result = np.zeros(leftImg['data'].shape, dtype=np.uint8)
    h, w, d = leftImg['data'].shape
    for y in range(h):
        for x in range(w):
            y_p = round(y - alpha * x)
            if y_p >= 0 and y_p < h:
                no_drift_result[y_p, x, :] = leftImg['data'][y, x, :]
    no_drift_result = imageStitching.removeBlackBorderTB(no_drift_result)
    plt.imsave(os.path.join(args.input_dir, "no_drift_result.jpg"), no_drift_result)
        