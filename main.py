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
            "brightness": INT
        }
    ]
    '''

    for file in os.listdir(args.input_dir):
        file_lower = file.lower()
        if file_lower.endswith('.jpg') or file_lower.endswith('.png'):
            img_filePath = os.path.join(args.input_dir, file)
            img = utils.imgImportFromPil(img_filePath)

            img_contents.append({
                'filepath': img_filePath,
                'data': img,
                'keypoints': None,
                'descriptors': None,
            })

    for img_content in img_contents:
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
        img_content['keypoints'] = idv_keypoints
        img_content['descriptors'] = descrptrs
        #utils.imshow_plt(descrptrs)
        cv_keypoints = [kpnt.to_CV_keypoint() for kpnt in idv_keypoints]

        im_key = cv2.drawKeypoints(img_content['data'], cv_keypoints, np.array([]), (255, 0, 0))
        # im_key = cv2.drawKeypoints(imgs, cv_keypoints, np.array([]), (255, 0, 0))

        utils.imshow_plt(im_key)

    # 測試stitching
    firstImg = img_contents[0]
    secondImg = img_contents[1]

    # Keypoint matching
    keypointPairs = []
    # 使用cv2版本的SIFT測試
    
    sift = cv2.xfeatures2d.SIFT_create()
    kps1, dscrts1 = sift.detectAndCompute(firstImg['data'], None)
    kps2, dscrts2 = sift.detectAndCompute(secondImg['data'], None)
    print("keypoint matching")
    progress = tqdm(total=len(kps1))
    for i in range(len(kps1)):
        firstKP = kps1[i].pt
        targetDescriptor = dscrts1[i]
        minDistance = 100000.0
        secondMinDis = 100000.0
        bestMatchIdx = -1

        for j in range(len(kps2)):
            descriptor = dscrts2[j]
            # Calculate Euclidean distance
            dist = np.linalg.norm(targetDescriptor - descriptor)
            if dist < minDistance:
                secondMinDis = minDistance
                minDistance = dist
                bestMatchIdx = j
        
        if minDistance / secondMinDis <= 0.2:
            secondKP = kps2[bestMatchIdx].pt
            keypointPairs.append([firstKP, secondKP])
        progress.update(1)
    
    progress.close()
    print("match count:" + str(len(keypointPairs)))
    total_img = np.concatenate((firstImg['data'], secondImg['data']), axis=1)
    keypointPairs = np.array(keypointPairs)
    # Good matches
    utils.plot_matches(keypointPairs, total_img)
    
    '''
    print("keypoint matching")
    progress = tqdm(total=len(firstImg['keypoints']))
    for i in range(len(firstImg['keypoints'])):
        firstKP = firstImg['keypoints'][i].pt
        targetDescriptor = firstImg['descriptors'][i]
        minDistance = 100000.0
        secondMinDis = 100000.0
        bestMatchIdx = -1

        for j in range(len(secondImg['keypoints'])):
            descriptor = secondImg['descriptors'][j]
            # Calculate Euclidean distance
            dist = np.linalg.norm(targetDescriptor - descriptor)
            if dist < minDistance:
                secondMinDis = minDistance
                minDistance = dist
                bestMatchIdx = j
        
        if minDistance / secondMinDis <= 0.2:
            secondKP = secondImg['keypoints'][bestMatchIdx].pt
            keypointPairs.append([firstKP, secondKP])
        progress.update(1)
    
    progress.close()
    print("match count:" + str(len(keypointPairs)))
    total_img = np.concatenate((firstImg['data'], secondImg['data']), axis=1)
    keypointPairs = np.array(keypointPairs)
    # Good matches
    utils.plot_matches(keypointPairs, total_img)
    '''

    # Image stitching