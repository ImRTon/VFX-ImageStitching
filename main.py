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
    img_content['keypoints'] = idv_keypoints
    img_content['descriptors'] = descrptrs
    #utils.imshow_plt(descrptrs)
    cv_keypoints = [kpnt.to_CV_keypoint() for kpnt in idv_keypoints]

    im_key = cv2.drawKeypoints(img_content['data'], cv_keypoints, np.array([]), (255, 0, 0))
    # im_key = cv2.drawKeypoints(imgs, cv_keypoints, np.array([]), (255, 0, 0))

    utils.imshow_plt(im_key)


def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

def cylindricalWarpV2(img, f):
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

    focals = [705.102, 704.537, 704.847, 704.676, 704.289, 703.895, 704.696, 704.325, 703.794, 
704.696, 705.327, 705.645, 706.587, 706.645, 705.849, 706.286, 704.916, 705.576]
    # map to cylinder
    for i in range(len(img_contents)):
        h, w = img_contents[i]['data'].shape[:2]
        f = focals[i]
        result = cylindricalWarpV2(img_contents[i]['data'], f)
        result = imageStitching.removeBlackBorder(result)
        img_contents[i]['data'] = result.copy().astype(np.uint8)
        #plt.imshow(img_contents[i]['data'])
        #plt.show()

    # 計算鄰近圖之間的最佳offset
    offsets = []
    for i in range(1, len(img_contents)):
        leftImg = img_contents[i - 1]
        rightImg = img_contents[i]

        # Keypoint matching
        keypointPairs = []

        # 使用cv2版本的SIFT測試
        sift = cv2.xfeatures2d.SIFT_create()
        kps1, dscrts1 = sift.detectAndCompute(leftImg['data'], None)
        kps2, dscrts2 = sift.detectAndCompute(rightImg['data'], None)
        #GetKeyPointAndDescriptor(leftImg)
        #GetKeyPointAndDescriptor(rightImg)
        #kps1 = leftImg['keypoints']
        #kps2 = rightImg['keypoints']
        #dscrts1 = leftImg['descriptors']
        #dscrts2 = rightImg['descriptors']
        print("keypoint matching")       

        progress = tqdm(total=len(kps1))
        for j in range(len(kps1)):
            firstKP = kps1[j].pt
            targetDescriptor = dscrts1[j]
            
            # kd-tree
            tree = spatial.KDTree(dscrts2)
            distance, resultIdx = tree.query(targetDescriptor, 2)
            
            if distance[0] / distance[1] <= 0.6:
                secondKP = kps2[resultIdx[0]].pt
                keypointPairs.append([firstKP, secondKP])
            
            progress.update(1)
        
        progress.close()
        print("match count:" + str(len(keypointPairs)))
        keypointPairs = np.array(keypointPairs)
        # Good matches
        #utils.plot_matches(keypointPairs, total_img, leftImg['data'].shape[1])

        bestHomography = imageStitching.compute_best_Homography(keypointPairs)
        offsets.append(bestHomography)

    # Image stitching
    leftImg = img_contents[0]
    offset = np.zeros((2))
    for i in range(1, len(img_contents)):
        print("stitching %d" % i)
        offset = offset + offsets[i - 1]
        leftImg = img_contents[i - 1]
        rightImg = img_contents[i]
        result = imageStitching.warp(leftImg['data'], rightImg['data'], offset).astype(np.uint8)
        leftImg['data'] = result
        plt.imsave("test2/result%s.jpg" % str(i), result)
        