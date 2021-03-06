import math
from random import random
from cv2 import threshold
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_best_Translate(keypointPairs):
    '''
        Compute best translate vector using RANSAC algorithm
        需要1組關鍵點pair來解
        
        args:
            keypointPairs: npArray of matched keypoint pairs
    '''

    # RANSAC迭代次數
    iteration = 3000
    # 用來判斷是否為inlier的閾值
    threshold = 0.5 
    sampleNum = np.shape(keypointPairs)[0]
    maxInlier = 0
    bestTranslate = keypointPairs[0][1] - keypointPairs[0][0]
    print("compute best homography")
    progress = tqdm(total=iteration)

    for iter in range(iteration):
        subSampleIndices = random.sample(range(sampleNum), 1)
        pair = keypointPairs[subSampleIndices[0]]
        shift = pair[1] - pair[0]
        # compute total inlier for this homography
        inlierNum = 0
        for i in range(sampleNum):
            if i not in subSampleIndices:
                origin = keypointPairs[i][1]
                target = keypointPairs[i][0]
                dstPoint = origin - shift

                # calculate Euclidean distance
                if np.linalg.norm(dstPoint - target) < threshold:
                    inlierNum = inlierNum + 1

            if inlierNum > maxInlier:
                maxInlier = inlierNum
                bestTranslate = shift

        progress.update(1)
    progress.close()
    return bestTranslate

def linear_blending(leftImg, rightImg):
    '''
        針對重疊區域做線性插值
        基本上rightImg傳入的是mapping過後的stitch image
    '''
    leftHeight, leftWidth = leftImg.shape[:2]
    rightHeight, rightWidth = rightImg.shape[:2]
    img_left_mask = np.zeros((rightHeight, rightWidth), dtype="int")
    img_right_mask = np.zeros((rightHeight, rightWidth), dtype="int")
    
    # find the left image and right image mask region(None zero pixels)
    img_left_mask[:leftHeight, :leftWidth] = np.where(np.count_nonzero(leftImg[:, :]) > 0, 1, 0)
    non_black_pixels_mask = np.any(rightImg != [0, 0, 0], axis=-1)
    img_right_mask[non_black_pixels_mask] = 1

    overlap_mask = np.logical_and(img_left_mask, img_right_mask).astype(np.uint8)

    # compute the alpha mask to linear blending the overlap region
    alpha_mask = np.zeros((rightHeight, rightWidth)) # alpha value depend on left image
    for i in range(rightHeight): 
        minIdx = maxIdx = -1
        for j in range(rightWidth):
            if (overlap_mask[i, j] == 1 and minIdx == -1):
                minIdx = j
            if (overlap_mask[i, j] == 1):
                maxIdx = j
        
        if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
            continue
            
        decrease_step = 1 / (maxIdx - minIdx)
        for j in range(minIdx, maxIdx + 1):
            alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
    
    blendedImage = np.copy(rightImg)
    # 先複製left image過來
    blendedImage[:leftHeight, :leftWidth] = np.copy(leftImg)
    # 針對重疊區域做blending
    for i in range(rightHeight):
        for j in range(rightWidth):
            if (overlap_mask[i, j] == 1):
                blendedImage[i, j] = alpha_mask[i, j] * leftImg[i, j] + (1 - alpha_mask[i, j]) * rightImg[i, j]

    return blendedImage

def removeBlackBorderLR(img):
        '''
            Remove img's left and right black border 
        '''
        h, w, d = img.shape
        #left limit
        for i in range(w):
            if np.sum(img[:,i,:]) > 0:
                break
        #right limit
        for j in range(w-1, 0, -1):
            if np.sum(img[:,j,:]) > 0:
                break

        return img[:,i:j+1,:].copy()

def removeBlackBorderTB(img):
        '''
            Remove img's top and bottom black border 
        '''
        h, w, d = img.shape
        #left limit
        for i in range(h):
            if np.sum(img[i,:,:]) > 0:
                break
        #right limit
        for j in range(h-1, 0, -1):
            if np.sum(img[j,:,:]) > 0:
                break

        return img[i:j+1,:,:].copy()

def warp(leftImg, rightImg, offset):
    leftHeight, leftWidth = leftImg.shape[:2]
    rightHeight, rightWidth = rightImg.shape[:2]

    # 計算連接之後的圖片大小
    stitchImage = np.zeros((round(max(leftHeight, rightHeight + abs(offset[1]))), leftWidth + rightWidth, 3), dtype=int)
    # stitchImage[:leftHeight, :leftWidth] = leftImg

    print("warping")
    progress = tqdm(total = stitchImage.shape[0])
    for i in range(stitchImage.shape[0]):
        for j in range(leftWidth + math.floor(offset[0]), stitchImage.shape[1]):
            pixel = np.array([j, i])
            rightImagePixel = pixel + offset
            x, y = int(round(rightImagePixel[0])), int(round(rightImagePixel[1]))

            # 如果超出圖片的範圍，代表無法找到對應的pxiel
            if (y < 0 or y >= rightHeight or x < 0 or x >= rightWidth):
                continue

            stitchImage[i, j] = rightImg[y, x]
        progress.update(1)
    progress.close()
    #plt.imshow(stitchImage)
    #plt.show()
    print("blending")
    stitchImage = linear_blending(leftImg, stitchImage)
    print("remove black border")
    stitchImage = removeBlackBorderLR(stitchImage)
    return stitchImage