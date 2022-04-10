from random import random
from cv2 import threshold
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_homography(keypointPairs):
    # Construct matrix A
    # Ref: https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
    A = []
    for i in range(len(keypointPairs)):
        origin = keypointPairs[i][1]
        target = keypointPairs[i][0]
        A.append([-origin[0], -origin[1], -1, 0, 0, 0, target[0]*origin[0], target[0]*origin[1], target[0]])
        A.append([0, 0, 0, -origin[0], -origin[1], -1, target[1]*origin[0], target[1]*origin[1], target[1]])

    # solving Ah = 0 using SVD
    u, s, v = np.linalg.svd(A)
    Homography = np.reshape(v[-1], (3, 3))
    Homography = Homography / Homography[2, 2]

    return Homography

def compute_best_Homography(keypointPairs):
    '''
        keypointPairs: npArray of matched keypoint pairs
    '''
    # Compute best Homography matrix using RANSAC algorithm
    # 至少需要4組關鍵點pair才能解出一個homography(因為有8個自由度)
    iteration = 1000
    # 用來判斷是否為inlier的閾值
    threshold = 0.5
    sampleNum = np.shape(keypointPairs)[0]
    maxInlier = 0
    bestHomography = None
    print("compute best homography")
    progress = tqdm(total=iteration)

    for iter in range(iteration):
        # pick 4 random number
        subSampleIndices = random.sample(range(sampleNum), 4)
        Homography = compute_homography(keypointPairs[subSampleIndices])

        # compute total inlier for this homography
        inlierNum = 0
        for i in range(sampleNum):
            if i not in subSampleIndices:
                origin = keypointPairs[i][1]
                target = keypointPairs[i][0]
                origin = np.append(origin, 1)
                target = np.append(target, 1)
                dstPoint = Homography @ origin.T

                # prevent to divide by 0 or small value
                if dstPoint[2] <= 1e-8: 
                    continue
                dstPoint = dstPoint / dstPoint[2]
                # calculate Euclidean distance
                if np.linalg.norm(dstPoint - target) < threshold:
                    inlierNum = inlierNum + 1

            if inlierNum > maxInlier:
                maxInlier = inlierNum
                bestHomography = Homography

        progress.update(1)
    progress.close()
    return bestHomography

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
    for i in range(leftHeight):
        for j in range(leftWidth):
            if np.count_nonzero(leftImg[i, j]) > 0:
                img_left_mask[i, j] = 1
    for i in range(rightHeight):
        for j in range(rightWidth):
            if np.count_nonzero(rightImg[i, j]) > 0:
                img_right_mask[i, j] = 1
    
    # find the overlap mask(overlap region of two image)
    overlap_mask = np.zeros((rightHeight, rightWidth), dtype="int")
    for i in range(rightHeight):
        for j in range(rightWidth):
            if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                overlap_mask[i, j] = 1
    
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
            if (np.count_nonzero(overlap_mask[i, j]) > 0):
                blendedImage[i, j] = alpha_mask[i, j] * leftImg[i, j] + (1 - alpha_mask[i, j]) * rightImg[i, j]

    return blendedImage

def linear_blending_np(leftImg, rightImg):
    '''
        針對重疊區域做線性插值
        基本上rightImg傳入的是mapping過後的stitch image
    '''
    leftHeight, leftWidth = leftImg.shape[:2]
    rightHeight, rightWidth = rightImg.shape[:2]
    img_left_mask = np.zeros((rightHeight, rightWidth), dtype="int")
    img_right_mask = np.zeros((rightHeight, rightWidth), dtype="int")
    
    # find the left image and right image mask region(None zero pixels)
    img_left_mask[:, :] = np.count_nonzero(leftImg[:, :]) > 0
    img_right_mask[:, :] = np.count_nonzero(rightImg[:, :]) > 0
    
    # find the overlap mask(overlap region of two image)
    overlap_mask = np.logical_and(img_left_mask, img_right_mask).astype(int)
    
    # compute the alpha mask to linear blending the overlap region
    # 1到0的線性插值
    alpha_mask = np.zeros((rightHeight, rightWidth))
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
    blendedImage[:leftHeight, :leftWidth] = np.copy(leftImg)
    # 針對重疊區域做blending
    blendedImage[:, :] = np.where(overlap_mask[:, :], alpha_mask[:, :leftImg.shape[1]] * leftImg[:, :] + (1 - alpha_mask[:, :]) * rightImg[:, :],  blendedImage[:, :])

    return blendedImage

def removeBlackBorder(img):
        '''
            Remove img's the black border 
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1
                
        # bottom to top 
        
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1
        
        return img[:reduced_h, :reduced_w]

def warp(leftImg, rightImg, homography):
    leftHeight, leftWidth = leftImg.shape[:2]
    rightHeight, rightWidth = rightImg.shape[:2]

    # 計算連接之後的圖片大小
    stitchImage = np.zeros((max(leftHeight, rightHeight), leftWidth + rightWidth, 3), dtype=int)
    stitchImage[:leftHeight, :leftWidth] = leftImg
    # 計算反矩陣
    homographyInverse = np.linalg.inv(homography)
    print("warping")
    progress = tqdm(total = stitchImage.shape[0])
    for i in range(stitchImage.shape[0]):
        for j in range(stitchImage.shape[1]):
            pixel = np.array([j, i, 1])
            # 利用反矩陣計算對應到右邊圖片的座標
            rightImagePixel = homographyInverse @ pixel
            rightImagePixel = rightImagePixel / rightImagePixel[2]

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
    stitchImage = removeBlackBorder(stitchImage)
    return stitchImage