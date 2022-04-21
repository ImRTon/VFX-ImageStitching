# VFX Project2 Image Stitching

## 演算法
整體架構由 `main.py` 作為主要程式執行的區塊，而依據不同功能切割成 `utils.py`、`SIFT.py`、`imageStitching.py` 三個檔案。
程式執行首先會透過 argparse 取出我們所需要的參數，如圖片資料夾、用於配對Keypoint的Ratio、圖片的焦距等。

```python
def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('-i', '--input_dir', default='test2', type=str, help='Folder of input images.')
    parser.add_argument('-p', '--plot', default='False', type=str, help='Whether to plot result or not.')
    parser.add_argument('-r', '--match_ratio', default=0.6, type=float, help='Ratio for keypoint matching.')
    parser.add_argument('-f', '--focal_length', default=705, type=float, help='focal length of image.')
    return parser
```

### Cylinder Mapping
為了減少後面warping產生嚴重的影像扭曲，首先會將每張輸入的圖片投影到圓柱的座標系。
根據以下公式進行轉換:

$ x' = s * \theta = s * \arctan{\frac{x}{f}} $

$ y' = s * h = s * \frac{y}{\sqrt{x^2 + f^2}} $ 

其中f是圖片的焦距。

### Keypoint matching
透過前面的SIFT算出了每張圖的關鍵點座標以及相對應的128維的Feature Descriptor向量後。
這邊利用暴力法去計算兩張圖之間對應的Keypoint，我們首先將圖2的所有Descriptor向量儲存在KD-Tree當中。接著我們Query出目前檢查的這個圖1的keypoint在kd-tree中距離最短與第二短的keypoint，並且計算其Ratio，若低於一定值，我們就會將最短距離的Pair視為合法的Keypoint pair。

```python
for j in range(len(kps1)):
    firstKP = kps1[j].pt
    targetDescriptor = dscrts1[j]
    
    # kd-tree
    tree = spatial.KDTree(dscrts2)
    distance, resultIdx = tree.query(targetDescriptor, 2)
    
    if distance[0] / distance[1] <= args.match_ratio:
        secondKP = kps2[resultIdx[0]].pt
        keypointPairs.append([firstKP, secondKP])
```

### Find best translate
接這利用這些配對好的Keypoint pairs，透過RANSAC演算法找出最佳的位移向量。

RANSAC演算法:

1. 每次取出一組keypoint pair，計算兩個點之間的位移

2. 接著利用這個位移去計算其他pair的inlier，當一個pair內的其中一個點經過位移之後，與另一個點的距離在某個閾值內(我們設0.5)，就會視為inlier。

3. 經過多次迭代後(我們設3000次)，最後取出inlier最多的位移量作為最佳的位移向量。

### Stitching
將右邊的圖片利用前個步驟計算的最佳位移向量，進行位移之後，與左側圖片合併

### blending
針對左右圖片重疊的部分，透過線性的Alpha blending，避免出現明顯的接合邊界。

也就是逐行檢查每個row，針對重疊的部分，越左邊的pxiel，會傾向於左圖的顏色，越往右越傾向於右邊的顏色。

```python
for i in range(rightHeight):
    for j in range(rightWidth):
        if (overlap_mask[i, j] == 1):
            blendedImage[i, j] = alpha_mask[i, j] * leftImg[i, j] + (1 - alpha_mask[i, j]) * rightImg[i, j]
```

### Remove black board
圖片相接之後，周圍會產生黑色邊框。在這個步驟我們會由左而右，再由右而左檢查每個column，如果該column的所有pixel都是黑的話就會移除該column。

### Remove drift
由於每次的相接都會產生部分的y軸位移，當相接的圖片夠多時，就會造成第一張圖與最後一張圖產生嚴重的Y軸下移。

![](https://imgur.com/67Tjuut.jpg)

因此我們會將這段Y軸位移平均分散給每張圖的每個pixel，透過每個pixel些微的上移，來抵銷整體panorama的向下趨勢。

![](https://imgur.com/aExnNXc.jpg)