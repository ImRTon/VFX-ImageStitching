import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_pillow = np.array(Image.open('test3/no_drift_result.jpg'))
#im_bgr = cv2.cvtColor(im_pillow, cv2.COLOR_BGR2RGB)

cv2.imwrite('test3/no_drift_result_RGB.jpg', im_pillow)