import cv2, os
import numpy as np
from tqdm import tqdm

def edge_promoting(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    n = 1
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(root, f))
        gauss_img = cv2.GaussianBlur(rgb_img, (5, 5), 2.5)

        result = np.concatenate((rgb_img, gauss_img), 1)

        cv2.imwrite(os.path.join(save, str(n) + '.png'), result)
        n += 1