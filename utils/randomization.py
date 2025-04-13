from PIL import Image
from pathlib import Path
import random
import constants as c
import numpy as np

def random_image():
    png_paths = list(Path(c.DATA_DIR).glob("*.png"))
    img = Image.open(random.sample(png_paths, 1)[0])
    return np.asarray(img)

def random_mask():
    npy_paths = list(Path(c.DATA_DIR).glob("*.npy"))
    mask = np.load(random.sample(npy_paths, 1)[0])
    return mask


if __name__ == '__main__':
    import cv2

    cv2.imshow("sampled_image", np.asarray(random_image()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(random_mask())