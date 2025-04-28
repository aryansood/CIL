import cv2
import torch
import numpy as np
from PIL import Image
from time import sleep
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython.display import clear_output

def numpify(mask):
    if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
    if isinstance(mask, Image.Image):  mask = np.uint8(mask)
    if len(mask.shape) == 2:           mask = np.expand_dims(mask, 0)
    if len(mask.shape) == 4:           mask = mask.squeeze(0)
    if mask.shape[0] in (1,3):         mask = np.moveaxis(mask, 0, -1)
    return mask

def normalize_depth_image(mask):
    return cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)

def analyze_errors(ground_truth, output):
    diff = output - ground_truth
    abs_diff = np.abs(diff)

    print("=== Signed Error Statistics ===")
    print(f"Mean Error: {np.mean(diff):.4f}")
    print(f"Median Error: {np.median(diff):.4f}")
    print(f"Standard Deviation: {np.std(diff):.4f}")
    print(f"Min Error: {np.min(diff):.4f}")
    print(f"Max Error: {np.max(diff):.4f}")
    
    print("\n=== Absolute Error Statistics ===")
    print(f"Mean Absolute Error: {np.mean(abs_diff):.4f}")
    print(f"Median Absolute Error: {np.median(abs_diff):.4f}")
    print(f"Standard Deviation: {np.std(abs_diff):.4f}")
    print(f"Min Absolute Error: {np.min(abs_diff):.4f}")
    print(f"Max Absolute Error: {np.max(abs_diff):.4f}")
    print(f"25th Percentile: {np.percentile(abs_diff, 25):.4f}")
    print(f"75th Percentile: {np.percentile(abs_diff, 75):.4f}")
    print(f"95th Percentile: {np.percentile(abs_diff, 95):.4f}\n\n")



class Visualization:
    """
    Checks wheter you are running the code on terminal or through jupyter notebook
    and prints the images to the screen.
    """
    JUPYTER_EXECUTION = 'jupyter'
    TERMINAL_EXECUTION = 'terminal'

    def __init__(self, model=None, dataset=None):
        self.target = None
        self.output = None
        self.rgb_image = None
        
        self.model = model
        self.dataset = deepcopy(dataset)
        try:
            __IPYTHON__
            self._session = Visualization.JUPYTER_EXECUTION
        except NameError:
            self._session = Visualization.TERMINAL_EXECUTION


    def list_masks(self):
        return [('ground_thruth', self.target), ('model_output', self.output), ('rgb_image', self.rgb_image)]

    def visualize(self, ground_thruth=None, model_output=None, rgb_image=None, sleeptime=5):
        if rgb_image     is not None:   self.rgb_image = numpify(rgb_image)
        if ground_thruth is not None:   self.target    = numpify(ground_thruth)
        if   model_output  is not None: self.output    = numpify(model_output)
        elif self.model    is not None: self.output    = numpify(self.model(rgb_image)) if rgb_image is not None else None
        
        if (self.target is not None) and (self.output is not None):
            analyze_errors(self.target, self.output)

        if (self._session == Visualization.JUPYTER_EXECUTION):  self._visualize_jupyter(sleeptime)
        if (self._session == Visualization.TERMINAL_EXECUTION): self._visualize_cv2(sleeptime)


    def _visualize_jupyter(self, sleeptime):
        labels, imgs = zip(*[(x,y) for x,y in [("ground_thruth", self.target), ("model_output", self.output), ("rgb_image", self.rgb_image)] if y is not None])
        n_images = len(imgs)
        plt.figure(figsize=(5*n_images, 5))
                        
        for i, (label, img) in enumerate(list(zip(labels, imgs)), start=1):
            plt.subplot(1, n_images, i)
            plt.imshow(img)
            plt.title(label)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

        sleep(sleeptime)
        clear_output()

    def _visualize_cv2(self, sleeptime):
        
        normalized_target = cv2.cvtColor(np.uint8(normalize_depth_image(self.target)), cv2.COLOR_GRAY2BGR) if self.target is not None else None
        normalized_output = cv2.cvtColor(np.uint8(normalize_depth_image(self.output)), cv2.COLOR_GRAY2BGR) if self.output is not None else None
        rgb_img = cv2.cvtColor(np.uint8(normalize_depth_image(self.rgb_image)), cv2.COLOR_RGB2BGR)                   if self.rgb_image is not None else None

        label, img = zip(*[(x,y) for x,y in [("ground_thruth", normalized_target), ("model_output", normalized_output), ("rgb_image", rgb_img)] if y is not None])
        label = " | ".join(label)
        img = np.hstack(img)
        
        cv2.imshow(label, img)
        cv2.waitKey(sleeptime * 1000)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    # To run this code, you will need to temporarily move dataset.py into the utils module
    from dataset import RandomizedDataset, DepthDataset
    from constants import DATA_DIR

    randomizer = RandomizedDataset(DATA_DIR, DepthDataset.TRAIN)

    for img, target in randomizer.iter(4):
        Visualization().visualize(ground_thruth=target, model_output=randomizer.random_mask(), rgb_image=img)