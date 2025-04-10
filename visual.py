import numpy as np
import matplotlib.pyplot as plt
depth_image_vis = np.load('/home/aryan-sood/Documents/CIL/ethz-cil-monocular-depth-estimation-2025(1)/train/train/sample_000000_depth.npy')
#print(np.max(depth_image_vis))
depth_image_vis = depth_image_vis

print(np.max(depth_image_vis), np.min(depth_image_vis))
plt.imshow(depth_image_vis)
plt.show()