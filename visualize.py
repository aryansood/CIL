import torch
from utils.constants import DATA_DIR
from utils.visualization import Visualization
from models import UNetMonocularDepthEstimator
from dataset import RandomizedDataset, DepthDataset
from torchvision.transforms.functional import pil_to_tensor

CHECKPOINT_PATH1= "monocular_depth_estimation/qgboqkvv/checkpoints/checkpoint-epoch=0-valid_silog_loss=0.8199.ckpt"
CHECKPOINT_PATH2="monocular_depth_estimation/qgboqkvv/checkpoints/checkpoint-epoch=3-valid_silog_loss=1.3035.ckpt"

model1 = UNetMonocularDepthEstimator.load_from_checkpoint(CHECKPOINT_PATH1)
model2 = UNetMonocularDepthEstimator.load_from_checkpoint(CHECKPOINT_PATH2)
model1.eval()
model2.eval()
randomizer = RandomizedDataset(DATA_DIR, DepthDataset.TRAIN)
visualizer = Visualization()

for img, target in randomizer.iter(4):
    # press enter to go to next image
    img = pil_to_tensor(img).unsqueeze(0).to(torch.float32)
    visualizer.visualize(ground_thruth=target, model_output=model1(img.cuda()), rgb_image=img)
    visualizer.visualize(ground_thruth=target, model_output=model2(img.cuda()), rgb_image=img)