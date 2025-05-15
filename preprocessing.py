import torch.nn as nn
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import resnet101, ResNet101_Weights
from typing import Callable
from utils.dataset import DepthDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx
from typing import Tuple

class PreprocessingModel(nn.Module):

    def __init__(self, backbone: nn.Module, transforms: Callable):
        super().__init__()
        self.backbone = backbone
        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            xt = self.transforms(x)
            xb = self.backbone(xt)
        return xb

def __get_mobilenetv3():
    model = mobilenet_v3_small(MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return {
        "backbone": nn.Sequential(model.features, model.avgpool), 
        "transforms": MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    }

def __get_resnet():
    model_base = resnet101(ResNet101_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(
        model_base.conv1,
        model_base.bn1,
        model_base.relu,
        model_base.maxpool,
        model_base.layer1,
        model_base.layer2,
        model_base.layer3,
        model_base.layer4,
        model_base.avgpool
    )
    return {
        "backbone": backbone, 
        "transforms": ResNet101_Weights.IMAGENET1K_V2.transforms()
    }


SUPPORTED_MODELS = {
    "mobile_net_v3": PreprocessingModel(**__get_mobilenetv3()),
    "resnet": PreprocessingModel(**__get_resnet()),
}


def get_model(name: str) -> PreprocessingModel:
    return SUPPORTED_MODELS[name]

def get_embeddings(model: PreprocessingModel, 
                   ds: DepthDataset, 
                   device = "cpu", 
                   batch_size: int = 32) -> torch.Tensor:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    embeddings = {}
    for _, item in tqdm(enumerate(dl), total=len(dl), desc="calculating embeddings"):
        rgb_batch, paths = item[0].permute((0,3,1,2)), item[-1]
        
        embedded = model(rgb_batch.to(device))
        for idx_in_batch, path in enumerate(paths):
            embeddings[path] = embedded[idx_in_batch,...].cpu()
    return embeddings

def get_cosine_similarity_matrix(embeddings: dict, device = "cpu") -> Tuple[torch.Tensor, list]:
    sorted_keys = sorted(embeddings.keys())
    embs = []
    for key in sorted_keys:
        embs.append(embeddings[key].squeeze().to(device))
    embs = torch.stack(embs)
    normalized = embs / torch.norm(embs, dim=1).unsqueeze(1)
    cs_matr = (normalized @ normalized.T)
    return cs_matr, sorted_keys

def spectral_clustering(adj_matr: torch.Tensor) -> torch.Tensor:
    laplacian = torch.diag(torch.sum(adj_matr,dim=1)) - adj_matr
    l, v = torch.linalg.eigh(laplacian)
    return l,v


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    # TRAIN_DIR = "/home/alessandro/cil/project/data/ethz-cil-monocular-depth-estimation-2025/train/train"
    # ds = DepthDataset(data_dir=Path(TRAIN_DIR), has_gt = False, augmentations=None)
    # model = get_model("mobile_net_v3")
    # embeddings = get_embeddings(model, ds, "cpu", 48)
    # torch.save(embeddings, "scratch/mobilenet_embeddings.pt")
    embeddings = torch.load("scratch/mobilenet_embeddings.pt")
    cs_matr, sorted_keys = get_cosine_similarity_matrix(embeddings)
    bias = -0.8
    temperature = 0.01
    cs_matr = torch.sigmoid((1/temperature)*(cs_matr+bias))
    l,v  = spectral_clustering(cs_matr)
    torch.save((l,v),"scratch/spectral_decom.pt")

