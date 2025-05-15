import torch.nn as nn
import torch.nn.functional as F
import torch


class LowRankLinear(nn.Module):
    
    def build_layer(self, input_dim, output_dim, layer: nn.Linear = None):
        if layer is None:
            layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        u, sigma, v = torch.linalg.svd(
            layer.weight, full_matrices=False
        )
        return u, sigma, v, layer.bias


    def __init__(self, input_dim, output_dim, layer = None):
        super().__init__()
        u, sigma, v, bias = self.build_layer(input_dim, output_dim, layer)
        self.u = nn.Parameter(u)
        self.v = nn.Parameter(v)
        self.sigma = nn.Parameter(sigma)
        self.bias = nn.Parameter(bias)
        self.register_parameter("U", self.u)
        self.register_parameter("V", self.v)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("bias", self.bias)
        self.rank = min(self.u.shape[0], self.u.shape[1])

    def orthogonal_loss(self):
        u_square = self.u.T @ self.u
        u_square -= torch.eye(u_square.shape[0], device=u_square.device)
        v_square = self.v.T @ self.v
        v_square -= torch.eye(v_square.shape[0], device=u_square.device)
        return 1/self.rank * (u_square.square().sum() + v_square.square().sum())
    
    def hoyer_loss(self):
        l_1 = self.sigma.abs().sum()
        l_2 = torch.norm(self.sigma)
        return l_1/l_2
    
    def forward(self, x):
        x = F.linear(x, self.v)
        x = F.linear(x, torch.diag(self.sigma))
        x = F.linear(x, self.u, bias=self.bias)
        return x

class LowRankConv1x1(LowRankLinear):

    # assume 1x1 and no bias
    def build_layer(self, input_dim, output_dim, layer = None):
        if layer is None:
            layer = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, bias=True)
        weight = layer.weight.squeeze()
        u, sigma, v = torch.linalg.svd(
            weight, full_matrices=False
        )
        return u, sigma, v, layer.bias

    def __init__(self, in_channels, out_channels, layer: nn.Conv2d = None):
        super().__init__(in_channels, out_channels, layer)


    def forward(self, x):
        x = F.conv2d(input=x, weight=self.v[:,:,None,None])
        x = F.conv2d(input=x, weight=torch.diag(self.sigma)[:,:,None,None])
        bias = self.bias if self.bias.numel() != 0 else None
        x = F.conv2d(input=x, weight=self.u[:,:,None,None], bias=bias)
        return x

if __name__ == "__main__":
    # layer1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=False)
    # layer2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, bias=False)
    # layer_mult = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False)
    # layer1.weight.data = torch.ones((128,256,1,1))
    # layer2.weight.data = torch.ones((256,128,1,1))
    # x = torch.ones((256,5,5))

    # y = layer2(layer1(x))
    # layer_mult.weight.data = (torch.ones((256,128)) @ torch.ones((128, 256)))[:,:,None,None]
    # z = layer_mult(x)
    # print(torch.isclose(y,z).all())
    layer = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1, bias=False)
    layer_lr = LowRankConv1x1(in_channels=21, out_channels=6, layer=layer)
    x = torch.ones((12, 32, 32))
    print(torch.isclose(layer(x), layer_lr(x), rtol=1e-4).all())

