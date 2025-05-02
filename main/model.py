import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath
from nets.convnext_bn import ConvNeXt_BN
from config import cfg

class DeConv(nn.Sequential):
    def __init__(self, inplanes, planes, upscale_factor=2, kernel_size = 3, up = True):
        super().__init__()
        size = kernel_size
        pad = 1
        if kernel_size == 7:
            pad = 3
        elif kernel_size == 5:
            pad = 2
        else:
            pad = 1
        self.dwconv = nn.Conv2d(inplanes, inplanes, kernel_size=size, stride=1, padding=pad, groups=inplanes)
        self.norm = nn.BatchNorm2d(inplanes)
        self.pwconv = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=upscale_factor) if up else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = self.upsample1(x)

        return x
        

class HeadNet(nn.Module):

    def __init__(self, joint_num, in_channel):
        self.inplanes = in_channel # 2048, 768
        super(HeadNet, self).__init__()

        self.depth_dim = cfg.depth_dim
        self.output_shape = cfg.output_shape
        self.depth = cfg.depth 
        self.deconv_layers_1 = DeConv(inplanes=self.inplanes,planes=self.depth, kernel_size = 3)
        self.deconv_layers_2 = DeConv(inplanes=self.depth, planes=self.depth, kernel_size = 3)
        self.deconv_layers_3 = DeConv(inplanes=self.depth, planes=self.depth, kernel_size = 3, up = False)
        self.final_layer = nn.Conv2d(
            in_channels=self.depth,
            out_channels=joint_num * self.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.deconv_layers_1(x)
        x = self.deconv_layers_2(x)
        x = self.deconv_layers_3(x)
        x = self.final_layer(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def soft_argmax(heatmaps: torch.Tensor,
                joint_num: int,
                depth_dim: int,
                output_shape: tuple[int,int]) -> torch.Tensor:
    B = heatmaps.shape[0]
    if heatmaps.dim() == 4:
        _,_,H,W = heatmaps.shape
    else:
        raise ValueError(f"Heatmaps should be 4D tensor, but got {heatmaps.dim()}D tensor")
    
    H_out, W_out = output_shape if (H != output_shape[0] or W != output_shape[1]) else (H, W)
    heat = heatmaps.view(B, joint_num, depth_dim, H_out, W_out)
    
    accu_x = heat.sum(dim=(2,3))
    accu_y = heat.sum(dim=(2,4))
    accu_z = heat.sum(dim=(3,4))

    device = heatmaps.device
    accu_x = (accu_x * torch.arange(W_out, device = device)[None,None,:]).sum(dim=2, keepdim=True)
    accu_y = (accu_y * torch.arange(H_out, device = device)[None,None,:]).sum(dim=2, keepdim=True)
    accu_z = (accu_z * torch.arange(depth_dim, device = device)[None,None,:]).sum(dim=2, keepdim=True)

    coord_out = torch.cat([accu_x, accu_y, accu_z], dim=2)

    return coord_out

class ConvNeXtPose(nn.Module):
    def __init__(self, backbone,joint_num, head, cfg):
        super(ConvNeXtPose, self).__init__()
        self.backbone = backbone
        self.head = head
        self.joint_num = joint_num
        self.depth_dim = cfg.depth_dim
        self.output_shape = (cfg.output_shape[0], cfg.output_shape[1])
        # self.loss = nn.MSELoss()

    def forward(self, input_img):
        hm= self.backbone(input_img)
        if self.head is not None:
            hm = self.head(hm)
        coord = soft_argmax(hm, self.joint_num, self.depth_dim, self.output_shape)
        
        return coord
    @torch.jit.ignore
    def compute_loss(self, 
                     coord: torch.Tensor, 
                     target_coord: torch.Tensor, 
                     target_vis: torch.Tensor, 
                     target_have_depth: torch.Tensor) -> torch.Tensor:
        loss_coord = torch.abs(coord - target_coord) * target_vis
        loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth) / 3.0
        return loss_coord
def get_pose_net(cfg, is_train, joint_num):
    drop_rate = 0
    if is_train:
        drop_rate = 0.1
    backbone = ConvNeXt_BN(depths=cfg.backbone_cfg[0], dims=cfg.backbone_cfg[1],drop_path_rate=drop_rate) 
    head_net = HeadNet(joint_num, in_channel = cfg.backbone_cfg[1][-1])

    model = ConvNeXtPose(backbone, joint_num, head =head_net, cfg = cfg)
    return model

