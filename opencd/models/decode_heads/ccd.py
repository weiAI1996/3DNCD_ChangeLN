# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, Sequential
from torch.nn import functional as F
import math
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS
from ..necks.feature_fusion import FeatureFusionNeck
from typing import List, Tuple
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor
from .Functions import Encoding, Mean, DropPath, Mlp, GroupNorm, LayerNormChannel, ConvBlock
class FDAF(BaseModule):
    """Flow Dual-Alignment Fusion Module.

    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg=None
        norm_cfg=dict(type='IN')
        act_cfg=dict(type='GELU')
        
        kernel_size = 5
        self.flow_make = Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
            nn.InstanceNorm2d(in_channels*2),
            nn.GELU(),
            nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1, x2, fusion_policy=None):
        """Forward function."""

        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1
        
        if fusion_policy == None:
            return x1_feat, x2_feat
        
        output = FeatureFusionNeck.fusion(x1_feat, x2_feat, fusion_policy)
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output





class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
# LightMLPBlock
class LightMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu",
    mlp_ratio=4., drop=0., act_layer=nn.GELU, 
    use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0., norm_layer=GroupNorm):  # act_layer=nn.GELU,
        super().__init__()
        self.dw = DWConv(in_channels, out_channels, ksize=1, stride=1, act="silu")
        self.linear = nn.Linear(out_channels, out_channels)  # learnable position embedding
        self.out_channels = out_channels

        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(in_channels)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block""" # CBL

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer. \
        Here MixFFN is uesd as projection head.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):

        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class Conv3d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[7,7,3], stride=1, padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv3d_cd, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # Calculating the kernel_diff for 3D
            [C_out, C_in, kernel_size, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None, None]
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff

class cross_feature_3d(nn.Module):
    def __init__(self, in_channels):
        super(cross_feature_3d, self).__init__()
        self.conv3d = nn.Sequential(
            Conv3d_cd(1, 1, kernel_size=[3,3,3], stride=1, padding=[1,1,1], bias=True, theta= 0.7),
            # nn.InstanceNorm3d(1),
            nn.GELU(),
            Conv3d_cd(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True, theta= 0.7),
            # nn.InstanceNorm3d(1),
            nn.GELU(),

        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.InstanceNorm2d(in_channels),
            nn.GELU(),
        )
        # self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):
        tensor1 = x[0]
        tensor2 = x[1]
        b,c,h,w = tensor1.shape
        tensor1 = tensor1.view(b, c, h*w )
        tensor2 = tensor2.view(b, c, h*w )
        cross_x = torch.cat((tensor1, tensor2), dim=2)
        cross_x = cross_x.view(b, c*2, h,w)
        cross_x = cross_x.unsqueeze(1)
        cross_x = self.conv3d(cross_x)
        cross_x = cross_x.squeeze(1)   
        cross_x = self.fuse_conv(cross_x)
        # cross_x = self.dropout(cross_x)
        return cross_x
class LaplacianConvFixed(nn.Module):
    def __init__(self, in_channels):
        super(LaplacianConvFixed, self).__init__()
        # 定义拉普拉斯矩阵
        self.laplacian_kernel = torch.tensor([[0, 1, 0],
                                              [1, -4, 1],
                                              [0, 1, 0]], dtype=torch.float32)

        # 将其转换为卷积核格式 [out_channels, in_channels, H, W]
        # 由于我们希望应用相同的卷积核到所有通道，因此使用unsqueeze来扩展维度
        self.laplacian_kernel = self.laplacian_kernel.unsqueeze(0).unsqueeze(0)

        # 使用repeat扩展卷积核维度以匹配输入的通道数
        self.laplacian_kernel = self.laplacian_kernel.repeat(in_channels, 1, 1, 1)

    def forward(self, x):
        # 使用自定义的拉普拉斯卷积核进行卷积操作
        # 使用groups=x.shape[1]来确保每个通道使用相同的卷积核
        x = F.conv2d(x, self.laplacian_kernel.cuda() , padding=1, groups=x.shape[1])
        return x



class scale_atten(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(scale_atten, self).__init__()
        self.ch_in = ch_in
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1) 
        x = nn.Sigmoid()(x)
        return x * y.expand_as(x) 

@MODELS.register_module()
class CCD(BaseDecodeHead):
    """The Head of Changer.

    This head is the implementation of
    `Changer <https://arxiv.org/abs/2209.08290>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.diff = cross_feature_3d(
                    in_channels=self.channels,)


        self.line_fuse_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
        self.atten = scale_atten(self.channels)
           
        self.laplace =  LaplacianConvFixed(in_channels=self.channels*2)
    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            # print(idx,conv(x).shape)
            if idx==0:
                detail_feature = conv(x)
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        
        out = self.fusion_conv(torch.cat(outs, dim=1))
        
        return out,detail_feature

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        out1, detail_feature1 = self.base_forward(inputs1)
        out2, detail_feature2 = self.base_forward(inputs2)

        x_line_laplace = self.laplace(torch.cat([detail_feature1,detail_feature2],dim=1))

        x_line_laplace = self.line_fuse_layer(x_line_laplace)

        out = self.diff([out1, out2])


        out = self.discriminator(out)

        out_sigmoid = self.atten(out)
        
        x_line_laplace = out_sigmoid*x_line_laplace
        out = self.cls_seg(out+x_line_laplace)

        return out

