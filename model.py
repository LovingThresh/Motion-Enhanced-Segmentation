# -*- coding: utf-8 -*-
# @Time    : 2023/5/3 22:13
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : repairerGAN_generator.py
# @Software: PyCharm


import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import warnings

from typing import Sequence
from itertools import chain
from functools import partial
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint
from mmengine.model import BaseModule, ModuleList, Sequential

from mmagic.registry import MODELS

from mmseg.models.decode_heads.decode_head import BaseDecodeHead


###############################################################################
# Functions
###############################################################################

@MODELS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
                             f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=None,
             use_parallel=True,
             learn_residual=False, mode='image'):
    if gpu_ids is None:
        gpu_ids = [0]
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual, mode=mode)
        netG.apply(weights_init)
    elif which_model_netG == 'convnext_9blocks':
        netG = ConvNextGenerator(mode, 3)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
        netG.apply(weights_init)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
        netG.apply(weights_init)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
        netG.apply(weights_init)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])

    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=None,
             use_parallel=True):
    if gpu_ids is None:
        gpu_ids = [0]
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=None, use_parallel=True, learn_residual=False, padding_type='reflect', mode='image'):
        if gpu_ids is None:
            gpu_ids = [0]
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.mode = mode
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2

        # 下采样
        # for i in range(n_downsampling): # [0,1]
        # 	mult = 2**i
        #
        # 	model += [
        # 		nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        # 		norm_layer(ngf * mult * 2),
        # 		nn.ReLU(True)
        # 	]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        # 上采样
        # for i in range(n_downsampling):
        # 	mult = 2**(n_downsampling - i)
        #
        # 	model += [
        # 		nn.ConvTranspose2d(
        # 			ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
        # 			padding=1, output_padding=1, bias=use_bias),
        # 		norm_layer(int(ngf * mult / 2)),
        # 		nn.ReLU(True)
        # 	]
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            # nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
        #     output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        if self.mode == 'image':
            output = self.model(input)
            output = nn.Tanh()(output)
        else:
            output = self.model(input)
            output = nn.Sigmoid()(output)

        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        padAndConv = {
            'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        try:
            blocks = padAndConv[padding_type] + [
                norm_layer(dim),
                nn.ReLU(True)
            ] + [nn.Dropout(0.5)] if use_dropout else [] + padAndConv[padding_type] + [
                norm_layer(dim)
            ]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block = nn.Sequential(*blocks)

    # self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    # def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
    #     padAndConv = {
    #         'reflect': [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
    #         'replicate': [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
    #         'zero': [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
    #     }
    #     try:
    #         blocks = [
    #             padAndConv[padding_type],
    #
    #             norm_layer(dim),
    #             nn.ReLU(True),
    #             nn.Dropout(0.5) if use_dropout else None,
    #
    #             padAndConv[padding_type],
    #
    #             norm_layer(dim)
    #         ]
    #     except:
    #         raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    #
    #     return nn.Sequential(*blocks)

    # blocks = []
    # if padding_type == 'reflect':
    # 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'replicate':
    # 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'zero':
    # 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
    # else:
    # 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    #
    # blocks += [
    # 	norm_layer(dim),
    # 	nn.ReLU(True),
    # 	nn.Dropout(0.5) if use_dropout else None
    # ]
    #
    # if padding_type == 'reflect':
    # 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'replicate':
    # 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'zero':
    # 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
    # else:
    # 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    #
    # blocks += [
    # 	norm_layer(dim)
    # ]
    #
    # return nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, gpu_ids=None, use_parallel=True, learn_residual=False):
        super(UnetGenerator, self).__init__()
        if gpu_ids is None:
            gpu_ids = [0]
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
            self, outer_nc, inner_nc, submodule=None,
            outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dConv]
            uModel = [uRelu, uConv, nn.Tanh()]
            model = [
                dModel,
                submodule,
                uModel
            ]
        # model = [
        # 	# Down
        # 	nn.Conv2d( outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        #
        # 	submodule,
        # 	# Up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
        # 	nn.Tanh()
        # ]
        elif innermost:
            uConv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv, uNorm]
            model = [
                dModel,
                uModel
            ]
        # model = [
        # 	# down
        # 	nn.LeakyReLU(0.2, True),
        # 	# up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        # 	norm_layer(outer_nc)
        # ]
        else:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv, dNorm]
            uModel = [uRelu, uConv, uNorm]

            model = [
                dModel,
                submodule,
                uModel
            ]
            model += [nn.Dropout(0.5)] if use_dropout else []

        # if use_dropout:
        # 	model = down + [submodule] + up + [nn.Dropout(0.5)]
        # else:
        # 	model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def build_norm_layer(cfg: dict, num_features: int) -> nn.Module:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.

        num_features (int): Number of input channels.

    Returns:
        nn.Module: The created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = MODELS.get(layer_type)
    if norm_layer is None:
        raise KeyError(f'Cannot find {layer_type} in registry under scope '
                       f'name {MODELS.scope}')

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        layer = norm_layer(num_channels=num_features, **cfg_)

    if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
        layer._specify_ddp_gpu_num(1)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer


class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <https://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def generation_init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use normal init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal. Default: 'normal'.
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal. Default: 0.02.
    """

    def init_func(m):
        """Initialization function.

        Args:
            m (nn.Module): Module to be initialized.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):

        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        output = resize(
            input=output,
            size=(512, 512),
            mode='bilinear',
            align_corners=self.align_corners)

        return output


# 构造一个ConvNext Block
class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        if act_cfg is None:
            act_cfg = dict(type='GELU')
        if norm_cfg is None:
            norm_cfg = dict(type='LN2d', eps=1e-6)
        if dw_conv_cfg is None:
            dw_conv_cfg = dict(kernel_size=7, padding=3)
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


# Down Network
# @MODELS.register_module()
class ConvNeXt(BaseModule):
    """ConvNeXt v1&v2 backbone.

    A PyTorch implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_ and
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    <https://arxiv.org/abs/2301.00808>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=2,
                 norm_cfg=None,
                 act_cfg=None,
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if norm_cfg is None:
            norm_cfg = dict(type='LN2d')

        if act_cfg is None:
            act_cfg = dict(type='GELU')

        if init_cfg is None:
            init_cfg = [dict(type='TruncNormal', std=.02, bias=0.)]
            dict(type='Constant', layer=['InstanceNorm'], val=1., bias=0.)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]

        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()


# @MODELS.register_module()
class ConvNextGenerator(nn.Module):
    """Construct a Resnet-based generator that consists of residual blocks
        between a few downsampling/upsampling operations.

        Args:
            out_channels (int): Number of channels in output images.
            base_channels (int): Number of filters at the last conv layer.
                Default: 64.
            norm_cfg (dict): Config dict to build norm layer. Default:
                `dict(type='IN')`.
            padding_mode (str): The name of padding layer in conv layers:
                'reflect' | 'replicate' | 'zeros'. Default: 'reflect'.
            init_cfg (dict): Config dict for initialization.
                `type`: The name of our initialization method. Default: 'normal'.
                `gain`: Scaling factor for normal, xavier and orthogonal.
                Default: 0.02.
        """

    def __init__(self,
                 output='image',
                 out_channels=3,
                 base_channels=48,
                 norm_cfg=None,
                 padding_mode='reflect',
                 init_cfg=None):
        super().__init__()

        if init_cfg is None:
            init_cfg = dict(type='normal', gain=0.02)

        if norm_cfg is None:
            norm_cfg = dict(type='IN')
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the resnet generator.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'
        self.output = output
        self.base_model = ConvNeXt(arch='tiny', out_indices=[0, 1, 2, 3],
                                   gap_before_final_norm=False)
        self.seg_head = UPerHead(in_channels=[96, 192, 384, 768],
                                 in_index=[0, 1, 2, 3],
                                 pool_scales=(1, 2, 3, 6),
                                 channels=512,
                                 dropout_ratio=0.1,
                                 num_classes=2,
                                 norm_cfg=dict(type='BN', requires_grad=True),
                                 align_corners=False)
        self.seg_act = nn.Softmax(dim=1)

        img_head = []
        num_down = 3
        # add upsampling layers
        for i in range(num_down):
            multiple = 2 ** (num_down - i)
            img_head += [
                ConvModule(
                    in_channels=base_channels * multiple,
                    out_channels=base_channels * multiple // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    conv_cfg=dict(type='deconv', output_padding=1),
                    norm_cfg=norm_cfg)
            ]

        img_head += [
            ConvModule(
                in_channels=base_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='Tanh'),
                padding_mode=padding_mode)
        ]

        self.img_head = nn.Sequential(*img_head)

        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, input):
        """Forward function.

        Args:
            input (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        feature = self.base_model(input)
        # x = ([96, 128, 128], [192, 64, 64], [384, 32, 32], [768, 16, 16])
        img_feature = feature[-2]
        img_output = self.img_head(img_feature)
        img_output = torch.clamp(input + img_output, min=-1, max=1)
        seg_output = self.seg_head(feature)
        seg_output = self.seg_act(seg_output)
        if self.output == 'image':
            return img_output
        elif self.output == 'segmentation':
            return seg_output
        else:
            return img_output, seg_output

    def init_weights(self, pretrained=None, strict=True):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
