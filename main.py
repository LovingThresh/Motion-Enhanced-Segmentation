# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 16:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm

import mmcv
import numpy as np
import torchmetrics.functional
from mmedit.models import MODELS
from mmedit.models import LOSSES

import torchsummary
import torchmetrics
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from train import *
from model import *
from comet_ml import Experiment
from utils.visualize import visualize_pair

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)

train_comet = False

hyper_params = {
    "ex_number"     : 'EDSR_3080Ti',
    "raw_size"      : (3, 448, 448),
    "crop_size"     : (3, 224, 224),
    "input_size"    : (3, 224, 224),
    "batch_size"    : 4,
    "learning_rate" : 1e-4,
    "epochs"        : 200,
    "threshold"     : 28,
    "checkpoint"    : False,
    "Img_Recon"     : True,
    "src_path"      : 'E:/BJM/Motion_Image',
    "check_path"    : 'E:/bjm/Super_Resolution/2022-05-16-09-44-25.670141/checkpoint/400.tar'
}

experiment  =   object
lr          =   hyper_params['learning_rate']
Epochs      =   hyper_params['epochs']
src_path    =   hyper_params['src_path']
batch_size  =   hyper_params['batch_size']
raw_size    =   hyper_params['raw_size'][1:]
crop_size   =   hyper_params['crop_size'][1:]
input_size  =   hyper_params['input_size'][1:]
threshold   =   hyper_params['threshold']
Checkpoint  =   hyper_params['checkpoint']
Img_Recon   =   hyper_params['Img_Recon']
check_path  =   hyper_params['check_path']
# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = Experiment(
        api_key      = "sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name = "Motion_Image_Enhancement",
        workspace    = "LovingThresh",
    )

# ===============================================================================
# =                                     Data                                    =
# ===============================================================================

train_loader, val_loader, test_loader = get_Motion_Image_Dataset(re_size=raw_size, batch_size=batch_size)

visualize_pair(train_loader, input_size=input_size, crop_size=crop_size)

# ===============================================================================
# =                                     Model                                   =
# ===============================================================================

generator = define_G(3, 3, 64, 'resnet_9blocks')
discriminator = define_D(3, 64, 'n_layers', 4)

# model = ResNet(101, double_input=Img_Recon)
# model.init_weights()
# model = SRResNet()

# torchsummary.summary(model, input_size=hyper_params['input_size'], batch_size=batch_size, device='cpu')

# ===============================================================================
# =                                    Setting                                  =
# ===============================================================================

gan_loss = dict(
    type='GANLoss',
    gan_type='wgan',
    loss_weight=5e-3,
    real_label_val=1.0,
    fake_label_val=0.)
gan_loss = mmcv.build_from_cfg(gan_loss, LOSSES)

pixel_loss = dict(type='L1Loss', loss_weight=1e-2, reduction='sum')
pixel_loss = mmcv.build_from_cfg(pixel_loss, LOSSES)

loss_function_D = {'loss_function_BCE': nn.BCEWithLogitsLoss()}

loss_function_G_ = {'loss_function_gan': gan_loss}
loss_function_G = {'loss_function_l1': pixel_loss}

eval_function_psnr = torchmetrics.functional.image.psnr.peak_signal_noise_ratio
eval_function_ssim = torchmetrics.functional.image.ssim.structural_similarity_index_measure
eval_function_mse = torchmetrics.functional.mean_squared_error

eval_function_D = {'eval_function_mse': eval_function_mse}
eval_function_G = {'eval_function_psnr': eval_function_psnr,
                   'eval_function_ssim': eval_function_ssim}

optimizer_ft_D     = optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
optimizer_ft_G     = optim.AdamW(generator.parameters(), lr=lr, betas=(0.9, 0.999))

# exp_lr_scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_ft_D, int(Epochs / 10))
# exp_lr_scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_ft_G, int(Epochs / 10))

exp_lr_scheduler_D = lr_scheduler.StepLR(optimizer_ft_D, step_size=30, gamma=0.)
exp_lr_scheduler_G = lr_scheduler.StepLR(optimizer_ft_G, step_size=30, gamma=0.)

# ===============================================================================
# =                                  Copy & Upload                              =
# ===============================================================================

output_dir   =  copy_and_upload(experiment, hyper_params, train_comet, src_path)
timestamp    =  datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_writer =  SummaryWriter('{}/trainer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))
val_writer   =  SummaryWriter('{}/valer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))


# ===============================================================================
# =                                Checkpoint                                   =
# ===============================================================================

# if Checkpoint:
#     checkpoint = torch.load(check_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
#     for state in optimizer_ft.state.values():
#         for k, v in state.items():
#             if isinstance(v, torch.Tensor):
#                 state[k] = v.cuda()
#     exp_lr_scheduler.load_state_dict(checkpoint['lr_schedule_state_dict'])

# ===============================================================================
# =                                    Training                                 =
# ===============================================================================

train(generator, optimizer_ft_G, loss_function_G, eval_function_G,
      train_loader, val_loader, Epochs, exp_lr_scheduler_G,
      device, threshold, output_dir, train_writer, val_writer, experiment, train_comet)

# train_GAN(generator, discriminator, optimizer_ft_G, optimizer_ft_D,
#           loss_function_G_, loss_function_G, loss_function_D, exp_lr_scheduler_G, exp_lr_scheduler_D,
#           eval_function_G, eval_function_D, train_loader, val_loader, Epochs, device, threshold,
#           output_dir, train_writer, val_writer, experiment, train_comet)
