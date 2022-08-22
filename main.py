# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 16:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm
from comet_ml import Experiment


import torchsummary

import mmcv
import random
# import torchmetrics.functional
# from mmedit.models import MODELS
from mmedit.models import LOSSES
from mmseg.models import BACKBONES, HEADS, LOSSES

import torchmetrics
# import torchsummary
import torch.optim as optim
import torch.backends.cudnn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from train import *
from model import *
from RepVGG import *
from MSBDN_RDFF import *
from utils.Loss import *
from utils.utils import seed_everything, dict_load
from utils.visualize import visualize_pair

from FPN_DenseNet import FPNDense
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_comet = False
autocast_button = False

random.seed(48)
np.random.seed(48)
torch.manual_seed(48)
torch.cuda.manual_seed_all(48)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# seed = 24
# seed_everything(24)

hyper_params = {
    "mode": 'image',
    "ex_number": 'EDSR_3080Ti_Segmentation_Init_New',
    "raw_size": (3, 512, 512),
    "crop_size": (3, 256, 256),
    "input_size": (3, 256, 256),
    "batch_size": 8,
    "learning_rate": 1e-4,
    "epochs": 400,
    "threshold": 28,
    "checkpoint": False,
    "Img_Recon": True,
    "src_path": 'E:/BJM/Motion_Image',
    "check_path": 'New_earthquake_crack_init_model.pt'
}

experiment = object
lr = hyper_params['learning_rate']
mode = hyper_params['mode']
Epochs = hyper_params['epochs']
src_path = hyper_params['src_path']
batch_size = hyper_params['batch_size']
raw_size = hyper_params['raw_size'][1:]
crop_size = hyper_params['crop_size'][1:]
input_size = hyper_params['input_size'][1:]
threshold = hyper_params['threshold']
Checkpoint = hyper_params['checkpoint']
Img_Recon = hyper_params['Img_Recon']
check_path = hyper_params['check_path']
# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = Experiment(
        api_key="sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name="Motion_Image_Enhancement",
        workspace="LovingThresh",
    )

# ===============================================================================
# =                                     Data                                    =
# ===============================================================================

train_loader, val_loader, test_loader = get_Motion_Image_Dataset(re_size=raw_size, batch_size=batch_size)
a = next(iter(train_loader))
visualize_pair(train_loader, input_size=input_size, crop_size=crop_size, mode=mode)

# ===============================================================================
# =                                     Model                                   =
# ===============================================================================
generator = FPNDense()

# if mode == 'image':
#     generator = define_G(3, 3, 64, 'resnet_9blocks', learn_residual=True, norm='instance', mode=mode)
# else:
#     generator = define_G(3, 2, 64, 'resnet_9blocks', learn_residual=False, norm='instance', mode=mode)

# generator = Net(mode=mode)
# dict_load(generator.segmentation_model.state_dict(), torch.load('for_segmentation_model.pt'))
# deploy = False
# generator = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2,
#                    width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)
# generator.apply(weights_init)

discriminator = define_D(3, 64, 'basic', use_sigmoid=True, norm='instance')


# model = ResNet(101, double_input=Img_Recon)
# model.init_weights()
# model = SRResNet()

# torchsummary.summary(model, input_size=hyper_params['input_size'], batch_size=batch_size, device='cpu')

# ===============================================================================
# =                                    Setting                                  =
# ===============================================================================

# perceptual_loss = dict(
#     type='PerceptualLoss',
#     layer_weights={'34': 1.0},
#     vgg_type='vgg19',
#     perceptual_weight=100.0,
#     style_weight=0,
#     norm_img=True,
#     criterion='mse')
# perceptual_loss = mmcv.build_from_cfg(perceptual_loss, LOSSES)


# pixel_loss = dict(type='L1Loss', loss_weight=10, reduction='mean')
# pixel_loss = mmcv.build_from_cfg(pixel_loss, LOSSES)

loss_function_D = {'loss_function_dis': nn.BCELoss()}

if mode == 'segmentation':
    loss_function_G_ = {'loss_function_dis': Asymmetry_Binary_Loss}
else:
    loss_function_G_ = {'loss_function_dis': nn.BCELoss()}

loss_function_G = {  # 'content_loss': pixel_loss,
    'perceptual_loss': perceptual_loss
}

eval_function_psnr = torchmetrics.functional.image.psnr.peak_signal_noise_ratio
eval_function_ssim = torchmetrics.functional.image.ssim.structural_similarity_index_measure
eval_function_iou = iou
eval_function_pr = pr
eval_function_re = re
eval_function_acc = torchmetrics.functional.accuracy

eval_function_D = {'eval_function_acc': eval_function_acc}

eval_function_G = {'eval_function_psnr': eval_function_psnr,
                   'eval_function_ssim': eval_function_ssim,
                   'eval_function_coef': correlation}

# eval_function_G = {'eval_function_iou': eval_function_iou,
#                    'eval_function_pr': eval_function_pr,
#                    'eval_function_re': eval_function_re,
#                    'eval_function_acc': eval_function_acc,
#                    }
optimizer_ft_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_ft_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

# exp_lr_scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_ft_D, int(Epochs / 10))
# exp_lr_scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_ft_G, int(Epochs / 10))

exp_lr_scheduler_D = lr_scheduler.StepLR(optimizer_ft_D, step_size=5, gamma=0.5)
exp_lr_scheduler_G = lr_scheduler.StepLR(optimizer_ft_G, step_size=5, gamma=0.5)

# ===============================================================================
# =                                  Copy & Upload                              =
# ===============================================================================

output_dir = copy_and_upload(experiment, hyper_params, train_comet, src_path)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_writer = SummaryWriter('{}/trainer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))
val_writer = SummaryWriter('{}/valer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))

# ===============================================================================
# =                                Checkpoint                                   =
# ===============================================================================

if Checkpoint:
    checkpoint = torch.load(check_path)
    generator.load_state_dict(checkpoint)
    print("Load CheckPoint!")
    # discriminator.load_state_dict(checkpoint['model_state_dict'][1])
    # optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
    # for state in optimizer_ft.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.cuda()
    # exp_lr_scheduler.load_state_dict(checkpoint['lr_schedule_state_dict'])

# ===============================================================================
# =                                    Training                                 =
# ===============================================================================

# train(generator, optimizer_ft_G, loss_function_G_, eval_function_G,
#       train_loader, val_loader, Epochs, exp_lr_scheduler_G,
#       device, threshold, output_dir, train_writer, val_writer, experiment, train_comet, mode=mode)

train_GAN(generator, discriminator, optimizer_ft_G, optimizer_ft_D,
          loss_function_G_, loss_function_G, loss_function_D, exp_lr_scheduler_G, exp_lr_scheduler_D,
          eval_function_G, eval_function_D, train_loader, val_loader, Epochs, device, threshold,
          output_dir, train_writer, val_writer, experiment, train_comet)


# G_path = 'E:/BJM/Motion_Image/2022-06-10-14-12-27.500277/save_model/Epoch_8_eval_22.14645609362372.pt'
# save_path = 'E:/BJM/Motion_Image/2022-06-10-14-12-27.500277'
# GAN_test(generator, discriminator, model_G_path=G_path, output_dir=save_path, loss_function_G_=loss_function_G_,
#          loss_fn_G=loss_function_G, loss_fn_D=loss_function_D, eval_fn_G=eval_function_G, eval_fn_D=eval_function_D,
#          test_load=test_loader, Device=device)
