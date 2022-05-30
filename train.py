# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 11:37
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : train.py
# @Software: PyCharm
import os
import json
import torch
import shutil
import datetime
import data_loader
from torch.utils.data import DataLoader
from utils.visualize import visualize_save_pair

# ==============================================================================
# =                                 data                                       =
# ==============================================================================

train_data_txt = 'L:/ALASegmentationNets_v2/Data/Stage_4/train.txt'
val_data_txt = 'L:/ALASegmentationNets_v2/Data/Stage_4/val.txt'
test_data_txt = 'L:/ALASegmentationNets_v2/Data/Stage_4/test.txt'

raw_train_dir = 'L:/ALASegmentationNets_v2/Data/Stage_4/train/img/'
raw_train_mask_dir = 'L:/ALASegmentationNets_v2/Data/Stage_4/train/mask/'

raw_val_dir = 'L:/ALASegmentationNets_v2/Data/Stage_4/val/img/'
raw_val_mask_dir = 'L:/ALASegmentationNets_v2/Data/Stage_4/val/mask/'

raw_test_dir = 'L:/ALASegmentationNets_v2/Data/Stage_4/test/img/'
raw_test_mask_dir = 'L:/ALASegmentationNets_v2/Data/Stage_4/test/mask/'


def get_Motion_Image_Dataset(re_size, batch_size):
    train_dataset = data_loader.Motion_Blur_Dataset(raw_image_path=raw_train_dir,
                                                    raw_mask_path=raw_train_mask_dir,
                                                    re_size=re_size,
                                                    data_txt=train_data_txt)

    val_dataset = data_loader.Motion_Blur_Dataset(raw_image_path=raw_val_dir,
                                                  raw_mask_path=raw_val_mask_dir,
                                                  re_size=re_size,
                                                  data_txt=val_data_txt)

    test_dataset = data_loader.Motion_Blur_Dataset(raw_image_path=raw_test_dir,
                                                   raw_mask_path=raw_test_mask_dir,
                                                   re_size=re_size,
                                                   data_txt=test_data_txt)

    # when using weightedRandomSampler, it is already balanced random, so DO NOT shuffle again

    Train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    Val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    Test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return Train_loader, Val_loader, Test_loader


# ==============================================================================
# =                           copy & upload                                    =
# ==============================================================================


# 本地复制源代码，便于复现(模型文件、数据文件、训练文件、测试文件)
# 冷代码
def copy_and_upload(experiment_, hyper_params_, comet, src_path):
    a = str(datetime.datetime.now())
    b = list(a)
    b[10] = '-'
    b[13] = '-'
    b[16] = '-'
    output_dir = ''.join(b)
    output_dir = os.path.join(src_path, output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'summary'))
    os.mkdir(os.path.join(output_dir, 'save_fig'))
    os.mkdir(os.path.join(output_dir, 'save_model'))
    os.mkdir(os.path.join(output_dir, 'checkpoint'))
    hyper_params_['output_dir'] = output_dir
    hyper_params_['ex_date'] = a[:10]
    shutil.copytree('utils', '{}/{}'.format(output_dir, 'utils'))

    # 个人热代码
    shutil.copy('main.py', output_dir)
    shutil.copy('model.py', output_dir)
    shutil.copy('train.py', output_dir)
    shutil.copy('data_loader.py', output_dir)

    # 云端上次源代码
    if comet:
        experiment_.log_asset_folder('utils', log_file_name=True)

        experiment_.log_code('main.py')
        experiment_.log_code('model.py')
        experiment_.log_code('train.py')
        experiment_.log_code('data_loader.py')
        hyper_params_['ex_key'] = experiment_.get_key()
        experiment_.log_parameters(hyper_params_)
        experiment_.set_name('{}-{}'.format(hyper_params_['ex_date'], hyper_params_['ex_number']))

    hyper_params_['output_dir'] = output_dir

    with open('{}/hyper_params.json'.format(output_dir), 'w') as fp:
        json.dump(hyper_params_, fp)
    return output_dir


def operate_dict_mean(ob_dict: dict, iter_num):
    new_ob_dict = {}

    def mean_dict(value):
        return value / iter_num

    for k, v in ob_dict.items():
        new_ob_dict[k] = mean_dict(v)

    return new_ob_dict


def calculate_loss(loss_fn, it, training_loss_sum, training_loss, output, target):
    sum_loss = 0
    if isinstance(loss_fn, dict):
        if it == 1:
            for k, _ in loss_fn.items():
                training_loss_sum[k] = 0
        for k, v in loss_fn.items():
            loss = v(output, target)
            if type(loss) is tuple:
                loss = loss[0]
            sum_loss += loss
            training_loss[k] = loss.item()
            training_loss_sum[k] += loss.item()
        loss = sum_loss
    else:
        if it == 1:
            training_loss_sum['training_loss_mean'] = 0
        loss = loss_fn(output, target)
        training_loss['training_loss'] = loss.item()
        training_loss_sum['training_loss_mean'] += loss.item()

    return loss, training_loss_sum, training_loss


def calculate_eval(eval_fn, it, training_eval_sum, training_evaluation, output, target):
    evaluation = 0
    if isinstance(eval_fn, dict):
        if it == 1:
            for k, _ in eval_fn.items():
                training_eval_sum[k] = 0
        for k, v in eval_fn.items():
            evaluation = v(output, target)
            training_evaluation[k] = evaluation.item()
            training_eval_sum[k] += evaluation.item()
    else:
        if it == 1:
            training_eval_sum['training_evaluation_mean'] = 0
        evaluation = eval_fn(output * 255, target * 255)
        training_evaluation['training_evaluation'] = evaluation.item()
        training_eval_sum['training_evaluation_mean'] += evaluation.item()

    return evaluation, training_eval_sum, training_evaluation


def train_epoch(train_model, train_load, Device, loss_fn, eval_fn, optimizer, scheduler, epoch, Epochs):
    it = 0
    training_loss = {}
    training_evaluation = {}

    training_loss_sum = {}
    training_eval_sum = {}

    training_loss_mean = {}
    training_eval_mean = {}
    for batch in train_load:
        it = it + 1

        inputs, target, mask = batch
        inputs = inputs.to(Device)
        target = target.to(Device)
        mask   = mask.to(Device)
        optimizer.zero_grad()

        output = train_model(inputs)

        loss, training_loss_sum, training_loss = \
            calculate_loss(loss_fn, it, training_loss_sum, training_loss, output, target)
        loss.backward()
        optimizer.step()

        evaluation, training_eval_sum, training_evaluation = \
            calculate_eval(eval_fn, it, training_eval_sum, training_evaluation, output, target)

        training_loss_mean = operate_dict_mean(training_loss_sum, it)
        training_eval_mean = operate_dict_mean(training_eval_sum, it)

        print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(train_load)))
        print(training_loss)
        print(training_evaluation)
        print("Epoch:{}/{}, Iter:{}/{}_Mean,".format(epoch, Epochs, it, len(train_load)))
        print(training_loss_mean)
        print(training_eval_mean)
        print("-" * 80)

    scheduler.step()

    training_dict = {
        'loss_mean': training_loss_mean,
        'evaluation_mean': training_eval_mean,
    }
    return training_loss_mean, training_eval_mean, training_dict


def train_generator_epoch(train_model_G, train_model_D,
                          train_load, Device,
                          loss_function_G_, loss_fn_G, loss_fn_D,
                          eval_fn_G, eval_fn_D,
                          optimizer_G, optimizer_D,
                          scheduler_G, scheduler_D, epoch, Epochs):
    it = 0
    label_size = 1
    training_loss_D = {}
    training_evaluation_D = {}

    training_loss_sum_D = {}
    training_eval_sum_D = {}

    training_loss_mean_D = {}
    training_eval_mean_D = {}

    training_loss_G = {}
    training_evaluation_G = {}

    training_loss_sum_G = {}
    training_eval_sum_G = {}

    training_loss_mean_G = {}
    training_eval_mean_G = {}

    for batch in train_load:
        it = it + 1
        optimizer_D.zero_grad()
        real_input, real_output, real_mask = batch
        real_input, real_output, real_mask = real_input.to(Device), real_output.to(Device), real_mask.to(Device)
        b_size = real_output.size(0)

        # Real Training
        real_predict = train_model_D(real_output)
        real_label = torch.ones_like(real_predict, dtype=torch.float, device=Device)

        loss_D_O, training_loss_sum_D, training_loss_D = \
            calculate_loss(loss_fn_D, it, training_loss_sum_D, training_loss_D, real_predict, real_label)

        # Fake Training
        fake_label = torch.zeros_like(real_predict, dtype=torch.float, device=Device)
        fake_predict = train_model_D(train_model_G(real_input))
        loss_D_T, training_loss_sum_D, training_loss_D = \
            calculate_loss(loss_fn_D, it, training_loss_sum_D, training_loss_D, fake_predict, fake_label)
        evaluation_D, training_eval_sum_D, training_evaluation_D = \
            calculate_eval(eval_fn_D, it, training_eval_sum_D, training_evaluation_D, fake_predict, fake_label)

        loss_D = loss_D_O + loss_D_T
        loss_D.backward()
        optimizer_D.step()

        # Generator Training
        optimizer_G.zero_grad()
        gen_predict = train_model_G(real_input)
        loss_G_, training_loss_sum_G, training_loss_G = \
            calculate_loss(loss_function_G_, it, training_loss_sum_G, training_loss_G, gen_predict, 1)

        loss_G, training_loss_sum_G, training_loss_G = \
            calculate_loss(loss_fn_G, it, training_loss_sum_G, training_loss_G, gen_predict, real_output)

        evaluation_G, training_eval_sum_G, training_evaluation_G = \
            calculate_eval(eval_fn_G, it, training_eval_sum_G, training_evaluation_G, gen_predict, real_output)

        loss_G = loss_G_ + loss_G

        loss_G.backward()
        optimizer_G.step()

        training_loss_mean_D = operate_dict_mean(training_loss_sum_D, it)
        training_eval_mean_D = operate_dict_mean(training_eval_sum_D, it)

        training_loss_mean_G = operate_dict_mean(training_loss_sum_G, it)
        training_eval_mean_G = operate_dict_mean(training_eval_sum_G, it)

        print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(train_load)))
        print(training_loss_D)
        print(training_evaluation_D)
        print(training_loss_G)
        print(training_evaluation_G)
        print("Epoch:{}/{}, Iter:{}/{}_Mean,".format(epoch, Epochs, it, len(train_load)))
        print(training_loss_mean_D)
        print(training_eval_mean_D)
        print(training_loss_mean_G)
        print(training_eval_mean_G)
        print("-" * 80)

    scheduler_D.step()
    scheduler_G.step()

    training_dict = {
        'loss_mean_D': training_loss_mean_D,
        'evaluation_mean_D': training_eval_mean_D,
        'loss_mean_G': training_loss_mean_G,
        'evaluation_mean_G': training_eval_mean_G,

    }
    return training_loss_mean_D, training_eval_mean_D, training_loss_mean_G, training_eval_mean_G, training_dict


def val_epoch(valid_model, val_load, Device, loss_fn, eval_fn, epoch, Epochs):
    it = 0
    valid_loss = {}
    valid_evaluation = {}

    valid_loss_sum = {}
    valid_eval_sum = {}

    valid_loss_mean = {}
    valid_eval_mean = {}

    for batch in val_load:
        it = it + 1
        inputs, target, mask = batch

        inputs = inputs.to(Device)
        output = valid_model(inputs)
        target = target.to(Device)
        mask   = mask.to(Device)

        loss, valid_loss_sum, valid_loss = \
            calculate_loss(loss_fn, it, valid_loss_sum, valid_loss, output, target)

        evaluation, valid_eval_sum, valid_evaluation = \
            calculate_eval(eval_fn, it, valid_eval_sum, valid_evaluation, output, target)

        valid_loss_mean = operate_dict_mean(valid_loss_sum, it)
        valid_eval_mean = operate_dict_mean(valid_eval_sum, it)

        print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(val_load)))
        print(valid_loss)
        print(valid_evaluation)
        print("Epoch:{}/{}, Iter:{}/{}_Mean,".format(epoch, Epochs, it, len(val_load)))
        print(valid_loss_mean)
        print(valid_eval_mean)
        print("-" * 80)

    valid_dict = {
        'loss_mean': valid_loss_mean,
        'evaluation_mean': valid_eval_mean
    }

    return valid_loss_mean, valid_eval_mean, valid_dict


def val_generator_epoch(train_model_G, train_model_D,
                        val_load, Device,
                        loss_function_G_, loss_fn_G, loss_fn_D,
                        eval_fn_G, eval_fn_D,
                        epoch, Epochs):
    it = 0
    label_size = 1
    training_loss_D = {}
    training_evaluation_D = {}

    training_loss_sum_D = {}
    training_eval_sum_D = {}

    training_loss_mean_D = {}
    training_eval_mean_D = {}

    training_loss_G = {}
    training_evaluation_G = {}

    training_loss_sum_G = {}
    training_eval_sum_G = {}

    training_loss_mean_G = {}
    training_eval_mean_G = {}

    for batch in val_load:
        it = it + 1

        real_input, real_output, real_mask = batch
        real_input, real_output, real_mask = real_input.to(Device), real_output.to(Device), real_mask.to(Device)
        b_size = real_output.size(0)

        # Real
        real_predict = train_model_D(real_output)
        real_label = torch.ones_like(real_predict, dtype=torch.float, device=Device)

        loss_D_O, training_loss_sum_D, training_loss_D = \
            calculate_loss(loss_fn_D, it, training_loss_sum_D, training_loss_D, real_predict, real_label)

        # Fake
        fake_label = torch.zeros_like(real_predict, dtype=torch.float, device=Device)
        fake_predict = train_model_D(train_model_G(real_input))
        loss_D_T, training_loss_sum_D, training_loss_D = \
            calculate_loss(loss_fn_D, it, training_loss_sum_D, training_loss_D, fake_predict, fake_label)
        evaluation_D, training_eval_sum_D, training_evaluation_D = \
            calculate_eval(eval_fn_D, it, training_eval_sum_D, training_evaluation_D, fake_predict, fake_label)

        # Generator
        gen_predict = train_model_G(real_input)
        loss_G_, training_loss_sum_G, training_loss_G = \
            calculate_loss(loss_function_G_, it, training_loss_sum_G, training_loss_G, gen_predict, 1)

        loss_G, training_loss_sum_G, training_loss_G = \
            calculate_loss(loss_fn_G, it, training_loss_sum_G, training_loss_G, gen_predict, real_output)

        evaluation_G, training_eval_sum_G, training_evaluation_G = \
            calculate_eval(eval_fn_G, it, training_eval_sum_G, training_evaluation_G, gen_predict, real_output)

        training_loss_mean_D = operate_dict_mean(training_loss_sum_D, it)
        training_eval_mean_D = operate_dict_mean(training_eval_sum_D, it)

        training_loss_mean_G = operate_dict_mean(training_loss_sum_G, it)
        training_eval_mean_G = operate_dict_mean(training_eval_sum_G, it)

        print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(val_load)))
        print(training_loss_D)
        print(training_evaluation_D)
        print(training_loss_G)
        print(training_evaluation_G)
        print("Epoch:{}/{}, Iter:{}/{}_Mean,".format(epoch, Epochs, it, len(val_load)))
        print(training_loss_mean_D)
        print(training_eval_mean_D)
        print(training_loss_mean_G)
        print(training_eval_mean_G)
        print("-" * 80)

    training_dict = {
        'loss_mean_D': training_loss_mean_D,
        'evaluation_mean_D': training_eval_mean_D,
        'loss_mean_G': training_loss_mean_G,
        'evaluation_mean_G': training_eval_mean_G,

    }
    return training_loss_mean_D, training_eval_mean_D, training_loss_mean_G, training_eval_mean_G, training_dict


def write_dict(dict_to_write, writer, step):
    for k, v in dict_to_write.items():
        for i, j in v.items():
            writer.add_scalar('{}/{}'.format(k, i), j, step)


def write_summary(train_writer_summary, valid_writer_summary, train_dict, valid_dict, step):
    write_dict(train_dict, train_writer_summary, step)
    write_dict(valid_dict, valid_writer_summary, step)


def train(training_model, optimizer, loss_fn, eval_fn,
          train_load, val_load, epochs, scheduler, Device,
          threshold, output_dir, train_writer_summary, valid_writer_summary,
          experiment, comet=False, init_epoch=1):
    training_model = training_model.to(Device)

    def train_process(B_comet, experiment_comet, threshold_value=threshold, init_epoch_num=init_epoch):

        for epoch in range(init_epoch_num, epochs + init_epoch_num):

            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            training_model.train(True)
            train_loss, train_evaluation, train_dict = train_epoch(training_model, train_load, Device, loss_fn, eval_fn,
                                                                   optimizer, scheduler, epoch, epochs)
            training_model.train(False)
            val_loss, val_evaluation, valid_dict = val_epoch(training_model, val_load, Device, loss_fn, eval_fn,
                                                             epoch, epochs)
            write_summary(train_writer_summary, valid_writer_summary, train_dict, valid_dict, step=epoch)

            if B_comet:
                for k, v in train_dict.items():
                    experiment_comet.log_metrics(v, step=epoch)
                for k, v in valid_dict.items():
                    experiment_comet.log_metrics(v, step=epoch)

            print('Epoch: {}, \n Mean Training Loss:{}, \n Mean Validation Loss: {}, \n'
                  'Mean Training evaluation:{}, \nMean Validation evaluation:{} '
                  .format(epoch, train_loss, val_loss, train_evaluation, val_evaluation))

            # 这一部分可以根据任务进行调整
            if val_evaluation['eval_function_psnr'] > threshold_value:
                torch.save(training_model.state_dict(),
                           os.path.join(output_dir, 'save_model',
                                        'Epoch_{}_eval_{}.pt'.format(epoch, val_evaluation['eval_function_psnr'])))
                threshold_value = val_evaluation['eval_function_psnr']

            # 验证阶段的结果可视化
            save_path = os.path.join(output_dir, 'save_fig')
            visualize_save_pair(training_model, val_load, save_path, epoch)

            if (epoch % 100) == 0:
                save_checkpoint_path = os.path.join(output_dir, 'checkpoint')
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": training_model.state_dict(),
                    "loss_fn": loss_fn,
                    "eval_fn": eval_fn,
                    "lr_schedule_state_dict": scheduler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(save_checkpoint_path, str(epoch), '.pth'))

    if not comet:
        train_process(comet, experiment)
    else:
        with experiment.train():
            train_process(comet, experiment)


def train_GAN(training_model_G, training_model_D,
              optimizer_G, optimizer_D, loss_function_G_, loss_fn_G, loss_fn_D,
              scheduler_G, scheduler_D, eval_fn_G, eval_fn_D,
              train_load, val_load, epochs, Device,
              threshold, output_dir, train_writer_summary, valid_writer_summary,
              experiment, comet=False, init_epoch=1):
    training_model_G = training_model_G.to(Device)
    training_model_D = training_model_D.to(Device)

    def train_process(B_comet, experiment_comet, threshold_value=threshold, init_epoch_num=init_epoch):

        for epoch in range(init_epoch_num, epochs + init_epoch_num):

            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            training_model_G.train(True)
            training_model_D.train(True)
            train_loss_D, train_eval_D, train_loss_G, train_eval_G, train_dict = \
                train_generator_epoch(training_model_G, training_model_D,
                                      train_load, Device, loss_function_G_, loss_fn_G, loss_fn_D,
                                      eval_fn_G, eval_fn_D, optimizer_G,
                                      optimizer_D, scheduler_G, scheduler_D,
                                      epoch, epochs)

            training_model_G.train(False)
            training_model_D.train(False)
            val_loss_D, val_eval_D, val_loss_G, val_eval_G, valid_dict = \
                val_generator_epoch(training_model_G, training_model_D,
                                    val_load, Device, loss_function_G_, loss_fn_G, loss_fn_D,
                                    eval_fn_G, eval_fn_D, epoch, epochs)
            write_summary(train_writer_summary, valid_writer_summary, train_dict, valid_dict, step=epoch)

            if B_comet:
                for k, v in train_dict.items():
                    experiment_comet.log_metrics(v, step=epoch)
                for k, v in valid_dict.items():
                    experiment_comet.log_metrics(v, step=epoch)

            print('Epoch: {}, \n '
                  'Mean Training Loss D:{}, \n '
                  'Mean Training Loss G:{}, \n '
                  'Mean Validation Loss D: {}, \n'
                  'Mean Validation Loss G:{}, \n '
                  'Mean Training evaluation D:{}, \n'
                  'Mean Training evaluation G:{}, \n'
                  'Mean Validation evaluation D:{}, \n'
                  'Mean Validation evaluation G:{} '
                  .format(epoch, train_loss_D, train_loss_G, val_loss_D, val_loss_G,
                          train_eval_D, train_eval_G, val_eval_D, val_eval_G))

            # 这一部分可以根据任务进行调整
            if val_eval_G['eval_function_psnr'] > threshold_value:
                torch.save(training_model_G.state_dict(),
                           os.path.join(output_dir, 'save_model',
                                        'Epoch_{}_eval_{}.pt'.format(epoch, val_eval_G['eval_function_psnr'])))
                threshold_value = val_eval_G['eval_function_psnr']

            # 验证阶段的结果可视化
            save_path = os.path.join(output_dir, 'save_fig')
            visualize_save_pair(training_model_G, val_load, save_path, epoch)

            if (epoch % 100) == 0:
                save_checkpoint_path = os.path.join(output_dir, 'checkpoint')
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": [training_model_G.state_dict(), training_model_D.state_dict()],
                    "loss_fn": [loss_fn_D, loss_fn_G],
                    "eval_fn": [eval_fn_D, eval_fn_G],
                    "lr_schedule_state_dict": [scheduler_D.state_dict(), scheduler_G.state_dict()],
                    "optimizer_state_dict": [optimizer_D.state_dict(), optimizer_G.state_dict()]
                }, os.path.join(save_checkpoint_path, str(epoch) + '.pth'))

    if not comet:
        train_process(comet, experiment)
    else:
        with experiment.train():
            train_process(comet, experiment)
