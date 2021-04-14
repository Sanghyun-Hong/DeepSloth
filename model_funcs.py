"""
    A script that implements the functions for training, testing SDNs and CNNs.
    [+ also implements the functions for computing confusion and confidence]
"""
# basics
import math, copy
import time, random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from random import choice, shuffle
from sklearn.metrics import confusion_matrix

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
import torchvision.utils as vutils

# custom libs
import datasets, models, utils
import attacks.ours_linf as ours_linf
from attacks.PGDs import PGD, PGD_avg, PGD_max
from profiler import profile_sdn


# ------------------------------------------------------------------------------
#   SDN training/test/misc. functions
# ------------------------------------------------------------------------------
def sdn_training_step(optimizer, model, coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        cur_loss = float(coeffs[ic_id])*utils.get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss += utils.get_loss_criterion()(output[-1], b_y)
    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss

def sdn_ic_only_step(optimizer, model, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    for output_id, cur_output in enumerate(output):
        if output_id == model.num_output - 1: # last output
        # if output_id == len(output) - 1: # modified by ionut: got error here because num_output doesn't exist in VGG_SDN when I trained AT_ic_only from Sanghyun
            break

        cur_loss = utils.get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss


def get_loader(data, augment):
    if augment:
        train_loader = data.aug_train_loader
    else:
        train_loader = data.train_loader
    return train_loader


def sdn_train(model, data, epochs, optimizer, scheduler, robust=False, device='cpu'):
    augment = model.augment_training
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}
    max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]) # max tau_i --- C_i values

    if model.ic_only:
        print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    else:
        print('sdn will be trained from scratch...(The SDN training)')

    for epoch in range(1, epochs+1):
        scheduler.step()
        cur_lr = utils.get_lr(optimizer)
        print(datetime.now())
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        if model.ic_only is False:
            # calculate the IC coeffs for this epoch for the weighted objective function
            cur_coeffs = 0.01 + epoch*(max_coeffs/epochs) # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            print('Cur coeffs: {}'.format(cur_coeffs))

        start_time = time.time()
        model.train()
        loader = get_loader(data, augment)
        for i, batch in enumerate(loader):
            if model.ic_only is False:
                total_loss = sdn_training_step(optimizer, model, cur_coeffs, batch, device)
            else:
                total_loss = sdn_ic_only_step(optimizer, model, batch, device)

            if i % 100 == 0:
                print('Loss: {}: '.format(total_loss))

        top1_test, top5_test = sdn_test(model, data.test_loader, device)

        print('Top1 Test accuracies: {}'.format(top1_test))
        print('Top5 Test accuracies: {}'.format(top5_test))
        end_time = time.time()

        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
        print('Top1 Train accuracies: {}'.format(top1_train))
        print('Top5 Train accuracies: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)

        epoch_time = int(end_time-start_time)
        metrics['epoch_times'].append(epoch_time)
        print('Epoch took {} seconds.'.format(epoch_time))

        metrics['lrs'].append(cur_lr)

    return metrics

def sdn_training_step_adv( \
    optimizer, model, coeffs, batch, \
    attack, iteration, eps_step, eps_max, device='cpu'):
    # batch of x and y
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)

    # forward pass with clean samples
    output = model(b_x)
    optimizer.zero_grad()

    # compute the loss for the clean samples (consider itermediate outputs)
    total_loss = 0.0
    criterion  = utils.get_loss_criterion()
    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        cur_loss = float(coeffs[ic_id]) * criterion(cur_output, b_y)
        total_loss += cur_loss
    total_loss += criterion(output[-1], b_y)

    # compute the loss for the adv. samples
    if 'PGD' == attack:
        b_advx = PGD( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., eps_step=iteration, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False, cnn=False).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        for ic_id in range(model.num_output - 1):
            cur_output = output_adv[ic_id]
            cur_loss = float(coeffs[ic_id]) * criterion(cur_output, b_y)
            total_loss += cur_loss
        total_loss += criterion(output_adv[-1], b_y)

    elif 'PGD-avg' == attack:
        b_advx = PGD_avg( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., eps_step=iteration, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        for ic_id in range(model.num_output - 1):
            cur_output = output_adv[ic_id]
            cur_loss = float(coeffs[ic_id]) * criterion(cur_output, b_y)
            total_loss += cur_loss
        total_loss += criterion(output_adv[-1], b_y)

    elif 'PGD-max' == attack:
        b_advx = PGD_max( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., eps_step=iteration, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        for ic_id in range(model.num_output - 1):
            cur_output = output_adv[ic_id]
            cur_loss = float(coeffs[ic_id]) * criterion(cur_output, b_y)
            total_loss += cur_loss
        total_loss += criterion(output_adv[-1], b_y)

    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss

def sdn_ic_only_step_adv( \
    optimizer, model, batch, \
    attack, iteration, eps_step, eps_max, device='cpu'):
    # batch of x and y
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)

    # forward pass with clean samples
    output = model(b_x)
    optimizer.zero_grad()

    # compute the loss for the clean samples (consider itermediate outputs)
    total_loss = 0.0
    criterion  = utils.get_loss_criterion()
    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        total_loss += criterion(cur_output, b_y)
    total_loss += criterion(output[-1], b_y)

    # compute the loss for the adv. samples
    if 'PGD' == attack:
        b_advx = PGD( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., steps=iteration, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False, cnn=False).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        for ic_id in range(model.num_output - 1):
            cur_output = output_adv[ic_id]
            total_loss += criterion(cur_output, b_y)
        total_loss += criterion(output_adv[-1], b_y)

    elif 'PGD-avg' == attack:
        b_advx = PGD_avg( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., steps=iteration, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        for ic_id in range(model.num_output - 1):
            cur_output = output_adv[ic_id]
            total_loss += criterion(cur_output, b_y)
        total_loss += criterion(output_adv[-1], b_y)

    elif 'PGD-max' == attack:
        b_advx = PGD_max( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., steps=iteration, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        for ic_id in range(model.num_output - 1):
            cur_output = output_adv[ic_id]
            total_loss += criterion(cur_output, b_y)
        total_loss += criterion(output_adv[-1], b_y)

    elif 'ours' == attack:
        b_advx = ours_linf.craft_per_sample_perturb_attack_on_samples( \
            model, b_x, max_iter=30, epsilon=0.03125, eps_step=0.002, device=device)
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        # loss conditions
        use_loss = 'same_logits'    # too strong - IC1's performance becomes 28% or something ...
        # use_loss = 'same_output'

        for ic_id in range(model.num_output - 1):
            if use_loss == 'same_logits':
                cur_output = output[ic_id]
                cur_outadv = output_adv[ic_id]
                total_loss += _cross_entropy_w_logits(cur_output, cur_outadv)
            elif use_loss == 'same_output':
                cur_outadv = output_adv[ic_id]
                total_loss += criterion(cur_outadv, b_y)

        if use_loss == 'same_logits':
            total_loss += _cross_entropy_w_logits(output[-1], output_adv[-1])
        elif use_loss == 'same_output':
            total_loss += criterion(output_adv[-1], b_y)

    elif 'mixs' == attack:
        """
            Compute the loss from the PGD perturbations
        """
        b_advx = PGD( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., steps=10, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False, cnn=False).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        for ic_id in range(model.num_output - 1):
            cur_output = output_adv[ic_id]
            total_loss += criterion(cur_output, b_y)
        total_loss += criterion(output_adv[-1], b_y)

        """
            Compute the loss from the DeepSloth perturbations
        """
        b_advx = ours_linf.craft_per_sample_perturb_attack_on_samples( \
            model, b_x, max_iter=30, epsilon=0.03125, eps_step=0.002, device=device)
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)

        # loss conditions
        # use_loss = 'same_logits'    # too strong - IC1's performance becomes 28% or something ...
        use_loss = 'same_output'

        for ic_id in range(model.num_output - 1):
            if use_loss == 'same_logits':
                cur_output = output[ic_id]
                cur_outadv = output_adv[ic_id]
                total_loss += _cross_entropy_w_logits(cur_output, cur_outadv)
            elif use_loss == 'same_output':
                cur_outadv = output_adv[ic_id]
                total_loss += criterion(cur_outadv, b_y)

        if use_loss == 'same_logits':
            total_loss += _cross_entropy_w_logits(output[-1], output_adv[-1])
        elif use_loss == 'same_output':
            total_loss += criterion(output_adv[-1], b_y)

    # end if

    # compute the others
    total_loss.backward()
    optimizer.step()

    return total_loss

def _cross_entropy_w_logits(output, labels):
    custom_loss = -F.softmax(labels, dim=1) * torch.log(F.softmax(output, dim=1))
    custom_loss = custom_loss.sum() / output.shape[0]
    return custom_loss

def sdn_advtrain( \
    model, data, epochs, optimizer, scheduler, save_func, \
    attack, iteration, eps_step, eps_max, device='cpu'):
    augment = model.augment_training
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    """
        Ionut's Comment:
            The old max_coeffs array that contains 6 values is from the SDN paper where the SDNs had 6 internal classifiers.
            Each value in the max_coeffs array is an approximation of the percentage GFLOPs of the hidden layer where the IC is attached to.
            However, in the context of DeepSloth, where the SDN attaches an IC at EVERY HIDDEN LAYER, those 6 values won't be enough.
            As a solution, the 6-values array would be replaced by the percentage of the GFLOPs obtained by the profile_sdn method
    """
    # max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]) # max tau_i --- C_i values
    ops_dict, _ = profile_sdn(model, model.input_size, device) # ops_dict: key=ic_number, value=ops of the layer where ic is attached to
    ops = np.array([ops_dict[k] for k in ops_dict])
    max_coeffs = ops.cumsum() / ops.sum()

    if model.ic_only:
        print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    else:
        print('sdn will be trained from scratch...(The SDN training)')

    for epoch in range(1, epochs+1):
        scheduler.step()
        cur_lr = utils.get_lr(optimizer)
        print(datetime.now())
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        if model.ic_only is False:
            # calculate the IC coeffs for this epoch for the weighted objective function
            cur_coeffs = 0.01 + epoch*(max_coeffs/epochs) # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            print('Cur coeffs: {}'.format(cur_coeffs))

        start_time = time.time()
        model.train()
        loader = get_loader(data, augment)
        for i, batch in enumerate(tqdm(loader, desc='[sdn-advtrain:{}]'.format(epoch))):
            if model.ic_only is False:
                total_loss = sdn_training_step_adv( \
                    optimizer, model, cur_coeffs, batch, \
                    attack, iteration, eps_step, eps_max, device=device)
            else:
                total_loss = sdn_ic_only_step_adv( \
                    optimizer, model, batch, \
                    attack, iteration, eps_step, eps_max, device=device)

            # if i % 100 == 0:
            #     print('Loss: {}: '.format(total_loss))

        top1_test, top5_test = sdn_test(model, data.test_loader, device)

        print('Top1 Test accuracies: {}'.format(top1_test))
        print('Top5 Test accuracies: {}'.format(top5_test))
        end_time = time.time()

        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
        print('Top1 Train accuracies: {}'.format(top1_train))
        print('Top5 Train accuracies: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)

        epoch_time = int(end_time-start_time)
        metrics['epoch_times'].append(epoch_time)
        print('Epoch took {} seconds.'.format(epoch_time))

        metrics['lrs'].append(cur_lr)

    return metrics

def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top5 = []
    for output_id in range(model.num_output):
        t1 = datasets.AverageMeter()
        t5 = datasets.AverageMeter()
        top1.append(t1)
        top5.append(t5)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = datasets.accuracy(cur_output, b_y, topk=(1, 5))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))

    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

    return top1_accs, top5_accs

def sdn_get_detailed_results(model, loader, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id*loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)
    return layer_correct, layer_wrong, layer_predictions, layer_confidence

def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = utils.get_confusion_scores(output, confusion_stats, device)

            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion

# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = utils.get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return (mean_con, std_con)

def sdn_test_early_exits(model, loader, device='cpu'):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = datasets.AverageMeter()
    top5 = datasets.AverageMeter()
    total_time = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            pred = output.max(1)[1][0].cpu().detach().numpy()
            preds.append(pred)
            labels.append(b_y.cpu().detach().numpy()[0])
            end_time = time.time()
            total_time+= (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1, prec5 = datasets.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    print(confusion_matrix(labels, preds))

    classes, dist = np.unique(preds, return_counts=True)
    print(f'Classes: {classes} - Dist: {dist}')

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, total_time


# ------------------------------------------------------------------------------
#   CNN training/test/misc. functions
# ------------------------------------------------------------------------------
def cnn_training_step(model, optimizer, data, labels, device='cpu'):
    b_x = data.to(device)   # batch x
    b_y = labels.to(device)   # batch y
    output = model(b_x)            # cnn final output
    criterion = utils.get_loss_criterion()
    loss = criterion(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients

def cnn_train(model, data, epochs, optimizer, scheduler, save_func, device='cpu'):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
        scheduler.step()

        cur_lr = utils.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print(datetime.now())
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y in tqdm(train_loader, desc='[cnn-train:{}]'.format(epoch)):
            cnn_training_step(model, optimizer, x, y, device)

        end_time = time.time()

        top1_test, top5_test = cnn_test(model, data.test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = cnn_test(model, train_loader, device)
        print('Top1 Train accuracy: {}'.format(top1_train))
        print('Top5 Train accuracy: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)
        epoch_time = int(end_time-start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)

        metrics['lrs'].append(cur_lr)

        """
            Comment added by ionmodo
            In case save_func is initialized with network_architectures.save_model in train_networks at the line
            metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, !!arcs.save_model!!!, device=device)
            then save_func in the next if MUST have 4 parameters, according to its definition in network_architectures.py:
            def save_model(model, model_params, models_path, model_name, epoch=-1):
        """
        if save_func is not None:
            save_func(model, epoch)
            # save_func(model=model, model_params=None, models_path=MISSING, model_name=MISSING, epoch=epoch)

    return metrics

def cnn_advtraining_step( \
    model, optimizer, data, labels, \
    attack, iteration, eps_step, eps_max, device='cpu'):
    # batch of x and y
    b_x = data.to(device)
    b_y = labels.to(device)

    # compute the loss for the clean output (final-cnn)
    output = model(b_x)
    criterion = utils.get_loss_criterion()
    loss = criterion(output, b_y)

    # create a batch of adversarial samples
    # Note - we forces the network is trained with PGD,
    #        as the vanilla network cannot be trained with PGD-avg...
    if 'PGD' in attack:
        # : adversarial samples
        b_advx = PGD( \
            b_x, None, criterion, \
            y=b_y, model=model, \
            eps=eps_max/255., steps=iteration, gamma=eps_step/255., randinit=True, \
            cuda=True if device != 'cpu' else False, cnn=True).data
        b_advx = b_advx.to(device)
        output_adv = model(b_advx)
        loss += criterion(output_adv, b_y)

    # Note: when we consider adaptive AT w.r.t the DeepSloth,
    #   1. we use the vanilla model (no AT)
    #   2. we use the AT model (trained on PGD)

    # update the model parameters by back prop...
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # done.

def cnn_advtrain( \
    model, data, epochs, optimizer, scheduler, save_func, \
    attack, iteration, eps_step, eps_max, device='cpu'):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
        scheduler.step()

        cur_lr = utils.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print(datetime.now())
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y in tqdm(train_loader, desc='[cnn-advtrain:{}]'.format(epoch)):
            cnn_advtraining_step( \
                model, optimizer, x, y, \
                attack, iteration, eps_step, eps_max, device)

        end_time = time.time()

        top1_test, top5_test = cnn_test(model, data.test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = cnn_test(model, train_loader, device)
        print('Top1 Train accuracy: {}'.format(top1_train))
        print('Top5 Train accuracy: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)
        epoch_time = int(end_time-start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)

        metrics['lrs'].append(cur_lr)

        """
            Comment added by ionmodo
            In case save_func is initialized with network_architectures.save_model in train_networks at the line
            metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, !!arcs.save_model!!!, device=device)
            then save_func in the next if MUST have 4 parameters, according to its definition in network_architectures.py:
            def save_model(model, model_params, models_path, model_name, epoch=-1):
        """
        if save_func is not None:
            save_func(model, epoch)
            # save_func(model=model, model_params=None, models_path=MISSING, model_name=MISSING, epoch=epoch)

    return metrics

def cnn_test_time(model, loader, device='cpu'):
    model.eval()
    top1 = datasets.AverageMeter()
    top5 = datasets.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            prec1, prec5 = datasets.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, total_time


def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = datasets.AverageMeter()
    top5 = datasets.AverageMeter()

    # all_gts = []
    # all_preds = []
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)

            # preds = output.max(1, keepdim=True)[1]
            # preds_np = preds.cpu().detach().numpy().flatten()
            # all_preds.extend(list(preds_np))

            # gt_np = b_y.cpu().detach().numpy().flatten()
            # all_gts.extend(list(gt_np))

            prec1, prec5 = datasets.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    # classes, dist = np.unique(all_preds, return_counts=True)
    # print(f'Pred Classes: {classes} - Dist: {dist}')

    # gt_classes, gt_dist = np.unique(all_gts, return_counts=True)
    # print(f'GT Classes: {gt_classes} - Dist: {gt_dist}')

    return top1_acc, top5_acc

def cnn_get_confidence(model, loader, device='cpu'):
    model.eval()
    correct = set()
    wrong = set()
    instance_confidence = {}
    correct_cnt = 0

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = nn.functional.softmax(output, dim=1)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            pred_prob = model_pred[0].to(device)

            is_correct = pred.eq(b_y.view_as(pred))
            correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)


    return correct, wrong, instance_confidence
