# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and SDNs and also plotting

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms

import os
import random
import os.path
import copy
import sys
import pickle
import itertools as it

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

from PIL import Image
from bisect import bisect_right
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader

# custom libs
import model_funcs as mf
import models
from datasets import CIFAR10, CIFAR100, TinyImagenet, SVHN


# ------------------------------------------------------------------------------
#   Pickle functions
# ------------------------------------------------------------------------------
def store_to_pickle(filename, data):
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile)
    # done.

def load_from_pickle(filename):
    data = None
    with open(filename, 'rb') as infile:
        data = pickle.load(infile)
    return data



"""
    Class logger to store the outputs
"""
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr
        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.flush()
        self.log.close()

def start_logger(filename):
    sys.stdout = Logger(filename, 'out')


# the learning rate scheduler
class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr,8)
            lrs.append(new_lr)
        return lrs

# flatten the output of conv layers for fully connected layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1

# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))


class ManualData(Dataset):
    def __init__(self, data, labels, device='cpu'):
        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.device = device
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        self.transforms = None

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])

    def get_item_w_transform(self, idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)
        return (self.transforms(img), target)

    def add_cifar10_transforms(self):
            # it is better to keep the dataset of numpy for transforms
            self.data = self.data.cpu().detach().numpy()
            self.labels = self.labels.cpu().detach().numpy()
            self.__getitem__ = self.get_item_w_transform
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_loader(dataset, batch_size=128, shuffle=False, device='cpu'):
        return DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)

# random seed functions - for reproducibility
def set_random_seed(seed=1221):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # done.

def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))

# misc. functions
def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])

def overlay_two_histograms(save_path, save_name, hist_first_values, hist_second_values, first_label, second_label, title):
    plt.hist([hist_first_values, hist_second_values], bins=25, label=[first_label, second_label])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()


def get_confusion_scores(outputs, normalize=None, device='cpu'):
    p = 1
    confusion_scores = torch.zeros(outputs[0].size(0))
    confusion_scores = confusion_scores.to(device)

    for output in outputs:
        cur_disagreement = nn.functional.pairwise_distance(outputs[-1], output, p=p)
        cur_disagreement = cur_disagreement.to(device)
        for instance_id in range(outputs[0].size(0)):
            confusion_scores[instance_id] +=  cur_disagreement[instance_id]

    if normalize is not None:
        for instance_id in range(outputs[0].size(0)):
            cur_confusion_score = confusion_scores[instance_id]
            cur_confusion_score = cur_confusion_score - normalize[0] # subtract mean
            cur_confusion_score = cur_confusion_score / normalize[1] # divide by the standard deviation
            confusion_scores[instance_id] = cur_confusion_score

    return confusion_scores


"""
    Dataset loaders
"""
def load_dataset(dataset, batch_size=128, doNormalization=True):
    if dataset == 'cifar10':
        return load_cifar10(batch_size, doNormalization)
    elif dataset == 'cifar100':
        return load_cifar100(batch_size, doNormalization)
    elif dataset == 'tinyimagenet':
        return load_tinyimagenet(batch_size, doNormalization)
    else:
        assert False, ('Error - undefined dataset name: {}'.format(dataset))

def load_cifar10(batch_size, normalize=True):
    return CIFAR10(batch_size=batch_size, doNormalization=normalize)

def load_cifar100(batch_size, normalize=True):
    return CIFAR100(batch_size=batch_size, doNormalization=normalize)

def load_tinyimagenet(batch_size, doNormalization=True):
    return TinyImagenet(batch_size=batch_size, doNormalization=doNormalization)



def get_output_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth

    #output_depths.append(total_depth)

    return np.array(output_depths)/total_depth, total_depth


# store-related functions
def create_folder(path):
    if not os.path.exists(path): os.makedirs(path)
    # done.


def model_exists(models_path, model_name):
    return os.path.isdir(models_path+'/'+model_name)

def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]

def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']

def load_optimizer(model, lr_params, stepsize_params):
    lr           = lr_params[0]
    weight_decay = lr_params[1]
    momentum     = lr_params[2]
    milestones   = stepsize_params[0]
    gammas       = stepsize_params[1]

    # batch normalization do not require gradients, so we filter them
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler

def load_sdn_ic_only_optimizer(model, lr_params, stepsize_params):
    freeze_except_outputs(model)

    lr=lr_params[0]
    weight_decay=lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    param_list = []
    for layer in model.layers:
        if layer.no_output == False:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})

    optimizer = Adam(param_list, lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler

def available_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


def get_loss_criterion():
    return CrossEntropyLoss()


def get_all_trained_models_info(models_path, use_profiler=False, device='gpu'):
    print('Testing all models in: {}'.format(models_path))

    for model_name in sorted(os.listdir(models_path)):
        try:
            model_params = arcs.load_params(models_path, model_name, -1)
            train_time = model_params['total_time']
            num_epochs = model_params['epochs']
            architecture = model_params['architecture']
            print(model_name)
            task = model_params['task']
            print(task)
            net_type = model_params['network_type']
            print(net_type)

            top1_test = model_params['test_top1_acc']
            top1_train = model_params['train_top1_acc']
            top5_test = model_params['test_top5_acc']
            top5_train = model_params['train_top5_acc']


            print('Top1 Test accuracy: {}'.format(top1_test[-1]))
            print('Top5 Test accuracy: {}'.format(top5_test[-1]))
            print('\nTop1 Train accuracy: {}'.format(top1_train[-1]))
            print('Top5 Train accuracy: {}'.format(top5_train[-1]))

            print('Training time: {}, in {} epochs'.format(train_time, num_epochs))

            if use_profiler:
                model, _ = arcs.load_model(models_path, model_name, epoch=-1)
                model.to(device)
                input_size = model_params['input_size']

                if architecture == 'dsn':
                    total_ops, total_params = p.profile_sdn(model, input_size, device)
                    print("#Ops (GOps): {}".format(total_ops))
                    print("#Params (mil): {}".format(total_params))

                else:
                    total_ops, total_params = p.profile(model, input_size, device)
                    print("#Ops: %f GOps"%(total_ops/1e9))
                    print("#Parameters: %f M"%(total_params/1e6))

            print('------------------------')
        except:
            print('FAIL: {}'.format(model_name))
            continue


def sdn_prune(sdn_path, sdn_name, prune_after_output, epoch=-1, preloaded=None):
    print('Pruning an SDN...')

    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    output_layer = get_nth_occurance_index(sdn_model.add_output, prune_after_output)

    pruned_model = copy.deepcopy(sdn_model)
    pruned_params = copy.deepcopy(sdn_params)

    new_layers = nn.ModuleList()
    prune_add_output = []

    for layer_id, layer in enumerate(sdn_model.layers):
        if layer_id == output_layer:
            break
        new_layers.append(layer)
        prune_add_output.append(sdn_model.add_output[layer_id])

    last_conv_layer = sdn_model.layers[output_layer]
    end_layer = copy.deepcopy(last_conv_layer.output)

    last_conv_layer.output = nn.Sequential()
    last_conv_layer.forward = last_conv_layer.only_forward
    last_conv_layer.no_output = True
    new_layers.append(last_conv_layer)

    pruned_model.layers = new_layers
    pruned_model.end_layers = end_layer

    pruned_model.add_output = prune_add_output
    pruned_model.num_output = prune_after_output + 1

    pruned_params['pruned_after'] = prune_after_output
    pruned_params['pruned_from'] = sdn_name

    return pruned_model, pruned_params


"""
    Functions for converting CNNs <-> SDNs
"""
def cnn_to_sdn(cnn_path, cnn_name, sdn_params, epoch=-1, preloaded=None):
    print ('[cnn_to_sdn] convert a CNN to an SDN...')
    if preloaded is None:
        cnn_model, _ = models.load_model(cnn_path, cnn_name, epoch=epoch)
    else:
        cnn_model = preloaded

    sdn_params['architecture'] = 'sdn'
    sdn_params['converted_from'] = cnn_name
    sdn_model = (models.load_sdn(cnn_model))(sdn_params)
    sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, cnn_layer in enumerate(cnn_model.layers):
        sdn_layer = sdn_model.layers[layer_id]
        sdn_layer.layers = cnn_layer.layers
        layers.append(sdn_layer)

    sdn_model.layers = layers
    sdn_model.end_layers = cnn_model.end_layers
    return sdn_model, sdn_params

def sdn_to_cnn(sdn_path, sdn_name, epoch=-1, preloaded=None):
    print ('[sdn_to_cnn] convert an SDN to a CNN...')
    if preloaded is None:
        sdn_model, sdn_params = models.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model  = preloaded[0]
        sdn_params = preloaded[1]

    cnn_params = copy.deepcopy(sdn_params)
    cnn_params['architecture'] = 'cnn'
    cnn_params['converted_from'] = sdn_name
    cnn_model = models.load_cnn(sdn_model)(cnn_params)
    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)

    cnn_model.layers = layers
    cnn_model.end_layers = sdn_model.end_layers
    return cnn_model, cnn_params


def freeze_except_outputs(model):
    model.frozen = True
    for param in model.init_conv.parameters():
        param.requires_grad = False

    for layer in model.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.end_layers.parameters():
        param.requires_grad = False


def save_tinyimagenet_classname():
    filename = 'tinyimagenet_classes'
    dataset = load_dataset('tinyimagenet')
    tinyimagenet_classes = {}

    for index, name in enumerate(dataset.testset_paths.classes):
        tinyimagenet_classes[index] = name

    with open(filename, 'wb') as f:
        pickle.dump(tinyimagenet_classes, f, pickle.HIGHEST_PROTOCOL)

def get_tinyimagenet_classes(prediction=None):
    filename = 'tinyimagenet_classes'
    with open(filename, 'rb') as f:
        tinyimagenet_classes = pickle.load(f)

    if prediction is not None:
        return tinyimagenet_classes[prediction]

    return tinyimagenet_classes


def loader_inst_counter(loader):
    num_instances = 0
    for batch in loader:
        num_instances += len(batch[1])
    return num_instances



def file_exists(filename):
    return os.path.isfile(filename)


def CrossEntropyLogitsWithUniform(predicted):
    p = nn.functional.softmax(predicted, dim=1)
    t = (torch.ones(predicted.shape)/predicted.shape[1]).to(predicted.device)
    return -(t * torch.log(p+1e-9)).sum(dim=1).mean()

def save_batch_of_tensor_images(save_path, data):
    vutils.save_image(data, nrow=16, filename=f'{save_path}.png', normalize=True, range=(0,1))


def get_task_num_classes(task):
    if task == 'cifar10':
        return 10
    elif task == 'tinyimagenet':
        return 200

def get_task_class_data(task, get_class=None, msdnet=False):
    if task == 'cifar10':
        return get_cifar10_class_data(get_class, msdnet)
    elif task == 'tinyimagenet':
        return get_tinyimagenet_class_data(get_class, msdnet)


def get_cifar10_class_data(get_class=0, msdnet=False):
    if not msdnet:
        path = os.path.join('datasets', 'originals', 'cifar10')
    else:
        path = os.path.join('..', 'datasets', 'originals', 'cifar10')

    train_dataset = datasets.CIFAR10(path, train=True)
    test_dataset = datasets.CIFAR10(path, train=False)

    train_data, test_data = (train_dataset.data/ 255) , (test_dataset.data / 255)
    train_data, test_data = train_data.transpose((0,3,1,2)), test_data.transpose((0,3,1,2))
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)

    train_indices = np.where(train_labels == get_class)[0]
    test_indices = np.where(test_labels == get_class)[0]

    train_data, train_labels, test_data, test_labels = train_data[train_indices], train_labels[train_indices], test_data[test_indices], test_labels[test_indices]
    # print(f'train data {train_data.shape}, test data: {test_data.shape}, train labels: {train_labels.shape}, test labels: {test_labels.shape}')

    return train_data, train_labels, test_data, test_labels


def get_tinyimagenet_class_data(get_class=0, msdnet=False):
    if not msdnet:
        path = os.path.join('datasets', 'originals', 'tiny-imagenet-200')
    else:
        path = os.path.join('..', 'datasets', 'originals', 'tiny-imagenet-200')

    train_dir = os.path.join(path, 'train')
    valid_dir = os.path.join(path, 'val', 'images')

    classes = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    classes.sort()

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    class_path = classes[get_class]
    train_target_dir = os.path.join(train_dir, class_path, 'images')
    for root, _, fnames in sorted(os.walk(train_target_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            image = default_loader(path)
            train_data.append(np.array(image).transpose(2, 0, 1)/255)
            train_labels.append(get_class)

    test_target_dir = os.path.join(valid_dir, class_path)
    for root, _, fnames in sorted(os.walk(test_target_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            image = default_loader(path)
            test_data.append(np.array(image).transpose(2, 0, 1)/255)
            test_labels.append(get_class)

    train_data, train_labels, test_data, test_labels = np.asarray(train_data), np.asarray(train_labels), np.asarray(test_data), np.asarray(test_labels)

    # print(f'train data {train_data.shape}, test data: {test_data.shape}, train labels: {train_labels.shape}, test labels: {test_labels.shape}')
    return train_data, train_labels, test_data, test_labels
