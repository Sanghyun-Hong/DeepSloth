"""
    Dataset-related functions
"""
# basics
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# torch
import torch
from torch.utils.data import Dataset, DataLoader, sampler, Subset
from torchvision import datasets, transforms, utils


# ------------------------------------------------------------------------------
#   Global locations for downloading datasets
# ------------------------------------------------------------------------------
_svhn     = os.path.join('datasets', 'originals', 'svhn')
_mnist    = os.path.join('datasets', 'originals', 'mnist')
_cifar10  = os.path.join('datasets', 'originals', 'cifar10')
_cifar100 = os.path.join('datasets', 'originals', 'cifar100')
_tinynet  = os.path.join('datasets', 'originals', 'tiny-imagenet-200')


# ------------------------------------------------------------------------------
#   Dataset classes
# ------------------------------------------------------------------------------
class SVHN:
    def __init__(self, batch_size=128):
        # http://ufldl.stanford.edu/housenumbers/
        # https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.SVHN
        # print('Loading SVHN (Street View House Numbers)...(LOADING TESTSET ONLY)')
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test  = 26032
        self.num_train = 73257
        self.num_extra = 531131

        self.trainset = datasets.SVHN(root=_svhn, split='train', download=False, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=False, num_workers=8)

        self.testset = datasets.SVHN(root=_svhn, split='test', download=dw, transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=8)

class MNIST:
    def __init__(self, batch_size=512, doNormalization=True, shuffle=True):
        print("MNIST::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size  = batch_size
        self.img_size    = 28
        self.num_classes = 10
        self.num_test    = 10000
        self.num_train   = 60000

        # preprocessing
        preprocList = [transforms.ToTensor()]
        if doNormalization:
            preprocList.append(transforms.Normalize(mean=[0.1307,], std=[0.3081,]))

        # normalizations...
        self.normalized = transforms.Compose(preprocList)

        self.trainset = datasets.MNIST(root=_mnist, train=True, download=True, transform=self.normalized)
        if shuffle:
            self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        else:
            self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=False)

        self.testset = datasets.MNIST(root=_mnist, train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

class CIFAR10:
    # Added by ionutmodo
    # added parameter to control normalization for training data. By default, CIFAR10 images have pixels in [0, 1]
    # implicit value was set to True to have the same functionality for all models
    def __init__(self, batch_size=128, doNormalization=True, add_trigger=False):
        print("CIFAR10::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        if doNormalization:
            preprocList.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        dw = True

        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter

        self.aug_trainset = datasets.CIFAR10(root=_cifar10, train=True, download=dw, transform=self.augmented)
        # self.aug_trainset = torch.utils.data.Subset(self.aug_trainset, np.random.randint(low=0, high=self.num_train, size=100))  # ionut: sample the dataset for faster training during my tests
        # print('[CIFAR10] Subsampling aug_trainset...')
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True)

        self.trainset = datasets.CIFAR10(root=_cifar10, train=True, download=dw, transform=self.normalized)
        # self.trainset = torch.utils.data.Subset(self.trainset, np.random.randint(low=0, high=self.num_train, size=100)) # ionut: sample the dataset for faster training during my tests
        # print('[CIFAR10] Subsampling trainset...')
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=False)

        self.testset = datasets.CIFAR10(root=_cifar10, train=False, download=dw, transform=self.normalized)
        # self.testset = torch.utils.data.Subset(self.testset, np.random.randint(low=0, high=self.num_test, size=100)) # ionut: sample the dataset for faster inference during my tests
        # print('[CIFAR10] Subsampling testset...')
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False)

class CIFAR100:
    def __init__(self, batch_size=128, doNormalization=False):
        print("CIFAR100::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000

        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        if doNormalization:
            preprocList.append(transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]))

        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter

        self.aug_trainset =  datasets.CIFAR100(root=_cifar100, train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.CIFAR100(root=_cifar100, train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.testset =  datasets.CIFAR100(root=_cifar100, train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TinyImagenet():
    def __init__(self, dataroot=None, batch_size=128, doNormalization=True):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000

        if not dataroot:
            train_dir = os.path.join(_tinynet, 'train')
            valid_dir = os.path.join(_tinynet, 'val', 'images')
        else:
            train_dir = os.path.join(dataroot, 'train')
            valid_dir = os.path.join(dataroot, 'val', 'images')

        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        if doNormalization:
            preprocList.append(transforms.Normalize(mean=[0.4802,  0.4481,  0.3975], std=[0.2302, 0.2265, 0.2262]))

        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2)] + preprocList)
        self.normalized = transforms.Compose(preprocList)   # contains normalization depending on doNormalization parameter

        self.aug_trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=8)

        self.trainset =  datasets.ImageFolder(train_dir, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset =  datasets.ImageFolder(valid_dir, transform=self.normalized)
        self.testset_paths = ImageFolderWithPaths(valid_dir, transform=self.normalized)

        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False)


# ------------------------------------------------------------------------------
#   Misc. functions
# ------------------------------------------------------------------------------
def load_train_loader(dataset, nbatch=1, normalize=False):
    if 'mnist' == dataset:
        return MNIST(batch_size=nbatch, doNormalization=normalize).train_loader
    elif 'cifar10' == dataset:
        return CIFAR10(batch_size=nbatch, doNormalization=normalize).train_loader
    elif 'cifar100' == dataset:
        return CIFAR100(batch_size=nbatch, doNormalization=normalize).train_loader
    elif 'tinyimagenet' == dataset:
        return TinyImagenet(batch_size=nbatch, doNormalization=normalize).train_loader
    else:
        assert False, ('Error: unsupported dataset - {}'.format(dataset))
    # done

def load_valid_loader(dataset, nbatch=1, normalize=False):
    if 'mnist' == dataset:
        return MNIST(batch_size=nbatch, doNormalization=normalize).test_loader
    elif 'cifar10' == dataset:
        return CIFAR10(batch_size=nbatch, doNormalization=normalize).test_loader
    elif 'cifar100' == dataset:
        return CIFAR100(batch_size=nbatch, doNormalization=normalize).test_loader
    elif 'tinyimagenet' == dataset:
        return TinyImagenet(batch_size=nbatch, doNormalization=normalize).test_loader
    else:
        assert False, ('Error: unsupported dataset - {}'.format(dataset))
    # done


# ------------------------------------------------------------------------------
#   Numpy dataset, for loading the adversarial samples in [0,1]
# ------------------------------------------------------------------------------
class NumpyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data      = data
        self.labels    = labels
        self.transform = transform
        print (' [NumpyDataset] load the [{}] data'.format(len(self.labels)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]

        # conversion
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)
        return data, label

class TensorDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data      = data
        self.labels    = labels
        self.transform = transform
        print (' [TensorDataset] load the [{}] data'.format(len(self.labels)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]

        # transform
        if self.transform is not None:
            data = self.transform(data)
        return data, label

def _compose_samples(dataloader, indexes):
    sample_data   = []
    sample_labels = []

    # loop over the loader
    for didx, (data, labels) in tqdm( \
        enumerate(dataloader), desc=' : [compose-samples]'):
        if (indexes is not None) and (didx not in indexes): continue
        sample_data.append(data.clone())
        sample_labels.append(labels.clone())
    # end for didx....

    # convert to the tensor
    sample_data   = torch.cat(sample_data, dim=0)
    sample_labels = torch.cat(sample_labels, dim=0)
    return sample_data, sample_labels

def create_val_folder(path, filename):
    """
        This method is responsible for separating validation images into separate sub folders
    """
    fp = open(filename, "r")    # open file in read mode
    data = fp.readlines()       # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in tqdm(val_img_dict.items(), desc='[create-val-folder]'):
        newpath = (os.path.join(path, folder))

        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


# ------------------------------------------------------------------------------
#   Tools for computing the accuracy
# ------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_w_preds(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
