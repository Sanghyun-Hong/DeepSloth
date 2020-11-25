"""
    A script that partitions the dataset for transferability scenarios
"""
# basics
import numpy as np
from PIL import Image

# torch...
import torch

# custom libs
import utils


# ------------------------------------------------------------------------------
#   Misc. functions
# ------------------------------------------------------------------------------
def update_numpy(acc, term, func):
    if acc is None:
        acc = term
    else:
        acc = func((acc, term))
    return acc


def get_class_wise_lists(n_classes_cifar10, return_test=False):
    if not return_test:
        class_wise_dataset = []
        for n_class in range(n_classes_cifar10):
            train_data, train_labels, _, _ = af.get_cifar10_class_data(n_class)  # don't use
            class_wise_dataset.append((train_data, train_labels))
        return class_wise_dataset

    else:
        class_wise_dataset = []
        test_class_wise_dataset = []
        for n_class in range(n_classes_cifar10):
            train_data, train_labels, test_data, test_labels = af.get_cifar10_class_data(n_class)  # don't use
            class_wise_dataset.append((train_data, train_labels))
            test_class_wise_dataset.append((test_data, test_labels))

    return class_wise_dataset, test_class_wise_dataset


# ------------------------------------------------------------------------------
#   Scenario related...
# ------------------------------------------------------------------------------
def scenario_1_split(int_percentages=None):
    np.random.seed(0)
    """
        Scenario 1) Train CIFAR10 models that use 10%, 25%, 50% of the full training set.
        Chooses p% of data in each class (and corresponding labels)

        Parameter int_percentages contains percentages as integers, NOT FLOATS!

        Returns:
        - percent_loaders (dict): each key p% contains an af.ManualData object containing p% of dataset (p% from each label)
            * Loader data contains p% of images (p% of class 0, ..., p% of class 9) - consecutive
            * Loader labels (np.ndarray): contains p% of labels (p%  0s, ..., p% 9s) - consecutive
    """
    if int_percentages is None:
        int_percentages = [10, 25, 50, 100]

    print('Running scenario_1_split\n')
    n_classes_cifar10 = 10
    # get a list containing CIFAR10 data class by class (class k at index k)
    class_wise_dataset = get_class_wise_lists(n_classes_cifar10)

    percent_loaders = {}
    for p in int_percentages:
        subset_data = None
        subset_labels = None
        for n_class in range(n_classes_cifar10):
            crt_train_data, crt_train_labels = class_wise_dataset[n_class]

            count = crt_train_data.shape[0]
            how_many_2_choose = int(count * p / 100.0)

            indexes = np.random.choice(np.arange(count), how_many_2_choose, replace=False)

            subset_data = update_numpy(acc=subset_data, term=np.copy(crt_train_data[indexes]), func=np.vstack)
            subset_labels = update_numpy(acc=subset_labels, term=np.copy(crt_train_labels[indexes]), func=np.hstack)
        # end for n_class

        print(f'p={p}, data: {subset_data.shape}, labels: {subset_labels.shape}\n')
        percent_loaders[p] = af.ManualData(data=subset_data, labels=subset_labels)
    # end for p
    np.random.seed(af.get_random_seed())
    return percent_loaders


def scenario_2_split(int_classes=None):
    np.random.seed(0)
    """
        Scenario 2) Split CIFAR10 training set into non-overlapping 5 classes - 5 classes, 6 - 6 and 7 - 7.
        Parameter int_classes_left:
            - each value c is used to generate the two datasets that contain c classes
        Returns:
            - percent_loaders (dict): each key c contains a pair of af.ManualData meaning ( Dataset w c classes, another dataset c classes)
                * Loader data contains p% of images (p% of class 0, ..., p% of class 9) - consecutive
                * Loader labels (np.ndarray): contains p% of labels (p%  0s, ..., p% 9s) - consecutive
    """
    if int_classes is None:
        int_classes = [5, 6, 7]

    print('Running scenario_2_split\n')
    n_classes_cifar10 = 10
    # get a list containing CIFAR10 data class by class (class k at index k)
    class_wise_dataset, test_class_wise_dataset = get_class_wise_lists(n_classes_cifar10, return_test=True)

    all_classes = np.arange(n_classes_cifar10)

    class_loaders = {}

    for classes in int_classes:


        num_class_overlap = 2*(classes - 5)

        class_indexes_overlap = np.random.choice(all_classes, num_class_overlap, replace=False)
        left_unique_classes = np.random.choice([x for x in all_classes if x not in class_indexes_overlap], classes-num_class_overlap, replace=False)
        right_unique_classes = [x for x in all_classes if (x not in class_indexes_overlap) and (x not in left_unique_classes)]


        class_indexes_left = np.array(list(left_unique_classes) + list(class_indexes_overlap))
        class_indexes_right = np.array(list(right_unique_classes) + list(class_indexes_overlap))

        print(class_indexes_left)
        print(class_indexes_right)

        subset_data_left, subset_labels_left = None, None
        subset_data_right, subset_labels_right = None, None
        subset_test_data_left, subset_test_labels_left = None, None
        subset_test_data_right, subset_test_labels_right = None, None


        label_left = 0
        label_right = 0

        for n_class in all_classes:
            crt_train_data, crt_train_labels = class_wise_dataset[n_class]
            crt_test_data, crt_test_labels = test_class_wise_dataset[n_class]

            if n_class in class_indexes_left:
                new_train_labels = np.ones(crt_train_labels.shape) * label_left # we have to relabel the dataset because pytorch expects labels as 0,1,2,3,...
                subset_data_left = update_numpy(acc=subset_data_left, term=np.copy(crt_train_data), func=np.vstack)
                subset_labels_left = update_numpy(acc=subset_labels_left, term=np.copy(new_train_labels), func=np.hstack)

                new_test_labels = np.ones(crt_test_labels.shape) * label_left
                subset_test_data_left = update_numpy(acc=subset_test_data_left, term=np.copy(crt_test_data), func=np.vstack)
                subset_test_labels_left = update_numpy(acc=subset_test_labels_left, term=np.copy(new_test_labels), func=np.hstack)
                label_left += 1


            if n_class in class_indexes_right:
                new_train_labels = np.ones(crt_train_labels.shape) * label_right # we have to relabel the dataset because pytorch expects labels as 0,1,2,3,...
                subset_data_right = update_numpy(acc=subset_data_right, term=np.copy(crt_train_data), func=np.vstack)
                subset_labels_right = update_numpy(acc=subset_labels_right, term=np.copy(new_train_labels), func=np.hstack)

                new_test_labels = np.ones(crt_test_labels.shape) * label_right
                subset_test_data_right = update_numpy(acc=subset_test_data_right, term=np.copy(crt_test_data), func=np.vstack)
                subset_test_labels_right = update_numpy(acc=subset_test_labels_right, term=np.copy(new_test_labels), func=np.hstack)
                label_right += 1
        # end for n_class

        print(f'{classes}: train - data-left: {subset_data_left.shape}, labels-left: {subset_labels_left.shape}, data-right: {subset_data_right.shape}, labels-right: {subset_labels_right.shape}\n')

        print(f'{classes}: test - data-left: {subset_test_data_left.shape}, labels-left: {subset_test_labels_left.shape}, data-right: {subset_test_data_right.shape}, labels-right: {subset_test_labels_right.shape}\n')


        loaders_left = (af.ManualData(data=subset_data_left, labels=subset_labels_left), af.ManualData(data=subset_test_data_left, labels=subset_test_labels_left))
        loaders_right = (af.ManualData(data=subset_data_right, labels=subset_labels_right),  af.ManualData(data=subset_test_data_right, labels=subset_test_labels_right))

        class_loaders[classes] = (loaders_left, loaders_right)

    np.random.seed(af.get_random_seed())

    # end for class_left, class_right
    return class_loaders