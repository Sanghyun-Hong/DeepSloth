import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import utils

# these are the default params for num iters and eps_step
def craft_per_sample_perturb_attack( \
    model, test_loader, \
    max_iter=20, epsilon=10, eps_step=1.0, grad_sparsity=99, nbatch=10, device='cpu'):

    print (' [Vanilla - L1] start crafting per-sample attacks...')
    print (' - iter: {}'.format(max_iter))
    print (' -  eps: {}'.format(epsilon))
    print (' - step: {}'.format(eps_step))
    print (' - spar: {}'.format(grad_sparsity))
    model.eval()

    # max iter and save parameters
    attack_data = [] # returns perturbed data
    attack_labels = []

    # to measure the time
    start_time = time.time()

    # loop over the test set
    for batch_idx, batch in enumerate(test_loader):
        # if nbatch > 0 and batch_idx == nbatch: break

        attack_labels.extend(batch[1])

        # : original dataset
        b_x = batch[0].to(device, dtype=torch.float)
        b_y = batch[1].to(device, dtype=torch.long)
        if b_x.min() < 0 or b_x.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        delta = torch.zeros_like(b_x) # init delta
        adv = b_x.clone().detach().requires_grad_(True)

        for ii in tqdm(range(1, max_iter+1), desc='[V-l1-{}]'.format(batch_idx)):
            adv = adv.clone().detach().to(torch.float).requires_grad_(True)

            logits = model(adv)[-1] # only the last layer output
            loss = F.cross_entropy(logits, b_y, reduction='mean')

            # Define gradient of loss wrt input
            grad, = torch.autograd.grad(loss, [adv])
            grad_view = grad.view(grad.shape[0], -1)
            abs_grad = torch.abs(grad_view)

            k = int(grad_sparsity/100.0 * abs_grad.shape[1])
            percentile_value, _ = torch.kthvalue(abs_grad, k, keepdim=True)

            percentile_value = percentile_value.repeat(1, grad_view.shape[1])
            tied_for_max = torch.ge(abs_grad, percentile_value).int().float()
            num_ties = torch.sum(tied_for_max, dim=1, keepdim=True)

            optimal_perturbation = (torch.sign(grad_view) * tied_for_max) / num_ties
            optimal_perturbation = optimal_perturbation.view(grad.shape)

            # Add perturbation to original example to obtain adversarial example
            adv = adv + optimal_perturbation * eps_step
            adv = torch.clamp(adv, 0, 1)

            # Clipping perturbation eta to the l1-ball
            delta = adv - b_x
            delta = delta.renorm(p=1, dim=0, maxnorm=epsilon)
            adv = torch.clamp(b_x + delta, 0, 1)

            del loss, logits

        attack_data.append(adv.detach().cpu().numpy())

    # to measure the time
    termi_time = time.time()
    print (' [Vanilla - L1] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # return the data
    attack_data = np.vstack(attack_data)
    attack_labels = np.asarray(attack_labels)
    return attack_data, attack_labels
