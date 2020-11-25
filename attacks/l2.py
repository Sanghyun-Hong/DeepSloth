import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import utils


def craft_per_sample_perturb_attack( \
    model, test_loader, \
    max_iter=1000, gamma=0.05, init_norm=1., quantize=True, levels=256, nbatch=10, device='cpu'):
    print (' [Vanilla - L2] start crafting per-sample attacks...')
    model.eval()

    # max iter and save parameters
    attack_data = [] # returns perturbed attack data (the best adversarial examples)
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

        batch_size = b_x.shape[0]
        delta = torch.zeros_like(b_x, requires_grad=True)
        norm = torch.full((batch_size,), init_norm, device=device, dtype=torch.float)
        worst_norm = torch.max(b_x, 1 - b_x).view(batch_size, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(b_x)
        adv_found = torch.zeros(b_x.size(0), dtype=torch.uint8, device=device)

        for i in tqdm(range(max_iter), desc='[V-l2-{}]'.format(batch_idx)):
            l2 = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            adv = b_x + delta
            logits = model(adv)[-1] # only the last layer output
            pred_labels = logits.argmax(1)
            loss = -1  * F.cross_entropy(logits, b_y, reduction='sum') # we want to increase this loss hence the -1

            is_adv = (pred_labels != b_y)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = 1
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(b_x)
            if quantize:
                delta.data.mul_(levels - 1).round_().div_(levels - 1)
            delta.data.clamp_(0, 1).sub_(b_x)
            scheduler.step()

        best_adv = b_x + best_delta
        attack_data.append(best_adv.detach().cpu().numpy())

    # to measure the time
    termi_time = time.time()
    print (' [Vanilla - L2] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # return the data
    attack_data = np.vstack(attack_data)
    attack_labels = np.asarray(attack_labels)
    return attack_data, attack_labels
