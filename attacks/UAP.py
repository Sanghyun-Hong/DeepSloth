"""
    The adversarial attacks presented in "Universal Adversarial Perturbation" paper.
"""
# basic
import numpy as np
from tqdm import tqdm

# torch libs
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.autograd.gradcheck import zero_gradients

# custom libs
from datasets import NumpyDataset, TensorDataset


# ------------------------------------------------------------------------------
#   Attack functions (DeepFool and UAP)
# ------------------------------------------------------------------------------
def DeepFool( \
    data, model, num_classes=10, overshoot=0.02, max_iter=50, cuda=False):
    # make a forward pass
    net_output = model.forward(data.unsqueeze(0))[-1].detach().cpu().numpy().flatten()
    net_I      = (np.array(net_output)).flatten().argsort()[::-1]

    # compose adv. data and label
    adv_data   = data
    adv_label  = net_I[0]

    # make the np array for the perturbations
    w     = np.zeros(data.shape)
    r_tot = np.zeros(data.shape)

    liter = 0

    x = adv_data.unsqueeze(0)
    x = Variable(x, requires_grad=True)

    fs      = model.forward(x)[-1]
    fs_list = [fs[0, net_I[k]] for k in range(num_classes)]
    k_i     = adv_label

    # run...
    while k_i == adv_label and liter < max_iter:

        pert = np.inf
        fs[0, net_I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, net_I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, net_I[k]] - fs[0, net_I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i   = pert * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if cuda:
            adv_data = data + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            adv_data = data + (1 + overshoot) * torch.from_numpy(r_tot)

        x = adv_data
        x.requires_grad_()
        fs  = model.forward(x)[-1]
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        liter += 1

    # report the output perturbation
    r_tot = (1 + overshoot) * r_tot
    return r_tot, liter, adv_label, k_i, adv_data

def UAP(data, labels, sdn_model, \
         data_shape=(1, 3, 32, 32), nclasses=10, \
         min_fool=0.8, max_uiter=1000, max_diter=10, max_norm=1, \
         overshoot=0.02, cuda=False):

    uap_data = torch.zeros(data_shape)
    if cuda: uap_data = uap_data.cuda()
    uap_data = Variable(uap_data, requires_grad=True)

    # compose two loaders: one for the perturbation, and the other for the validation
    base_loader = DataLoader(TensorDataset(data, labels), shuffle=True, batch_size=1)
    test_loader = DataLoader(TensorDataset(data, labels), shuffle=True, batch_size=1)
    print (' : [UAP] Compose two dataloader w. [{}] items'.format(len(base_loader.dataset)))

    # init the UAP procedure
    fool_rate = 0.0
    iteration = 0

    # run the UAP process
    while fool_rate < min_fool and iteration < max_uiter:

        # : p-bar setup
        cur_pbar = tqdm(base_loader, desc=' : [UAP-{}]'.format(iteration))

        # : iterate over the data-loader
        for bidx, (bdata, blabels) in enumerate(cur_pbar):
            if cuda:
                bdata, blabels = bdata.cuda(), blabels.cuda()
            bdata, blabels = Variable(bdata, requires_grad=False), Variable(blabels)

            # :: do a prediction (without gradient)
            with torch.no_grad():
                org_outputs = sdn_model(bdata)
                adv_outputs = sdn_model(bdata + uap_data)
                org_predict = torch.argmax(org_outputs[-1], 1)
                adv_predict = torch.argmax(adv_outputs[-1], 1)

            # :: perturb when the prediction is the same
            if (org_predict == adv_predict):
                dr, dfiter, _, _, _ = DeepFool( \
                    (bdata + uap_data).detach()[0], sdn_model,
                    num_classes=nclasses, overshoot=overshoot, max_iter=max_diter, cuda=cuda)

                # > projection to the l-inf norm
                if (dfiter < max_diter - 1):
                    tmp_data  = torch.from_numpy(dr)
                    if cuda: tmp_data = tmp_data.cuda()
                    uap_data = uap_data + tmp_data
                    uap_data = _linfball_projection(uap_data, max_norm)

            # :: set the p-bar desc
            if (bidx % 2 == 0):
                cur_pbar.set_description(' : [UAP-{}/{:.4f}/{:.4f}]'.format( \
                    iteration, torch.norm(uap_data).detach().cpu().numpy(), fool_rate))

        # : end for bidx...

        fool_rate = _compute_fool_rate(test_loader, uap_data, sdn_model, cuda=cuda)
        iteration += 1
    # end while...

    print (' [UAP] create an UAP [norm-{:.4f}], with the fool-rate [{:.4f}]'.format( \
        torch.norm(uap_data).detach().cpu().numpy(), fool_rate))
    return uap_data.detach().cpu()

def _linfball_projection(t, radius):
    return torch.clamp(t, -radius, radius)

def _compute_fool_rate(testloader, uap_pert, model, cuda=False):
    fooled = 0.0

    # iterate over the loader
    for tdata, tlabels in testloader:
        # : compose as a variable
        if cuda:
            tdata, tlabels = tdata.cuda(), tlabels.cuda()
        tdata, tlabels = Variable(tdata, requires_grad=False), Variable(tlabels)

        # : do a prediction
        with torch.no_grad():
            org_predict = torch.argmax(model(tdata)[-1], 1)
            adv_predict = torch.argmax(model(tdata + uap_pert)[-1], 1)

        # : compute the fool rate
        if (org_predict != adv_predict): fooled += 1
    return (fooled / len(testloader.dataset))
