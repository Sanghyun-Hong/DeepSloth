"""
    PGD attacks
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# ------------------------------------------------------------------------------
#   PGD attack and its variants in the TripleWins paper
# ------------------------------------------------------------------------------
def PGD( \
    x, preds, loss_fn, y=None, model=None, \
    eps=None, steps=3, gamma=None, norm='linf', \
    randinit=False, cuda=False, cnn=False, **kwargs):

    # convert to cuda...
    x_adv = x.clone()
    if cuda: x_adv = x_adv.cuda()

    # create an adv. example w. random init
    if randinit:
        x_rand = torch.rand(x_adv.shape)
        if cuda: x_rand = x_rand.cuda()
        x_adv += (2.0 * x_rand - 1.0) * eps
    x_adv = Variable(x_adv, requires_grad=True)

    # run steps
    for t in range(steps):
        out_adv_branch = model(x_adv)   # use the main branch
        if cnn:
            loss_adv = loss_fn(out_adv_branch, y)
        else:
            loss_adv = loss_fn(out_adv_branch[-1], y)
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

        # : compute based on the norm
        if 'linf' == norm:
            x_adv.data.add_(gamma * torch.sign(grad.data))
            _linfball_projection(x, eps, x_adv, in_place=True)

        elif 'l2' == norm:
            x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                        .norm(2, dim=-1).view(x_adv.shape[0], 1, 1, 1)
            x_adv.data.add_(gamma * x_add)
            x_adv = _l2_projection(x, eps, x_adv)

        elif 'l1' == norm:
            x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                        .norm(1, dim=-1).view(x_adv.shape[0], 1, 1, 1)
            x_adv.data.add_(gamma * x_add)
            x_adv = _l1_projection(x, eps, x_adv)

        else:
            assert False, ('Error: undefined norm for the attack - {}'.format(norm))

        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def PGD_avg( \
    x, preds, loss_fn, y=None, model=None, \
    eps=None, steps=3, gamma=None, norm='linf', \
    randinit=False, cuda=False, **kwargs):

    # convert to cuda...
    x_adv = x.clone()
    if cuda: x_adv = x_adv.cuda()

    # create an adv. example w. random init
    if randinit:
        x_rand = torch.rand(x_adv.shape)
        if cuda: x_rand = x_rand.cuda()
        x_adv += (2.0 * x_rand - 1.0) * eps
    x_adv = Variable(x_adv, requires_grad=True)

    # run steps
    for t in range(steps):
        out_adv_branch = model(x_adv)
        loss_adv = 0
        # : average the loss over the branches
        for i in range(len(out_adv_branch)):
             loss_adv += loss_fn(out_adv_branch[i], y) * (1.0/len(out_adv_branch))
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

        # : bound based on the norm
        if 'linf' == norm:
            x_adv.data.add_(gamma * torch.sign(grad.data))
            _linfball_projection(x, eps, x_adv, in_place=True)

        elif 'l2' == norm:
            x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                        .norm(2, dim=-1).view(x_adv.shape[0], 1, 1, 1)
            x_adv.data.add_(gamma * x_add)
            x_adv = _l2_projection(x, eps, x_adv)

        elif 'l1' == norm:
            x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                        .norm(1, dim=-1).view(x_adv.shape[0], 1, 1, 1)
            x_adv.data.add_(gamma * x_add)
            x_adv = _l1_projection(x, eps, x_adv)

        else:
            assert False, ('Error: undefined norm for the attack - {}'.format(norm))

        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def PGD_max( \
    x, preds, loss_fn, y=None, model=None, \
    eps=None, steps=3, gamma=None, norm='linf', \
    randinit=False, cuda=False, **kwargs):

    # convert to cuda...
    x_advs = [x.clone() for _ in range(model.num_output)]
    if cuda: x_advs = [each.cuda() for each in x_advs]

    # create the adv. example w. random init
    if randinit:
        x_rands = [torch.rand(each.shape) for each in x_advs]
        if cuda: x_rands = [each.cuda() for each in x_rands]
        x_advs  = [(each + (2.0 * each - 1.0) * eps) for each in x_advs]
    x_advs = [Variable(each, requires_grad=True) for each in x_advs]

    # run steps
    for t in range(steps):
        for i in range(model.num_output):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            out = out_adv_branch[i]
            loss_adv = loss_fn(out, y)
            grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

            # : bound based on the norm
            if 'linf' == norm:
                x_adv.data.add_(gamma * torch.sign(grad.data))
                _linfball_projection(x, eps, x_adv, in_place=True)

            elif 'l2' == norm:
                x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                            .norm(2, dim=-1).view(x_adv.shape[0], 1, 1, 1)
                x_adv.data.add_(gamma * x_add)
                x_adv = _l2_projection(x, eps, x_adv)

            elif 'l1' == norm:
                x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                            .norm(1, dim=-1).view(x_adv.shape[0], 1, 1, 1)
                x_adv.data.add_(gamma * x_add)
                x_adv = _l1_projection(x, eps, x_adv)

            else:
                assert False, ('Error: undefined norm for the attack - {}'.format(norm))

            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv

    # record average losses for each adv samples
    losses = []
    for i in range(model.num_output):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        # : compute the average loss
        for j in range(model.num_output):
            out = out_adv_branch[j]
            if j == 0:
                loss_adv  = F.cross_entropy(input=out, target=y, reduce=False)
            else:
                loss_adv += F.cross_entropy(input=out, target=y, reduce=False)
        losses.append(loss_adv)

    # choose the adv. sample by referencing average losses
    losses  = torch.stack(losses, dim=-1)
    x_advs  = torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1, dim=-1)
    idxs  = idxs.long().view(-1, 1)
    # hard-cord the image size - 3 x 32 x 32 for CIFAR-10
    idxs  = idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)
    return x_adv

def _tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]
    return res


"""
    Norm-based projections (ell-1, ell-2, and ell-inf)
"""
def _l1_projection(x_base, epsilon, x_adv):
    delta = x_adv - x_base

    # consider the batch run
    mask = delta.view(delta.shape[0], -1).norm(1, dim=1) <= epsilon

    # compute the scaling factor
    scaling_factor = delta.view(delta.shape[0], -1).norm(1, dim=1)
    scaling_factor[mask] = epsilon

    # scale delta based on the factor
    delta *= epsilon / scaling_factor.view(-1, 1, 1, 1)
    return (x_base + delta)

def _l2_projection(x_base, epsilon, x_adv):
    delta = x_adv - x_base

    # consider the batch run
    mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= epsilon

    # compute the scaling factor
    scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=1)
    scaling_factor[mask] = epsilon

    # scale delta based on the factor
    delta *= epsilon / scaling_factor.view(-1, 1, 1, 1)
    return (x_base + delta)

def _linfball_projection(center, radius, t, in_place=True):
    return _tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)
