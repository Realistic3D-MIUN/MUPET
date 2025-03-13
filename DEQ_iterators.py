import torch

def MAPGD(NN,degraded,heresig, maxit, convmarg, stepsize, device, deg = torch.nn.Identity(), adj = torch.nn.Identity(),  stopmarg = 100, 
                inittype = 'randstart', x0 = None):
    ctr = 0
    if inittype == 'randstart':
        estimate = 0.5 +  torch.randn_like(adj(degraded))
    elif inittype == 'bigstepfirst':
        estimate = adj(degraded)
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad + heresig ** 2 * NN(estimate, heresig*torch.ones(estimate.shape[0], device = device))
    elif inittype == 'rand_thenbigstep':
        estimate = torch.randn_like(adj(degraded))
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad + heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
    elif inittype == 'standard' or inittype == 'virtobs':
        estimate = adj(degraded)
    elif inittype == 'usex0' and x0 is not None:
        estimate = x0
    else:
        raise('bad inittype')

    for ctr in range(1,maxit+1):
        datagrad = adj(deg(estimate) - degraded)
        reggrad = - heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
        update = stepsize * (datagrad + reggrad)
        estimate = estimate - update
        if torch.mean(update ** 2) < convmarg:
            return estimate, ctr, 1
        if torch.mean(estimate ** 2) > stopmarg:
            return estimate, maxit + 0.0001,  0
    return estimate, maxit, 0

def adamMAPGD(NN, degraded,heresig, maxit, convmarg, stepsize, device,
                deg = torch.nn.Identity(), adj = torch.nn.Identity(), stopmarg = 100,  
                inittype = 'randstart', beta1 = 0.9, beta2 = 0.999, x0 = None):
    ctr = 0
    if inittype == 'randstart':
        estimate = 0.5 +  torch.randn_like(adj(degraded))
    elif inittype == 'bigstepfirst':
        estimate = adj(degraded)
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad + heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
    elif inittype == 'rand_thenbigstep':
        estimate = torch.randn_like(adj(degraded))
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad + heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
    elif inittype == 'standard' or inittype == 'virtobs':
        estimate = adj(degraded)
    elif inittype == 'usex0':
        assert x0 is not None, "x0 should not be undefined if using x0 for initialization of DEQ"
        estimate = x0
    else:
        raise('bad inittype')

    eps = 1e-8
    mom1 = torch.zeros_like(estimate)
    mom2 = torch.zeros_like(estimate)
    for ctr in range(1,maxit + 1):
        g = adj(deg(estimate) - degraded) - heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
        mom1 = beta1 * mom1 + (1 - beta1) * g
        mom2 = beta2 * mom2 + (1 - beta2 ) * torch.square(g)
        cormom1 = mom1 /(1 - beta1 ** (ctr + 1))
        cormom2 = mom2 / (1 - beta2 ** (ctr + 1))
        update =  stepsize * cormom1 / (torch.sqrt(cormom2) + eps)
        estimate = estimate - update
        if torch.mean(update ** 2) < convmarg:
            return estimate, ctr, 1
        if torch.norm(estimate, float('inf')) > stopmarg:
            return estimate, maxit + 0.001,  0
    return estimate, maxit, 0
