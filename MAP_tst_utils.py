import torch
import os
import deepinv
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import time


def prox(degraded , input , deg, adj, param, stepsize = 0.3):
    current = input
    stepdiff = 1e2
    ctr = 0
    while stepdiff > 1e-5 and ctr < 1e2:
        newest = current - stepsize * (adj(deg(current)- degraded) + param * (current - input))
        stepdiff = torch.mean((newest - current) ** 2)
        current = newest
        ctr += 1
    return newest

def getpsnr(img1, img2):
    mse = torch.mean((img1-img2)**2, dim=[1,2,3])
    psnr = 10* torch.log10(1 / mse)
    return psnr

def getavgpsnr(input1, input2):
    return torch.mean(getpsnr(input1, input2))

def MAPGD(NN, degraded, GT,AWGNsig, maxit, stepsize,  MSEloss , ssimloss , lpipsloss, device, maxsig = 50, downscale = False, fac = 1, deg = torch.nn.Identity(), adj = torch.nn.Identity(),
                inittype = 'randstart', x0 = None, addnoise = False, log_path = None ,log_freq = 1e7, lamb = 1):
    ctr = 0
    clampedGT = torch.clamp(GT, min=0, max=1)
    if inittype == 'randstart':
        estimate = 0.5 + torch.randn_like(GT)
    elif inittype == 'bigstepfirst':
        estimate = adj(degraded) + AWGNsig ** 2 * NN(adj(degraded), AWGNsig*torch.ones(GT.shape[0], device = device))
    elif inittype == 'standard' or inittype == 'virtobs':
        estimate = adj(degraded)
    elif inittype == 'usex0' and x0 is not None:
        estimate = x0
    elif inittype == 'noisyx0' and x0 is not None:
        estimate = x0 + AWGNsig * torch.randn_like(x0)
    elif inittype == 'maxrand':
        estimate = 0.5 + maxsig *torch.randn_like(GT)
    else:
        raise NotImplementedError(f'initialization type {inittype} has not been implemented')
    if downscale:
        usedsig = maxsig
    else:
        usedsig = AWGNsig
    save_image(estimate , log_path  + 'x0.png')
    stepmarg = torch.zeros(maxit)
    PSNRvals = torch.zeros(maxit+1, device = device)
    lpipsvals = torch.zeros(maxit+1, device = device)
    ssimvals = torch.zeros(maxit+1, device = device)
    clampedest = torch.clamp(estimate, min=0, max=1)
    hereloss = MSEloss(clampedest, clampedGT)
    PSNRvals[ctr] = getavgpsnr(clampedest, clampedGT)
    lpipsvals[ctr] = lpipsloss(clampedest, clampedGT)
    ssimvals[ctr] = ssimloss(clampedest, clampedGT)
    bestloss = 1e4
    for ctr in range(1,maxit+1):
        datagrad = adj(deg(estimate) - degraded) / lamb
        reggrad = - usedsig ** 2 * NN(estimate, usedsig*torch.ones(estimate.shape[0], device = device))
        update = stepsize * (datagrad + reggrad)
        if addnoise:
            update = update + (2 * stepsize) ** 0.5 * usedsig * torch.randn_like(estimate)
        newest = estimate - update 
        estimate = newest
        clampedest = torch.clamp(estimate, min=0, max=1)
        hereloss = MSEloss(clampedest, clampedGT)
        if hereloss < bestloss:
            bestloss = hereloss
            bestest = estimate
        PSNRvals[ctr] = getavgpsnr(clampedest, clampedGT)
        lpipsvals[ctr] = lpipsloss(clampedest, clampedGT)
        ssimvals[ctr] = ssimloss(clampedest, clampedGT)
        stepmarg[ctr-1] = torch.mean(update ** 2)
        if ctr % log_freq == 0:
            save_image(estimate , log_path + str(ctr) + 'MAPit.png')
        usedsig = max(usedsig * fac, AWGNsig)
    return bestest, PSNRvals, lpipsvals,  ssimvals, stepmarg, estimate


def adamMAPGD(NN, degraded, GT,AWGNsig, maxit, stepsize,  MSEloss , ssimloss , lpipsloss, device, maxsig = 50, downscale = False, fac = 1,
                deg = torch.nn.Identity(), adj = torch.nn.Identity(),inittype = 'randstart', beta1 = 0.9, 
                beta2 = 0.999, x0 = None, addnoise = False, log_path = None ,log_freq = 1e7, lamb = 1):
    ctr = 0
    clampedGT = torch.clamp(GT, min=0, max=1)
    if inittype == 'randstart':
        estimate = 0.5 + torch.randn_like(GT)
    elif inittype == 'bigstepfirst':
        print(degraded.shape[0])
        estimate = adj(degraded) + AWGNsig ** 2 *  NN(adj(degraded), AWGNsig*torch.ones(GT.shape[0], device = device))
    elif inittype == 'standard' or inittype == 'virtobs':
        estimate = adj(degraded)
    elif inittype == 'usex0' and x0 is not None:
        estimate = x0
    elif inittype == 'noisyx0' and x0 is not None:
        estimate = x0 + AWGNsig * torch.randn_like(x0)
    elif inittype == 'maxrand':
        estimate = 0.5 + maxsig *torch.randn_like(GT)
    else:
        raise NotImplementedError(f'initialization type {inittype} has not been implemented')
    save_image(estimate , log_path + 'x0.png')
    if downscale:
        usedsig = maxsig
    else:
        usedsig = AWGNsig
    eps = 1e-8
    mom1 = torch.zeros_like(GT)
    mom2 = torch.zeros_like(GT)
    stepmarg = torch.zeros(maxit)
    PSNRvals = torch.zeros(maxit+1, device = device)
    lpipsvals = torch.zeros(maxit+1, device = device)
    ssimvals = torch.zeros(maxit+1, device = device)
    clampedest = torch.clamp(estimate, min=0, max=1)
    hereloss = MSEloss(clampedest, clampedGT)
    bestloss = 1e9
    PSNRvals[ctr] = getavgpsnr(clampedest, clampedGT)
    #lpipsvals[ctr] = lpipsloss(2*estimate-1, 2*GT-1).mean()
    lpipsvals[ctr] = lpipsloss(clampedest, clampedGT)
    ssimvals[ctr] = ssimloss(clampedest, clampedGT)
    for ctr in range(1,maxit + 1):
        datagrad = adj(deg(estimate) - degraded) / lamb
        reggrad = - usedsig ** 2 * NN(estimate, usedsig*torch.ones(estimate.shape[0], device = device))
        g = datagrad + reggrad
        if addnoise:
            g = g + (2 * stepsize) ** 0.5 * usedsig * torch.randn_like(estimate)
        mom1 = beta1 * mom1 + (1 - beta1) * g
        mom2 = beta2 * mom2 + (1 - beta2 ) * torch.square(g)
        cormom1 = mom1 /(1 - beta1 ** (ctr + 1))
        cormom2 = mom2 / (1 - beta2 ** (ctr + 1))
        update =  stepsize * cormom1 / (torch.sqrt(cormom2) + eps)
        newest = estimate - update
        estimate = newest
        clampedest = torch.clamp(newest, min=0, max=1)
        hereloss = MSEloss(clampedest, clampedGT)
        if hereloss < bestloss:
            bestloss = hereloss
            bestest = estimate
        PSNRvals[ctr] = getavgpsnr(clampedest, clampedGT)
        #lpipsvals[ctr] = lpipsloss(2*estimate-1, 2*GT-1).mean()
        lpipsvals[ctr] = lpipsloss(clampedest, clampedGT)
        ssimvals[ctr] = ssimloss(clampedest, clampedGT)
        stepmarg[ctr-1] = torch.mean(update ** 2)
        if ctr % log_freq == 0:
            save_image(estimate , log_path + str(ctr) + 'MAPit.png')
        usedsig = max(usedsig * fac, AWGNsig)
    return bestest, PSNRvals, lpipsvals, ssimvals, stepmarg, estimate




def DiffPIR(test_data, log_path, dennet, device, MSEloss , ssimloss , lpipsloss , noise_sigma, maxsig = 50, 
            deg = torch.nn.Identity(), adj = torch.nn.Identity(), getx0 = torch.nn.Identity(), maxit = 50, 
               log_freq = 1e7, dsfac = 0, obsx0 = False, lamb = 1, mixparam = 0.5, onestepprox = False, full = False,
               save_fullbatch = False):
    t = time.time()
    PSNRarray = torch.zeros(maxit, device = device)
    ssimarray = torch.zeros(maxit, device = device)
    lpipsarray = torch.zeros(maxit, device = device)
    if onestepprox:
        print('Because of VE setup, truncate the 1st order approximation of proximal operator')
    minsig = 0.01
    calcfac = (minsig / maxsig) ** (1 / (maxit - 1))
    bctr = 0
    if 0 < dsfac <= 1:
        fac = dsfac
        if dsfac > calcfac:
            print('Warning, manual set factor is too large to go to minimal sigma')
        # print(f'using preset noise-level factor {fac}')
    else:
        fac = calcfac 
        # print(f'using auto noise-level factor {fac}')
    for batch, _ in test_data:
        bctr +=1
        batch = batch.to(device=device)
        clampedbatch = torch.clamp(batch, min=0, max=1)
        degraded = deg(batch)
        AWGN = noise_sigma * torch.randn_like(degraded)
        degraded = degraded + AWGN
        adjdeg = adj(degraded)
        noisy_startimg = 0.5 + maxsig * torch.randn_like(batch)
        if obsx0:
            x0 = getx0(degraded) + maxsig * torch.randn_like(batch)
        else:
            x0 = noisy_startimg
        genimg = x0
        prev_sig = maxsig 
        if save_fullbatch:
            for sctr in range(batch.size(0)):
                save_image(degraded[sctr], log_path + str(sctr) +  'degd.png')
                save_image(x0[sctr], log_path + str(sctr)+ 'x0.png')
                save_image(getx0(degraded)[sctr], log_path + str(sctr)+ 'resizeddeg.png')
                save_image(batch[sctr], log_path + str(sctr)+ 'GT.png')
        save_image(degraded, log_path + 'degd.png')
        save_image(x0, log_path + 'x0.png')
        save_image(getx0(degraded), log_path + 'resizeddeg.png')
        save_image(batch, log_path + 'GT.png')
        for noisectr in range(1,maxit):
            current_sig = max(prev_sig * fac, noise_sigma)
            score = dennet(genimg, prev_sig*torch.ones(genimg.shape[0], device = device))
            x0est = genimg + prev_sig ** 2 * score
            param = lamb * (noise_sigma  / prev_sig) ** 2
            if onestepprox:
                effmult = min(1, 1 / param)
                afterdata = x0est -  effmult * (adj(deg(x0est)- degraded))
            else:
                afterdata = prox(degraded = degraded, input = x0est, deg = deg, adj = adj, param = param)

            eff_eps =  (genimg - afterdata) / prev_sig
            eps = torch.randn_like(genimg)
            genimg = afterdata + current_sig * ((1 - mixparam) ** 0.5 * eff_eps + mixparam ** 0.5 * eps)
            clampedgen = torch.clamp(genimg, min=0, max=1).to(device=device)
            PSNRarray[noisectr-1] += getavgpsnr(clampedgen, clampedbatch) #10 * torch.log10(1 / MSEloss(clampedgen, clampedbatch))
            ssimarray[noisectr-1] += ssimloss(clampedgen, clampedbatch)
            lpipsarray[noisectr-1] += lpipsloss(clampedgen, clampedbatch)

            if noisectr % log_freq == 0:
                save_image(genimg , log_path + str(noisectr) + '.png')
            prev_sig = current_sig

        score = dennet(genimg, prev_sig*torch.ones(genimg.shape[0], device = device))
        x0est = genimg + prev_sig ** 2 * score
        param = lamb * (noise_sigma  / prev_sig) ** 2
        if onestepprox:
            effmult = min(1, 1 / param)
            afterdata = x0est -  effmult * (adj(deg(x0est)- degraded))
        else:
            afterdata = prox(degraded = degraded, input = x0est, deg = deg, adj = adj, param = param)
        clampaft = torch.clamp(afterdata, min = 0, max = 1).to(device=device)
        PSNRarray[maxit-1] +=  getavgpsnr(clampaft, clampedbatch)
        ssimarray[maxit-1] += ssimloss(clampaft, clampedbatch)
        lpipsarray[maxit-1] += lpipsloss(clampaft, clampedbatch)
        if not full:
            break
    PSNRarray = PSNRarray/bctr
    ssimarray = ssimarray/bctr
    lpipsarray = lpipsarray/bctr
    save_image(afterdata, log_path + 'sig0.png')
    save_image(genimg, log_path + 'sig1.png')

    if save_fullbatch:
        for sctr in range(batch.size(0)):
            save_image(afterdata[sctr], log_path + 'sig0.png')
            save_image(genimg[sctr], log_path + 'sig1.png')
    elapsed = time.time() - t
    if elapsed > 3600:
        timeprint = f'{round(elapsed/3600 , 2)}h'
    elif elapsed > 60:
        timeprint = f'{round(elapsed/60 , 2)}m'
    else:
        timeprint = f'{elapsed}s'
    print(f"after {timeprint}: PSNR $|$ SSIM $|$ LPIPS: {round(PSNRarray[-1].item(),3)} $|$ {round(ssimarray[-1].item(),4)} $|$ {round(lpipsarray[-1].item(), 4)}")
    plt.plot(PSNRarray.cpu().numpy(force = True), 'b', label = 'MAP_reconstruction')
    plt.xlim(0,maxit -1)
    plt.ylim(5, torch.max(PSNRarray.cpu()) + 0.5)
    plt.xlabel('iterations')
    plt.ylabel('PSNR')
    #plt.plot(avgloss.numpy(force = True))
    plt.legend()
    plt.savefig(log_path +  '_sig' + str(noise_sigma) + 'avgpsnr.png')
    plt.clf()
    plt.plot(ssimarray.cpu().numpy(force = True), 'b', label = 'MAP_reconstruction')
    plt.xlim(0,maxit -1)
    plt.xlabel('iterations')
    plt.ylabel('SSIM')
    #plt.plot(avgloss.numpy(force = True))
    plt.legend()
    plt.savefig(log_path +  '_sig' + str(noise_sigma) + 'ssim.png')
    plt.clf()
    plt.plot(lpipsarray.cpu().numpy(force = True), 'b', label = 'MAP_reconstruction')
    plt.xlim(0,maxit -1)
    plt.xlabel('iterations')
    plt.ylabel('LPIPS')
    #plt.plot(avgloss.numpy(force = True))
    plt.legend()
    plt.savefig(log_path +  '_sig' + str(noise_sigma) + 'lpips.png')
    plt.clf()
    save_image(genimg , log_path + 'MAPdenoised.png')


def MAPGD_tst(test_data, log_path, dennet, device,  MSEloss , ssimloss , lpipsloss, noise_sigma, maxsig = 50, downscale = False, fac = 1, 
            deg = torch.nn.Identity(), adj = torch.nn.Identity(), getx0 = torch.nn.Identity(),  maxit = 50, stepsize = 0.1, adam = False, inittype = 'randstart', 
               addnoise = False, genfromnoise = False, log_freq = 1e7, adamb2 = 0.999, lamb = 1, full = False):
    t=time.time()
    optdennet = dennet
    PSNRarray = torch.zeros(maxit+1, device = device)
    ssimarray = torch.zeros(maxit+1, device = device)
    lpipsarray = torch.zeros(maxit+1, device = device)
    ctr = 0
    loss = torch.nn.MSELoss()
    single_step_PSNR = 0
    for batch, _ in test_data:
        batch = batch.to(device=device)
        degraded = deg(batch)
        AWGN = noise_sigma * torch.randn_like(degraded)
        degraded = degraded + AWGN
        adjdeg = adj(degraded)
        #noisy = torch.randn_like(batch)
        x0 = getx0(degraded)
        with torch.no_grad():
            if adam:
                [bestest, PSNRvals, lpipsvals, ssimvals, stepmarg, denoised] = adamMAPGD(NN = optdennet, degraded = degraded, GT = batch, AWGNsig =noise_sigma, maxit = maxit, 
                        stepsize = stepsize, MSEloss = MSEloss, ssimloss = ssimloss, lpipsloss = lpipsloss, device = device, maxsig = maxsig, downscale = downscale, fac = fac, deg = deg, adj = adj, 
                        inittype = inittype, beta2 = adamb2, x0 = x0,  addnoise = addnoise, 
                        log_path = log_path, log_freq = log_freq, lamb = lamb)
            else:
                [bestest, PSNRvals, lpipsvals, ssimvals, stepmarg, denoised] = MAPGD(NN = optdennet, degraded = degraded, GT = batch, AWGNsig =noise_sigma, maxit = maxit, 
                        stepsize = stepsize, MSEloss = MSEloss, ssimloss = ssimloss, lpipsloss = lpipsloss,  device = device, maxsig = maxsig, downscale = downscale, fac = fac, deg = deg, adj = adj, inittype = inittype, x0 = x0,  addnoise = addnoise, 
                        log_path = log_path, log_freq = log_freq, lamb = lamb)
            PSNRarray += PSNRvals
            ssimarray += ssimvals
            lpipsarray += lpipsvals
            lpipsarray += lpipsvals
            ctr += 1
        if not full:
            break

    avgpsnr = PSNRarray / ctr
    avgssim = ssimarray / ctr
    avglpips = lpipsarray / ctr
    elapsed = time.time() - t
    if elapsed > 3600:
        timeprint = f'{round(elapsed/3600 , 2)}h'
    elif elapsed > 60:
        timeprint = f'{round(elapsed/60 , 2)}m'
    else:
        timeprint = f'{elapsed}s'
    print('######################################################')
    print(f"after {timeprint} PSNR $|$ SSIM $|$ LPIPS: {round(avgpsnr[-1].item(),2)} $|$  {round(avgssim[-1].item(),3)} $|$ {round(avglpips[-1].item(), 3)}")
    print('######################################################')
    plt.plot(avgpsnr.cpu().numpy(force = True), 'b', label = 'MAP_reconstruction')
    plt.xlim(0,maxit)
    plt.xlabel('iterations')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig(log_path + str(stepsize) +  '_sig' + str(noise_sigma) + 'avgpsnr.png')
    plt.clf()

    plt.plot(avgssim.cpu().numpy(force = True), 'b', label = 'MAP_reconstruction')
    plt.xlim(0,maxit)
    plt.xlabel('iterations')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig(log_path + str(stepsize) +  '_sig' + str(noise_sigma) + 'SSIM.png')
    plt.clf()

    plt.plot(avglpips.cpu().numpy(force = True), 'b', label = 'MAP_reconstruction')
    plt.xlim(0,maxit)
    plt.xlabel('iterations')
    plt.ylabel('LPIPS')
    plt.legend()
    plt.savefig(log_path + str(stepsize) +  '_sig' + str(noise_sigma) + 'LPIPS.png')
    plt.clf()

    plt.plot(stepmarg)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('MSE between iterates')
    plt.savefig(log_path + str(stepsize) +  '_sig' + str(noise_sigma) + 'margins.png')
    save_image(denoised , log_path + 'MAPdenoised.png')
    save_image(degraded, log_path + 'degd.png')
    save_image(getx0(degraded), log_path + 're(obs).png')
    save_image(bestest, log_path + 'bestmap.png')
    save_image(batch, log_path + 'GT.png')