import torch
import random
import time
from utils.net_wrap import divbynoise
import deepinv.physics as ph


class set_of_degradations():
    def __init__(self,device, ipset = ['denoising', 'pixel_inpaint', 'bw_inpaint', 'blur', 'super_resolution'], noise_sigmas = [0.01, 50],
     pixel_chances = [0.0, 1.0], blocksizes = [0, 32], blurfacs = [0.0, 8.0], srfacs = [1,8], imsizes = [32, 32]):
        self.ipset = ipset
        self.noise_sigmas = noise_sigmas
        self.pixel_chances = pixel_chances
        self.blocksizes = blocksizes
        self.blurfacs = blurfacs
        self.srfacs = srfacs
        self.imsizes = imsizes
        self.device = device
        #self.batchsize = batchsize

    def picksigma(self):
        noise_randvar = random.uniform(0,1)
        sigma = (self.noise_sigmas[1] - self.noise_sigmas[0] + 1) ** noise_randvar - 1 + self.noise_sigmas[0]
        return sigma

    def getnoiseIP(self):
        deg = ph.Inpainting(tensor_size = (3,self.imsizes[0], self.imsizes[1]), mask = 1, device = self.device)
        adj = deg
        return deg, adj, 1.0

    def getpixelcompIP(self, fac):
        fac = 1 - fac
        deg = ph.Inpainting(tensor_size = (3,self.imsizes[0], self.imsizes[1]), mask = fac, device = self.device)
        return deg, deg, fac

    def getblockcompIP(self,ipfac):
        if isinstance(ipfac, list):
            boxsizes = ipfac
        else:
            ipfac = int(ipfac)
            boxsizes = [ipfac, ipfac]
        boxpos1 = random.choice(range(self.imsizes[0] - boxsizes[0]+1))
        boxpos2 = random.choice(range(self.imsizes[1] - boxsizes[1]+1))
        mask = torch.ones((3,self.imsizes[0], self.imsizes[1])).to(device=self.device)
        mask[:,boxpos1:boxpos1+boxsizes[0],boxpos2:boxpos2+boxsizes[1]] = torch.zeros((3,boxsizes[0], boxsizes[1])).to(device=self.device)
        deg = ph.Inpainting(tensor_size = (3,self.imsizes[0], self.imsizes[1]), mask = mask, device = self.device)
        return deg, deg, (boxsizes[0] * boxsizes[1] / (self.imsizes[0] * self.imsizes[1]))

    def getblurIP(self, blurfac):
        filter = ph.blur.gaussian_blur(sigma=blurfac)
        # deg = ph.Blur(filter=filter, device = self.device)
        # adj = ph.forward.adjoint_function(deg, (self.batchsize, 3, self.imsizes[0], self.imsizes[1]), self.device)
        deg = ph.Blur(filter=filter, device = self.device)
        adj = deg.A_adjoint
        return deg, adj, (1 - 1 / (1 + blurfac))
    
    def getSRIP(self, SRfac):
        if SRfac < 1:
            raise Exception('downsampling factor is smaller than 1')
        deg = ph.Downsampling(img_size = ( 3, self.imsizes[0], self.imsizes[1]), factor = int(SRfac), filter = 'bicubic', device = self.device)
        adj = deg.A_adjoint
        #adj = ph.forward.adjoint_function(deg, (self.batchsize, 3, self.imsizes[0], self.imsizes[1]), self.device)
        return deg, adj, (1 - 1 / SRfac ** 2)

    def getranddeg(self):
        inverse_problem = random.choice(self.ipset)
        if inverse_problem == 'denoising':
            return self.getnoiseIP(), inverse_problem
        elif inverse_problem == 'pixel_inpaint':
            fac = random.uniform(self.pixel_chances[0], self.pixel_chances[1])
            return self.getpixelcompIP(fac), inverse_problem
        elif inverse_problem == 'bw_inpaint':
            blocksizes = [random.randint(self.blocksizes[0], self.blocksizes[1]), random.randint(self.blocksizes[0], self.blocksizes[1])]
            return self.getblockcompIP(blocksizes), inverse_problem
        elif inverse_problem == 'blur':
            blurfac = random.uniform(self.blurfacs[0], self.blurfacs[1])
            return self.getblurIP(blurfac), inverse_problem
        elif inverse_problem == 'super_resolution':
            SRfac = random.randint(self.srfacs[0],self.srfacs[1])
            return self.getSRIP(SRfac), inverse_problem
        else:
            raise Exception(f'inverse problem {inverse_problem} not implemented')
        
    def getIPandsig(self):
        deg, adj, fac, degtype = self.getranddeg()
        sigma = self.picksigma()
        return deg, adj,fac, sigma, degtype
    

def wrappedMAPdenoise(NN,degraded,heresig, maxit, convmarg, stepsize, 
                      deg = torch.nn.Identity(), adj = torch.nn.Identity(),  stopmarg = 100, 
                inittype = 'randstart', x0 = None, scorefac = -1):
    ctr = 0
    if inittype == 'randstart':
        estimate = 0.5 + torch.randn_like(adj(degraded))
    elif inittype == 'bigstepfirst':
        estimate = adj(degraded)
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * divbynoise(NN, estimate, heresig)
    elif inittype == 'rand_thenbigstep':
        estimate = torch.randn_like(adj(degraded))
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * divbynoise(NN, estimate, heresig)
    elif inittype == 'standard' or inittype == 'virtobs':
        estimate = adj(degraded)
    elif inittype == 'usex0' and x0 is not None:
        estimate = x0
    else:
        raise NotImplementedError('bad inittype')


    for ctr in range(1,maxit+1):
        datagrad = adj(deg(estimate) - degraded)
        reggrad = scorefac * heresig ** 2 * divbynoise(NN,estimate, heresig)
        update = stepsize * (datagrad + reggrad)
        estimate = estimate - update
        if torch.mean(update ** 2) < convmarg:
            return estimate, ctr, 1
        if torch.mean(estimate ** 2) > stopmarg:
            return estimate, maxit + 0.0001,  0
    return estimate, maxit, 0

def wrappedadamMAPdenoise(NN, degraded,heresig, maxit, convmarg, stepsize,
                          deg = torch.nn.Identity(), adj = torch.nn.Identity(), stopmarg = 100,  
inittype = 'randstart', beta1 = 0.9, beta2 = 0.999, x0 = None, scorefac = -1):
    ctr = 0
    if inittype == 'randstart':
        estimate = 0.5 +  torch.randn_like(adj(degraded))
    elif inittype == 'bigstepfirst':
        estimate = adj(degraded)
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * divbynoise(NN, estimate, heresig)
    elif inittype == 'rand_thenbigstep':
        estimate = torch.randn_like(adj(degraded))
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * divbynoise(NN, estimate, heresig)
    elif inittype == 'standard' or inittype == 'virtobs':
        estimate = adj(degraded)
    elif inittype == 'usex0' and x0 is not None:
        estimate = x0
    else:
        raise NotImplementedError('bad inittype')

    eps = 1e-8
    mom1 = torch.zeros_like(estimate)
    mom2 = torch.zeros_like(estimate)
    for ctr in range(1,maxit + 1):
        g = adj(deg(estimate) - degraded) + scorefac * heresig ** 2 * divbynoise(NN,estimate, heresig)
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




def tembMAPGD(NN,degraded,heresig, maxit, convmarg, stepsize, device, deg = torch.nn.Identity(), adj = torch.nn.Identity(),  stopmarg = 100, 
                inittype = 'randstart', x0 = None, scorefac = -1):
    ctr = 0
    if inittype == 'randstart':
        estimate = 0.5 +  torch.randn_like(adj(degraded))
    elif inittype == 'bigstepfirst':
        estimate = adj(degraded)
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * NN(estimate, heresig*torch.ones(estimate.shape[0], device = device))
    elif inittype == 'rand_thenbigstep':
        estimate = torch.randn_like(adj(degraded))
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
    elif inittype == 'standard' or inittype == 'virtobs':
        estimate = adj(degraded)
    elif inittype == 'usex0' and x0 is not None:
        estimate = x0
    else:
        raise('bad inittype')


    for ctr in range(1,maxit+1):
        datagrad = adj(deg(estimate) - degraded)
        reggrad = scorefac * heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
        update = stepsize * (datagrad + reggrad)
        estimate = estimate - update
        if torch.mean(update ** 2) < convmarg:
            return estimate, ctr, 1
        if torch.mean(estimate ** 2) > stopmarg:
            return estimate, maxit + 0.0001,  0
    return estimate, maxit, 0

def tembadamMAPGD(NN, degraded,heresig, maxit, convmarg, stepsize, device,
                          deg = torch.nn.Identity(), adj = torch.nn.Identity(), stopmarg = 100,  
inittype = 'randstart', beta1 = 0.9, beta2 = 0.999, x0 = None, scorefac = -1):
    ctr = 0
    if inittype == 'randstart':
        estimate = 0.5 +  torch.randn_like(adj(degraded))
    elif inittype == 'bigstepfirst':
        estimate = adj(degraded)
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
    elif inittype == 'rand_thenbigstep':
        estimate = torch.randn_like(adj(degraded))
        datagrad = adj(deg(estimate) - degraded)
        estimate = estimate - datagrad - scorefac * heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
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
        g = adj(deg(estimate) - degraded) + scorefac * heresig ** 2 * NN(estimate, heresig * torch.ones(estimate.shape[0], device = device))
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

def GDIP(training_data, dennet, ema, logger, optimizer, device, save_path, loss, trainepochs,
             start_epoch, degset,  maxit = 50, stepsize = 0.1, adam = False, convmarg = 1e-5, stopmarg = 500,
             bpstepsize = 0.01, inittype = 'bigstepfirst', useMMSE = False, stopnumb = 1e9, beta1 = 0.9, rescaledloss = True,
             MMSEfac = 1, scaleconvmarg = False, varstepsize = False):
    for epoch in range(start_epoch, trainepochs):
        t=time.time()
        convcount = 0
        fullitcount = 0
        runningMAPloss = 0
        runningMMSEloss = 0
        for itctr, [batch, labels] in enumerate(training_data):
            optimizer.zero_grad()
            batch = batch.to(device=device)
            deg, adj, margfac, heresig = degset.getIPandsig()
            puredegimg = deg(batch)
            AWGN =  heresig * torch.randn_like(batch)
            noisy = batch + AWGN
            degraded = puredegimg + heresig * torch.randn_like(puredegimg)

            ### Compute different denoising estimates ###
            if scaleconvmarg:
                usemarg = convmarg * heresig #convmarg * ( margfac + heresig / degset.noise_sigmas[1] ) / 2
            else:
                usemarg = convmarg
            with torch.no_grad():
                if adam:
                    [estimate, itcount, is_conv] = wrappedadamMAPdenoise(NN = dennet, degraded = degraded,
                                                                         heresig = heresig, maxit = maxit, convmarg = usemarg, 
                                                                         stepsize = stepsize , deg=deg, adj = adj,stopmarg = stopmarg,  inittype = inittype, 
                                                                         scorefac = 1, beta1 = beta1)
                else:
                    if varstepsize:
                        effstep = stepsize * (heresig + margfac) ** 2 / heresig ** 2
                        herestep = min(effstep * (), 0.24)
                    else:
                        herestep = stepsize
                        
                    [estimate, itcount, is_conv] = wrappedMAPdenoise(NN = dennet, degraded = degraded,
                                                                         heresig = heresig, maxit = maxit, convmarg = usemarg, 
                                                                         stepsize = herestep , deg = deg, adj = adj, stopmarg = stopmarg,  inittype = inittype, 
                                                                         scorefac = 1)
                fullitcount += itcount
                convcount += is_conv

            ### Compute Losses for Backpropagation ###
            trainloss = 0
            estimate.requires_grad = True
            datagrad = adj(deg(estimate) - degraded)
            reggrad =  divbynoise(dennet,estimate, heresig)
            if rescaledloss:
                hereloss = loss(heresig * reggrad, ((estimate - batch) / bpstepsize - datagrad  ) / heresig) 
            else:
                hereloss = loss((estimate - bpstepsize* (datagrad + heresig ** 2 * reggrad)/ heresig), batch / heresig)
            #hereloss = loss(batch, bp1step)
            trainloss = hereloss
            if useMMSE:
                ## compute DSM loss for backpropagation ###
                noisy.requries_grad = True
                MMSEloss = loss(heresig * divbynoise(dennet,noisy,heresig), AWGN / heresig)
                runningMMSEloss += MMSEloss.item()
                trainloss += MMSEfac * MMSEloss
            runningMAPloss+= hereloss.item()
            
            trainloss.backward()
            optimizer.step()
            ema.update(dennet.parameters())
            if itctr > 0 and (itctr + 1) % stopnumb == 0:
                print(f'epoch ended after {itctr + 1} iterations')
                logger.info(f'epoch ended after {itctr + 1} iterations')
                break

        print(f'epoch {epoch} counvcount {convcount} it_used {fullitcount} time {round(time.time()-t)}')
        print(f'            MAPloss: {runningMAPloss}, MMSEloss: {runningMMSEloss}')
        logger.info(f'epoch {epoch} counvcount {convcount} it_used {fullitcount} time {round(time.time()-t)}')
        logger.info(f'            MAPloss: {runningMAPloss}, MMSEloss: {runningMMSEloss}')

        torch.save({'model': dennet.state_dict(),
                    'ema': ema.state_dict(),
                    'epoch': epoch + 1,
                    'optimizer_state': optimizer.state_dict()}, save_path)

def temb_GDIP(training_data, dennet, ema, logger, optimizer, device, save_path, loss, DEQlossfunc, trainepochs,
             start_epoch, degset,  maxit = 50, stepsize = 0.1, adam = False, convmarg = 1e-5, stopmarg = 500,
             bpstepsize = 0.01, inittype = 'bigstepfirst', useMMSE = False, stopnumb = 1e9, beta1 = 0.9, rescaledloss = True,
             MMSEfac = 1, scaleconvmarg = False, varstepsize = False, usedirectloss = False):
    for epoch in range(start_epoch, trainepochs):
        t=time.time()
        convcount = 0
        fullitcount = 0
        runningMAPloss = 0
        runningMMSEloss = 0
        for itctr, [batch, labels] in enumerate(training_data):
            optimizer.zero_grad()
            batch = batch.to(device=device)
            deg, adj, margfac, heresig, degtype = degset.getIPandsig()
            puredegimg = deg(batch)
            AWGN =  heresig * torch.randn_like(batch)
            noisy = batch + AWGN
            degraded = puredegimg + heresig * torch.randn_like(puredegimg)

            ### Compute different denoising estimates ###
            if scaleconvmarg:
                usemarg = convmarg * heresig #convmarg * ( margfac + heresig / degset.noise_sigmas[1] ) / 2
            else:
                usemarg = convmarg
            with torch.no_grad():
                if adam:
                    [estimate, itcount, is_conv] = tembadamMAPGD(NN = dennet, degraded = degraded,
                                                                         heresig = heresig, maxit = maxit, convmarg = usemarg, 
                                                                         stepsize = stepsize , deg=deg, adj = adj,stopmarg = stopmarg,  inittype = inittype, 
                                                                         scorefac = -1, beta1 = beta1, device = device)
                else:
                    if varstepsize:
                        effstep = stepsize * (heresig + margfac) ** 2 / heresig ** 2
                        herestep = min(effstep * (), 0.24)
                    else:
                        herestep = stepsize
                        
                    [estimate, itcount, is_conv] = tembMAPGD(NN = dennet, degraded = degraded,
                                                                         heresig = heresig, maxit = maxit, convmarg = usemarg, 
                                                                         stepsize = herestep , deg = deg, adj = adj, stopmarg = stopmarg,  inittype = inittype, 
                                                                         scorefac = -1, device = device)
                fullitcount += itcount
                convcount += is_conv

            ### Compute Losses for Backpropagation ###
            trainloss = 0
            estimate.requires_grad = True
            datagrad = adj(deg(estimate) - degraded)
            reggrad =  dennet(estimate, heresig * torch.ones(estimate.shape[0], device = device))
            if rescaledloss and not usedirectloss:
                hereloss = DEQlossfunc(- heresig * reggrad, ((estimate - batch) / bpstepsize - datagrad  ) / heresig) 
            elif usedirectloss:
                hereloss = DEQlossfunc(estimate - bpstepsize* (datagrad - heresig ** 2 * reggrad), batch )
            else:
                hereloss = DEQlossfunc(((estimate - bpstepsize* (datagrad - heresig ** 2 * reggrad))/ heresig), batch / heresig)
            #hereloss = loss(batch, bp1step)
            trainloss = hereloss
            if useMMSE:
                ## compute DSM loss for backpropagation ###
                noisy.requries_grad = True
                MMSEloss = loss(- heresig * dennet(noisy, heresig * torch.ones(noisy.shape[0], device = device)), AWGN / heresig)
                runningMMSEloss += MMSEloss.item()
                trainloss += MMSEfac * MMSEloss
            runningMAPloss+= hereloss.item()
            
            trainloss.backward()
            optimizer.step()
            ema.update(dennet.parameters())
            if itctr > 0 and (itctr + 1) % stopnumb == 0:
                print(f'epoch ended after {itctr + 1} iterations')
                logger.info(f'epoch ended after {itctr + 1} iterations')
                break

        print(f'epoch {epoch} counvcount {convcount} it_used {fullitcount} time {round(time.time()-t)}')
        print(f'            MAPloss: {runningMAPloss}, MMSEloss: {runningMMSEloss}')
        logger.info(f'epoch {epoch} counvcount {convcount} it_used {fullitcount} time {round(time.time()-t)}')
        logger.info(f'            MAPloss: {runningMAPloss}, MMSEloss: {runningMMSEloss}')

        torch.save({'model': dennet.state_dict(),
                    'ema': ema.state_dict(),
                    'epoch': epoch + 1,
                    'optimizer_state': optimizer.state_dict()}, save_path)
