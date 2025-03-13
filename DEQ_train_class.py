import torch
import time
from DEQ_iterators import MAPGD, adamMAPGD
import random 

class MAP_trainer:
    def __init__(self, dennet, ema, logger, optimizer, scheduler, device, save_path, DSMloss, DEQlossfunc, trainepochs,
             start_epoch, degset,  maxit , stepsize , adambool , convmarg, stopmarg, bpstepsize, inittype, DSMfac, 
             lossscaling, scaleconvmarg):
        self.dennet = dennet
        self.ema = ema
        self.logger = logger
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.DSMloss = DSMloss
        self.DEQlossfunc = DEQlossfunc
        self.trainepochs = trainepochs
        self.start_epoch = start_epoch
        self.degset = degset
        self.maxit = maxit
        self.stepsize = stepsize
        self.adambool = adambool
        self.convmarg = convmarg
        self.stopmarg = stopmarg
        self.bpstepsize = bpstepsize
        self.inittype = inittype
        self.DSMfac = DSMfac
        self.lossscaling = lossscaling
        self.scaleconvmarg = scaleconvmarg
        self.scheduler = scheduler
        if torch.cuda.device_count() > 1:
            print(f"Number of GPUs used: {torch.cuda.device_count()}")
            self.logger.info(f"Number of GPUs used: {torch.cuda.device_count()}")
            self.optnet = torch.nn.DataParallel(dennet).cuda()
        else:
            self.logger.info(f"Number of GPUs used: {torch.cuda.device_count()}")
            print(f"Number of GPUs used: {torch.cuda.device_count()}")
            self.optnet = dennet

    def single_training_iteration(self, batch):
        self.optimizer.zero_grad()
        batch = batch.to(device=self.device)
        deg, adj, getx0, heresig, degtype = self.degset.getIPandsig()
        puredegimg = deg(batch)
        AWGN =  heresig * torch.randn_like(batch)
        noisy = batch + AWGN
        degraded = puredegimg + heresig * torch.randn_like(puredegimg)

        if self.scaleconvmarg:
            hereconvmarg = self.convmarg * heresig
        else:
            hereconvmarg = self.convmarg

        x0 = getx0(degraded)
        ### This estimates the equilibrium point ###
        with torch.no_grad():
            if self.adambool:
                [estimate, itcount, is_conv] = adamMAPGD(NN = self.optnet, degraded = degraded,
                                                            heresig = heresig, maxit = self.maxit, convmarg = hereconvmarg, 
                                                            stepsize = self.stepsize , deg=deg, adj = adj,stopmarg = self.stopmarg,  inittype = self.inittype, 
                                                            device = self.device, x0 = x0)
            else:
                [estimate, itcount, is_conv] = MAPGD(NN = self.optnet, degraded = degraded,
                                                        heresig = heresig, maxit =self.maxit, convmarg = hereconvmarg, 
                                                        stepsize = self.stepsize , deg = deg, adj = adj, stopmarg = self.stopmarg,  inittype = self.inittype, 
                                                            device = self.device, x0 = x0)

        ### Compute Losses for Backpropagation using the equilibrium point ###
        estimate.requires_grad = True
        datagrad = adj(deg(estimate) - degraded)
        reggrad =  self.optnet(estimate, heresig * torch.ones(estimate.shape[0], device = self.device))
        bp_1stepest = estimate - self.bpstepsize* (datagrad - heresig ** 2 * reggrad)
        if self.lossscaling == 'divbynoise_and_stepsize' or self.lossscaling == 'both':
            DEQloss = self.DEQlossfunc(bp_1stepest / (heresig * self.bpstepsize),
                                         batch / (heresig * self.bpstepsize))
            #DEQloss = self.DEQlossfunc(- heresig * reggrad, ((estimate - batch) / self.bpstepsize - datagrad  ) / heresig) 
        elif self.lossscaling == 'usedirectloss' or self.lossscaling == 'none':
            DEQloss = self.DEQlossfunc(bp_1stepest, batch)
        elif self.lossscaling == 'specloss':
            DEQloss = self.DEQlossfunc(bp_1stepest, batch, heresig)
        elif self.lossscaling == 'divbynoise':
            DEQloss = self.DEQlossfunc(bp_1stepest / heresig, batch / heresig)
        else:
            raise NotImplementedError(f"lossscaling {self.lossscaling} is not implemented")
        trainloss = DEQloss


        if self.DSMfac > 0:
            ## compute DSM loss for backpropagation ###
            noisy.requries_grad = True
            DSMloss = self.DSMloss(- heresig * self.optnet(noisy, heresig * torch.ones(noisy.shape[0], device = self.device)), AWGN / heresig)
            DSMlosshere = DSMloss.item()
            trainloss += self.DSMfac * DSMloss
        else:
            DSMlosshere = 0
        
        trainloss.backward()
        self.optimizer.step()
        self.ema.update(self.dennet.parameters())
        return(DEQloss.item(), DSMlosshere, itcount, is_conv)
    
    def train(self, training_data, stopnumb):
        for epoch in range(self.start_epoch, self.trainepochs):
            t=time.time()
            convcount = 0
            fullitcount = 0
            runningDEQloss = 0
            runningDSMloss = 0
            for itctr, [batch, _] in enumerate(training_data):
                ### Training step of NN ###
                iterationDEQloss, iterationDSMloss, itcount, is_conv = self.single_training_iteration(batch)
                runningDEQloss += iterationDEQloss
                runningDSMloss += iterationDSMloss
                fullitcount += itcount
                convcount += is_conv
                if itctr > 0 and (itctr + 1) % stopnumb == 0:
                    print(f'epoch ended after {itctr + 1} iterations')
                    self.logger.info(f'epoch ended after {itctr + 1} iterations')
                    break

            self.scheduler.step()
            print(f'epoch {epoch} counvcount {convcount} it_used {fullitcount} time {round(time.time()-t)}')
            print(f'            DEQloss: {runningDEQloss}, DSMloss: {runningDSMloss}')
            self.logger.info(f'epoch {epoch} counvcount {convcount} it_used {fullitcount} time {round(time.time()-t)}')
            self.logger.info(f'            DEQloss: {runningDEQloss}, DSMloss: {runningDSMloss}')

            torch.save({'model': self.dennet.state_dict(),
                        'ema': self.ema.state_dict(),
                        'epoch': epoch + 1,
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.scheduler.state_dict()}, self.save_path)
            
        torch.save({'model': self.dennet.state_dict(),
            'ema': self.ema.state_dict(),
            'epoch': epoch + 1,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()}, self.save_path + '_final')
        
        print(f'training finished after {epoch + 1} epochs')
        self.logger.info(f'training finished after {epoch + 1} epochs')

