import torch
from ncsnpp.models.ncsnpp import NCSNpp
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage as propema
import os
from setseed import set_seed
import argparse
from ncsnpp.configs.ve.bedroom_ncsnpp_continuous import get_config as bedroomconfig
from ncsnpp.configs.ve.cifar10_ncsnpp_continuous import get_config as cifarconfig
import time

""""
Re-Implementation of VEPC sampler from:
 Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." arXiv preprint arXiv:2011.13456 (2020)."""
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds','--dataset', type=str, default= 'cifar10', help='choose dataset from: bedroom, cifar10')
    parser.add_argument('--itnum', type = int, default = 1, help='number of iterations')
    parser.add_argument('--numofimg', type = int, default = 1, help='how many img to generate')
    parser.add_argument('--imsize', type = int, default = 32, help='imgsize for square images')
    parser.add_argument('--maximgs', type = int, default = 1, help='how many images to save if saveall is used')
    parser.add_argument('-bs','--batchsize', type = int, default = 32, help='batchsize')
    parser.add_argument('-r','--snr', default = None, help='set snr (optional)')
    parser.add_argument('--condx0',  default= False, action='store_true', help='use condx0 score')
    parser.add_argument('--saveall',  default= False, action='store_true', help='save all images for comparison')
    parser.add_argument('--skip_pred',  default= False, action='store_true', help='skip predictor for SDE')
    parser.add_argument('--noema',  default= False, action='store_true', help='do not use ema params: False uses EMA params')
    parser.add_argument('--noaddnoise',  default= False, action='store_true', help='use condx0 score')
    parser.add_argument('-np','--netpath',  type = str, required = True , help='path to nn')
    parser.add_argument('--basepath', type=str, required = True, help='location for log_files')
    args = parser.parse_args()

    print(args)
    set_seed(seed = 42, inference = True)
    dataset = args.dataset
    condx0 = args.condx0
    noaddnoise = args.noaddnoise
    bs = args.batchsize
    
    net_path = args.netpath
    if dataset == 'CIFAR10' or dataset == 'cifar10':
        imsize = 32
        noisenum = 1000 #232
        minsig = 0.01
        maxsig =  50
        numofimg = 1
        printfreq = 1e9
        if type(args.snr) is str:
            print('modified snr to ', args.snr)
            r = args.snr
        else:
            r = 0.16 #SNR for corrector

        maximgs = args.maximgs
        config = cifarconfig()
    elif dataset == 'bedroom':
        maximgs = args.maximgs
        imsize = 256
        noisenum = 2000 #232
        minsig = 0.01
        maxsig =  378
        numofimg = 1
        printfreq = 1e9
        r = 0.075 #SNR for corrector
        config = bedroomconfig()
    else:
        raise Exception('bad dataset')


    if args.saveall:
        numofimg = 100
    else:
        numofimg = args.numofimg
    net_path = net_path

    save_path = args.basepath 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    

    model = NCSNpp(config)
    model = model.to(device=device)

    #model = nn.DataParallel(model)
    print(net_path)

    dict = torch.load(net_path, map_location = device)
    model.load_state_dict(dict['model'], strict=False) #LearnedOpt_state  model
    if not args.noema:
        # ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        ema = propema(model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(dict['ema'])
        ema.to(device=device)
        ema.copy_to(model.parameters())
        
    model.eval()
    model.to(device=device)

    save_all_images = args.basepath + 'all/'
    

    mult = 1


    fac = (minsig / maxsig) ** (1 / (noisenum - 1))
    torch.set_grad_enabled(False) 
    savectr = 0
    for imgctr in range(numofimg): 
        noisy_startimg = 0.5* torch.ones(bs,3,imsize,imsize, device = device) + maxsig * torch.randn(bs,3,imsize,imsize, device = device)
        #noisy_startimg = maxsig * torch.randn(bs,3,imsize,imsize, device = device)
        genimg = noisy_startimg
        img_path = save_path + str(imgctr) + '/'
        prev_sig = maxsig 
        #epsilon = 0.05
        
        for noisectr in range(1,noisenum):
            current_sig = prev_sig * fac

            if not args.skip_pred:
            ### PREDICTOR #####
                score = model(genimg,prev_sig*torch.ones(genimg.shape[0], device = device))
                vardiff = prev_sig ** 2 - current_sig ** 2
                if condx0:
                    condprob = mult * score + ( 1 / maxsig ** 2 ) * (noisy_startimg - genimg)
                    if noaddnoise:
                        genimg = genimg +  vardiff * condprob
                    else:
                        genimg = genimg +  vardiff * condprob + (vardiff) ** 0.5  * torch.randn_like(genimg)
                else:
                    if noaddnoise:
                        genimg = genimg + vardiff * score
                    else:
                        genimg = genimg + vardiff * score + (vardiff) ** 0.5  * torch.randn_like(genimg)

            for _ in range(args.itnum):
                ### Corrector ###
                score = model(genimg.clone(), current_sig * torch.ones(genimg.shape[0], device= device))
                step_AWGN = torch.randn_like(genimg)
                epsilon = 2 * (r * torch.norm(step_AWGN, 'fro') / torch.norm( score, 'fro')) ** 2
                if condx0:
                    condprob = mult * score + ( 1 / maxsig ** 2 ) * (noisy_startimg - genimg)
                    if noaddnoise:
                        genimg = genimg + epsilon * condprob 
                    else:
                        genimg = genimg + epsilon * condprob  + (2 * epsilon) ** 0.5  * step_AWGN
                else:
                    if noaddnoise:
                        genimg = genimg + epsilon * score
                    else:
                        genimg = genimg + epsilon * score + (2 * epsilon) ** 0.5  * step_AWGN


            if noisectr % printfreq == 0:
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                save_image(genimg , img_path + str(noisectr) + '.png')
            prev_sig = current_sig

            
        netted = model(genimg.clone(),current_sig*torch.ones(genimg.shape[0], device = device))
        lastdend = genimg + mult * current_sig **2 * netted 
        if args.saveall:
            if not os.path.exists(save_all_images):
                os.makedirs(save_all_images)
            for batchcount in range(bs):
                save_image(lastdend[batchcount], save_all_images + str(savectr) + '.png')
                savectr += 1
                if savectr >= maximgs:
                    return
        else:
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            save_image(genimg, img_path + 'last.png')
            save_image(lastdend, img_path + 'lastden.png')

if __name__ == "__main__":
    t=time.time()
    main()
    print(f'generation took {time.time()-t} seconds')

