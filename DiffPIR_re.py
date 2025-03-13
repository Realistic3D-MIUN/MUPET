import torch
import os
import argparse
from MAP_tst_utils import DiffPIR
from gen_utils import getdeg
from setseed import set_seed
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import ssim

from gen_utils import simplegetnet, getds

"""
Variance-Exploding version based on the algorithm presented in:
Zhu, Yuanzhi, et al. "Denoising diffusion models for plug-and-play image restoration." 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
"""


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds','--dataset', type=str, default= 'cifar10', help='choose dataset from: bedroom, cifar10')
    parser.add_argument('-ns','--noise_sigma', type=float, default= 0.01, help='added noise')
    parser.add_argument('-r','--snr', type=float, default= 0.16, help='signal to noise ratio for corrector')
    parser.add_argument('-bs','--batchsize', type=int, default= 256, help='batch-size')
    parser.add_argument('-it','--iterations', type=int, default= 1500, help='# of iterations')
    parser.add_argument('--logfreq', type=int, default= float('inf'), help='frequency of saved images')
    parser.add_argument('-ms','--maxsig', type=float, default= 50, help='maximum sigma for regularization. Only in effect if larger than noise sigma')
    parser.add_argument('-f','--ds_factor', type=float, default= -1, help='factor for the downscaling of regularization. Must not be larger than 1')
    parser.add_argument('--lamb', type=float, default= 1, help='weight for regularization. 1 is normal, increase for stronger regularization')
    parser.add_argument('--mixparam', type=float, default= 0.5, help='number to weigh randomness 1 is purely random 0 is deterministic')
    parser.add_argument('--noema',  default= False, action='store_true', help='do not use ema params: False uses EMA params')
    parser.add_argument('--onestepprox',  default= False, action='store_true', help='skip predictor step')
    parser.add_argument('-non', '--nonoise',  default= False, action='store_true', help='skip noise-adding step')
    parser.add_argument('-ox0', '--obsx0',  default= False, action='store_true', help='use rescaled observation plus noise as x0')
    parser.add_argument('--full',  default= False, action='store_true', help='run through whole test set')
    parser.add_argument('--save_fullbatch',  default= False, action='store_true', help='save each image seperately')
    parser.add_argument('-np','--netpath',  type = str, required = True , default = 'none', help='path to nn')
    parser.add_argument('-ip', '--inverse_problem', type = str, default='pix_comp', help='type of inverse problem, chosen from pix_comp, bw_comp, sr, blur')
    parser.add_argument('-ipf', '--ipfac', type = float, default=0.9, help='level of ip')
    parser.add_argument('--basepath', type=str, required = True, help='location for log_files and saved images')
    parser.add_argument('--dspath', type=str, required = True, help='location to data-set location')
    args = parser.parse_args()
    print(args)

    basepath = args.basepath

    set_seed(seed = 42, inference = True)
    noise_sigma = args.noise_sigma
    maxit = args.iterations
    log_freq = args.logfreq
    dataset = args.dataset
    assert args.ds_factor <= 1, "fac must be no greater than 1"
    MSEloss = torch.nn.MSELoss()
    ssimloss = lambda x, y: ssim(x, y, data_range = 1, size_average = True)
    lpipsloss = LearnedPerceptualImagePatchSimilarity(net_type = 'alex', reduction = 'mean', normalize = True).to(device)

    test_data, imsizes = getds(args.dataset, args.batchsize, root = args.dspath)
    inverse_problem = args.inverse_problem 
    ipfac = args.ipfac
    deg, adj, getx0 = getdeg(inverse_problem = inverse_problem, ipfac = ipfac, imsizes = imsizes, device = device, batchsize = args.batchsize)


    model = simplegetnet(device = device, netpath = args.netpath, dataset = dataset, emabool = not args.noema)
    log_path = basepath \
          + dataset + '/' + args.inverse_problem + str(args.ipfac) + '_l' + str(args.lamb) + '_mp' \
            + str(args.mixparam)+ '_it' + str(args.iterations) + '_ms' + str(args.maxsig) +  '/'

    try:
        os.makedirs(log_path)
    except:
        pass

    with torch.no_grad():
        DiffPIR(test_data = test_data, log_path = log_path, dennet = model, device = device, maxsig = args.maxsig, 
                            MSEloss = MSEloss, ssimloss = ssimloss, lpipsloss = lpipsloss,  noise_sigma = noise_sigma, deg = deg, adj = adj, getx0 = getx0, maxit = maxit, log_freq = log_freq, 
                            dsfac = args.ds_factor, obsx0 = args.obsx0, lamb = args.lamb, mixparam = args.mixparam, onestepprox = args.onestepprox, full = args.full,
                            save_fullbatch = args.save_fullbatch)

if __name__ == "__main__":
  main()