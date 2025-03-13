import torch
import os
import argparse
from MAP_tst_utils import MAPGD_tst
from gen_utils import getdeg
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import ssim

from gen_utils import simplegetnet, getds

"""MAP GD testing script """

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds','--dataset', type=str, default= 'cifar10', help='choose dataset from: bedroom, cifar10')
    parser.add_argument('-ns','--noise_sigma', type=float, default= 0.01, help='added noise')
    parser.add_argument('-ss','--stepsize', type=float, default= 0.05, help='stepsize for GD')
    parser.add_argument('-bs','--batchsize', type=int, default= 16, help='batch-size')
    parser.add_argument('-it','--iterations', type=int, default= 500, help='# of iterations')
    parser.add_argument('--logfreq', type=int, default= 100000, help='after how many iterations an image is saved')
    parser.add_argument('--lamb', type=float, default= 1, help='weight for regularization. 1 is normal, increase for stronger regularization')
    parser.add_argument('--noadam', action='store_false', dest='adam', help='do not use adam')
    parser.add_argument('-ms','--maxsig', type=float, default= 0, help='maximum sigma for regularization. Only in effect if larger than noise sigma')
    parser.add_argument('-f','--factor', type=float, default= 1, help='factor for the downscaling of regularization. Must not be larger than 1')
    parser.add_argument('-b2','--adam_beta2', type=float, default= 0.999, help='beta2 for adam')
    parser.add_argument('-ip', '--inverse_problem', type = str, default='pix_comp', help='type of inverse problem, chosen from pix_comp, bw_comp, sr, blur')
    parser.add_argument('-ipf', '--ipfac', type = float, default=0.9, help='level of ip')
    parser.add_argument('--noema',  default= False, action='store_true', help='do not use ema params: False uses EMA params')
    parser.add_argument('--inittype', type = str, default='usex0', help='chose selection of x0 from: randstart, bigstepfirst, virtobs, maxrand, usex0, noisyx0')
    parser.add_argument('--addnoise', default= False, action='store_true', help='add noise, turning algorithm into langevin dynamics')
    parser.add_argument('--genfromnoise', default= False, action='store_true', help='sets obersvation as pure noise')
    parser.add_argument('-np','--netpath',  type = str, required = True , default = 'none', help='path to nn')
    parser.add_argument('--full',  default= False, action='store_true', help='run through whole test set')
    parser.add_argument('--basepath', type=str, required = True, help='location for log_files and saved images')
    parser.add_argument('--dspath', type=str, required = True, help='location to data-set location')
    args = parser.parse_args()
    print(args)

    noise_sigma = args.noise_sigma
    maxit = args.iterations
    stepsize = args.stepsize
    adam = args.adam
    addnoise = args.addnoise
    genfromnoise = args.genfromnoise
    log_freq = args.logfreq
    dataset = args.dataset
    torch.manual_seed(0)
    fac = args.factor
    if args.maxsig < args.noise_sigma:
        maxsig = args.noise_sigma
        print('maxsig defaulted to noise_sigma')
    else:
        maxsig = args.maxsig
        if not args.factor < 1:
            fac = (args.noise_sigma / maxsig) ** (1 / (args.iterations - 1))
            print(f'automated factor for manually chosen maxsig. Factor is {fac}')

    assert args.factor <= 1, "fac must be no greater than 1"
    if args.maxsig > args.noise_sigma or args.factor < 1:
        downscale = True
        print('using downscaling for regularization')
    else:
        downscale = False

    
        

    
    MSEloss = torch.nn.MSELoss()
    ssimloss = lambda x, y: ssim(x, y, data_range = 1, size_average = True)
    lpipsloss = LearnedPerceptualImagePatchSimilarity(net_type = 'alex', reduction = 'mean', normalize = True).to(device)


    test_data, imsizes = getds(args.dataset, args.batchsize, root = args.dspath)
    inverse_problem = args.inverse_problem 
    ipfac = args.ipfac
    deg, adj, getx0 = getdeg(inverse_problem = inverse_problem, ipfac = ipfac, imsizes = imsizes, device = device, batchsize = args.batchsize)


    model = simplegetnet(device = device, netpath = args.netpath, dataset = dataset, emabool = not args.noema)
    log_path = args.basepath  + '/' + args.inverse_problem + str(args.ipfac) + '_' + str(args.lamb) + '/'

    try:
        os.makedirs(log_path)
    except:
        pass

    MAPGD_tst(test_data = test_data, log_path = log_path, dennet = model, device = device, maxsig = maxsig, downscale = downscale, fac = fac,
                MSEloss = MSEloss, ssimloss = ssimloss, lpipsloss = lpipsloss, noise_sigma = noise_sigma, deg = deg, adj = adj, getx0 = getx0, maxit = maxit, stepsize = stepsize, adam = adam, inittype = args.inittype, 
               addnoise = addnoise, genfromnoise = genfromnoise, log_freq = log_freq, adamb2 = args.adam_beta2, lamb = args.lamb, full = args.full)


if __name__ == "__main__":
  main()