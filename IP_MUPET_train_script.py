import torch
import torch.optim as optim
import os
import argparse
from degs import set_of_degradations, only_denoising
import logging
from getconf import get_ds_and_net
from DEQ_train_class import MAP_trainer
from custom_losses import clampLPIPS
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


""" Training script for MUPET """
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--lrate', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--stepsize', type=float, default=0.05, help='GD-stepsize')
    parser.add_argument('--scheduler_stepsize', type=int, default=1000000, help='scheduler_stepsize')
    parser.add_argument('--stepgamma', type=float, default=1.0, help='scheduler step gamma')
    parser.add_argument('-cm','--convmarg', type=float, default=5e-5, help='margin for convergence')
    parser.add_argument('--bpstepsize', type=float, default=0.05, help='step-size used for the GD step in DEQ back-propagation')
    parser.add_argument('--stopmarg', type=float, default=1000, help='If any absolute value entry of estimate is larger than this, stop iteration')
    parser.add_argument('-ds','--dataset', type=str, default= 'cifar10', help='Name for the save_file')
    parser.add_argument('--maxit', type=int, default=500, help='maximum iterations for DEQ estimation')
    parser.add_argument('--lossscaling', type=str, default='divbynoise', help='how the DEQloss is scaled w.r.t. sigma and stepsize')
    parser.add_argument('--maxnoisesig', type=float, default=50, help='maximum noise-level')
    parser.add_argument('--epochs', type=int, default= 100, help='Number of epochs')
    parser.add_argument('-bs','--batchsize', type=int, default= 128, help='batch size')
    parser.add_argument('--maxblocksize', type=int, default= 32, help='maximum blocksize for inpainting')
    parser.add_argument('--stopnumb', type=int, default= 1e9, help='stop epoch after this many iterations')
    parser.add_argument('--DEQloss', type=str, default= 'MSE', help='Lossfunc for DEQ back-propagation')
    parser.add_argument('-n','--name', type=str, default= 'MUPETtrain', help='Name for the save_file')
    parser.add_argument('--noadam', action='store_false', dest='adam', help='do not use adam')
    parser.add_argument('-scm','--scaleconvmarg',  default= False, action='store_true', help='scale convmargin with AWGN sigma')
    parser.add_argument('--onlyden',  default= False, action='store_true', help='only use denoising and no other inverse problems')
    parser.add_argument('--DSMfac', type=float, default=1.0, help='factor for DSMloss')
    parser.add_argument('--lpips_inDEQlossfac', type=float, default=1.0, help='factor for lpips in DEQloss. Only relevant if using DEQloss MSELPIPS')
    parser.add_argument('--inittype',  type = str, default = 'randstart', help='how to initialize GD procedure for \
                        MAP estimation from: randstart, bigstepfirst, virtobs, maxrand, usex0')
    parser.add_argument('--dspath', type=str, required = True, help='path to data-set location')
    parser.add_argument('--basepath', type=str, required = True, help='location for log_files and checkpoints')
    parser.add_argument('--pre_path', type=str, required = True, help='directory of pretrained net.')
    args = parser.parse_args()

    print(args)


    dennet, training_data, ema, imsizes, pretrain_path = get_ds_and_net(dataset = args.dataset, 
                                                                        batchsize = args.batchsize, device =device, root = args.dspath, 
                                                                        pretrain_path = args.pre_path)

    DSMloss = torch.nn.MSELoss()
    if args.DEQloss == 'MSE':
        DEQlossfunc = torch.nn.MSELoss()
    elif args.DEQloss == 'L1':
        DEQlossfunc = torch.nn.L1Loss()
    elif args.DEQloss == 'LPIPS':
        DEQlossfunc = clampLPIPS(reduction = 'mean', device = device)
    else:
        raise NotImplementedError(f'loss function {args.DEQloss} not implemented')


    save_path = args.basepath + args.name + ".ckpt"


    optimizer = optim.Adam(params=dennet.parameters(), lr=args.lrate, eps = 1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.scheduler_stepsize, gamma=args.stepgamma)

    ### load checkpoints ###
    if os.path.exists(save_path):
        print('reloading nets')
        if torch.cuda.is_available():
            dict = torch.load(save_path)
        else:
            dict = torch.load(save_path, map_location='cpu')
        start_epoch = dict['epoch']
        ema.load_state_dict(dict['ema'])
        optimizer.load_state_dict(dict['optimizer_state'])
        dennet.load_state_dict(dict['model'])
        if args.scheduler_stepsize < args.epochs and args.stepgamma < 1:
            scheduler.load_state_dict(dict['scheduler_state'])
    else:
        if torch.cuda.is_available():
            dict = torch.load(pretrain_path)
        else:
            dict = torch.load(pretrain_path, map_location='cpu')
        ema.load_state_dict(dict['ema'])
        start_epoch = 0
        dennet.load_state_dict(dict['model']) #, strict=False)
        print('loaded pretrained ncsnpp')

    ema.to(device=device)
    if args.onlyden:
        setofdeg = only_denoising(device = device, noise_sigmas = [0.01, args.maxnoisesig], imsizes = imsizes)
    else:
        setofdeg = set_of_degradations(device = device, 
                      ipset = ['denoising', 'pixel_inpaint', 'bw_inpaint', 'blur', 'super_resolution'], noise_sigmas = [0.01, args.maxnoisesig],
                      pixel_chances = [0.0, 1.0], blocksizes = [0, args.maxblocksize], blurfacs = [0.0, 8.0], srfacs = [1,8], imsizes = imsizes)


    
    if not os.path.exists(args.basepath):
            os.makedirs(args.basepath)

    log_path = args.basepath + args.name + '.log'
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename= log_path, level=logging.DEBUG)
    logger.info(args)
    dennet.to(device=device)

    trainer = MAP_trainer(dennet = dennet, ema = ema, logger = logger, optimizer = optimizer, scheduler = scheduler, device = device, 
                        save_path = save_path, DSMloss = DSMloss, DEQlossfunc = DEQlossfunc, trainepochs = args.epochs,
                        start_epoch = start_epoch, degset = setofdeg,  maxit = args.maxit, stepsize = args.stepsize,
                        adambool = args.adam, convmarg = args.convmarg, stopmarg = args.stopmarg, bpstepsize = args.bpstepsize, 
                        inittype = args.inittype, DSMfac = args.DSMfac, lossscaling = args.lossscaling,
                        scaleconvmarg = args.scaleconvmarg)

    trainer.train(training_data, args.stopnumb)

    print(args.name, 'complete')

if __name__ == "__main__":
  main()
