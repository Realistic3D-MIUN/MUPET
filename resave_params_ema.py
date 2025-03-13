import torch
from ncsnpp.models.ncsnpp import NCSNpp
from torchvision.utils import save_image
import os
from torch import nn
import argparse
import copy
from ncsnpp.models.ema import updExponentialMovingAverage as ExponentialMovingAverage
from ncsnpp.configs.ve.bedroom_ncsnpp_continuous import get_config as bedroomconfig
from ncsnpp.configs.ve.cifar10_ncsnpp_continuous import get_config as cifarconfig

"""
This restructures the ncsnpp to work with torch-ema
"""
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds','--dataset', type=str, required = True, help='choose dataset from: bedroom, cifar10')
    parser.add_argument('--netloc', type=str, required = True, help='location of the pre-trained checkpoint')
    parser.add_argument('--saveloc', type=str, required = True, help='location for the restructured checkpoint to be saved')
    args = parser.parse_args()

    print(args)

    dataset = args.dataset

    if dataset == 'CIFAR10' or dataset == 'cifar10':
        net_path = args.netloc
        config = cifarconfig()
        save_path = args.saveloc
    elif dataset == 'bedroom':
        net_path = args.netloc
        config = bedroomconfig()
        save_path = args.saveloc
    else:
        raise Exception('bad dataset')

    model = NCSNpp(config)
    model = model.to(device=device)

    parmodel = nn.DataParallel(model)

    loaddict = torch.load(net_path, map_location = device)

    parmodel.load_state_dict(loaddict['model'], strict=False) #LearnedOpt_state  model

    savingdict = dict()
    savingdict['model'] = copy.deepcopy(model.state_dict())
    savingdict['ema'] =  copy.deepcopy(loaddict['ema'])
    if True:
        ema = ExponentialMovingAverage(parmodel.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(loaddict['ema'])
        ema.to(device=device)

        ema.copy_to(parmodel.parameters())

        parameters = list(model.parameters())
        shadow_params = [
            p.clone().detach()
            for p in parameters
        ]
        savingdict['ema']['shadow_params'] = shadow_params
        savingdict['ema']['collected_params'] = None
            
    torch.save(savingdict,save_path)
    print(f'new net saved to: {save_path}')
            







if __name__ == "__main__":
  main()



