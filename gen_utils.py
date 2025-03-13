import torch
from ncsnpp.models.ncsnpp import NCSNpp
from torchvision import transforms, datasets
from numpy import random
import deepinv.physics as ph
import matplotlib.pyplot as plt
from torchvision.utils import save_image
#from ncsnpp.models.ema import ExponentialMovingAverage
from torch_ema import ExponentialMovingAverage as torchema
from ncsnpp.configs.ve.bedroom_ncsnpp_continuous import get_config as bedroomconfig
from ncsnpp.configs.ve.cifar10_ncsnpp_continuous import get_config as cifarconfig



def getdeg(inverse_problem, ipfac, imsizes, device, batchsize):
    if inverse_problem == 'denoising' or inverse_problem== 'denoise':
        deg = ph.Inpainting(tensor_size = (3,imsizes[0], imsizes[1]), mask = 1, device = device)
        getx0 = lambda x: x
        return deg, deg, getx0
    elif inverse_problem in ['pix_comp' , 'pixel_inpaint']:
        deg = ph.Inpainting(tensor_size = (3,imsizes[0], imsizes[1]), mask = 1 - ipfac, device = device)
        #adj = deg.A_adj
        getx0 = lambda x: x + 0.5 * (torch.ones_like(x)  - deg(torch.ones_like(x)))
        return deg, deg, getx0
    elif inverse_problem in ['bw_comp' , 'bw_inpaint']:
        if isinstance(ipfac, list):
            boxsizes = ipfac
        else:
            ipfac = int(ipfac)
            boxsizes = [ipfac, ipfac]
        boxpos1 = random.choice(range(imsizes[0] - boxsizes[0]+1))
        boxpos2 = random.choice(range(imsizes[1] - boxsizes[1]+1))
        mask = torch.ones((3,imsizes[0], imsizes[1])).to(device=device)
        mask[:,boxpos1:boxpos1+boxsizes[0],boxpos2:boxpos2+boxsizes[1]] = torch.zeros((3,boxsizes[0], boxsizes[1])).to(device=device)
        deg = ph.Inpainting(tensor_size = (3,imsizes[0], imsizes[1]), mask = mask, device = device)
        getx0 = lambda x: x + 0.5 * (torch.ones_like(x)  - deg(torch.ones_like(x)))
        return deg, deg, getx0
        
    elif inverse_problem == 'blur' or inverse_problem == 'deblur':
        filter = ph.blur.gaussian_blur(sigma=ipfac)
        deg = ph.Blur(filter=filter, device = device)
        getx0 = lambda x: x
        return deg, deg.A_adjoint, getx0
    elif inverse_problem in ['super_resolution', 'sr']:
        if ipfac < 1:
            raise Exception('downsampling factor is smaller than 1')
        #deg = ph.Downsampling(img_size = (batchsize, 3, imsizes[0], imsizes[1]), factor = int(ipfac), filter = 'bicubic', device = device)
        deg = ph.Downsampling(img_size = (3, imsizes[0], imsizes[1]), factor = int(ipfac), filter = 'bicubic', device = device)
        #adj = deg.A_adj
        getx0 = transforms.Resize(imsizes, interpolation = transforms.InterpolationMode.BICUBIC)
        return deg, deg.A_adjoint, getx0
    else:
        raise Exception(f'inverse problem {inverse_problem} not implemented')


def getds(dataset, batch_size, root):

    if dataset == 'CIFAR10' or dataset == 'cifar10':
        transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
        )
        test_dataset = datasets.CIFAR10(root= root, train=False, download = False, transform = transform)
        test_data = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 4, pin_memory = True)
        imsizes = [32,32]
    elif dataset == 'bedroom':
        transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()
                                    ])
        workers = 4
        test_data = datasets.LSUN(root= root, classes = ["bedroom_val"], transform = transform)
        test_data = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 4, pin_memory = True)
        imsizes = [256,256]
    else:
        raise Exception('bad dataset')

    return (test_data, imsizes)

def simplegetnet(device, netpath, dataset = 'cifar10', emabool = True):
    net_path = netpath
    if dataset == 'CIFAR10' or dataset == 'cifar10':
        config = cifarconfig()
    elif dataset == 'bedroom':
        config = bedroomconfig()
    else:
        raise Exception('bad dataset')


    model = NCSNpp(config)
    model = model.to(device=device)

    dict = torch.load(net_path, map_location = device)
    model.load_state_dict(dict['model'], strict=False) #LearnedOpt_state  model
    if emabool:
        ema = torchema(model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(dict['ema'])
        ema.to(device=device)
        ema.copy_to(model.parameters())
    model.eval()
    model.to(device=device)
    return model
