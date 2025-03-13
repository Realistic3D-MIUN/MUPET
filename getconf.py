import torch
from torchvision import transforms, datasets
from ncsnpp.models.ncsnpp import NCSNpp
# from ncsnpp_re.models.ncsnpp import NCSNpp
from torch_ema import ExponentialMovingAverage
from ncsnpp.configs.ve.bedroom_ncsnpp_continuous import get_config as bedroomconfig
from ncsnpp.configs.ve.cifar10_ncsnpp_continuous import get_config as cifarconfig
# from ncsnpp_re.configs.ve.bedroom_ncsnpp_continuous import get_config as bedroomconfig
# from ncsnpp_re.configs.ve.cifar10_ncsnpp_continuous import get_config as cifarconfig

def get_ds_and_net(dataset, batchsize, device, root, pretrain_path):
    if dataset == 'CIFAR10' or dataset == 'cifar10':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5)
            ]
            )
        cifar_path = root
        workers = 4
        training_dataset = datasets.CIFAR10(root= cifar_path, train=True, download = True, transform = transform)
        config = cifarconfig()
        dennet = NCSNpp(config)
        imsizes = [32,32]
    elif dataset == 'bedroom':
        transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(p=0.5)
                                    ])
        workers = 4
        bedrooms_path = root
        training_dataset = datasets.LSUN(root= bedrooms_path, classes = ["bedroom_train"], transform = transform)
        config = bedroomconfig()
        dennet = NCSNpp(config)
        imsizes = [256,256]
    else:
        raise NotImplementedError(f'dataset {dataset} not implemented')

    training_data = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers = workers, pin_memory = True)

    dennet = dennet.to(device=device)
    ema = ExponentialMovingAverage(dennet.parameters(),  decay=config.model.ema_rate)

    return dennet, training_data, ema, imsizes, pretrain_path

