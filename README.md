# Implementation for the paper "Maximum a Posteriori Training of Diffusion Models for Image Restoration"


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Used repository


This uses the code provided at https://github.com/yang-song/score_sde_pytorch for its NCSNpp architecture, making up the folder ncsnpp. The code is an implementation for the paper

Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." International Conference on Learning Representations. 2021.

The files are under the Apache License, Version 2.0 (License file included in the folder ncsnpp).

The code in the repository was modified in the following way:

Modified import statements to fit folder-structure
added updExponentialMovingAverage to ncnspp.model.ema



## Before training or testing

Before use of checkpoints (found at https://github.com/yang-song/score_sde_pytorch), please run resave_params_ema.py This re-saves pre-trained checkpoints to make them compatible

Datasets will be automatically downloaded into the folders provided, if they are not there already.

```pre
python resave_params_ema.py --netloc /path/to/net/checkpointname.ckpt --saveloc /path/to/desired/reparametrized.ckpt -ds bedroom
```


## Training

MUPET is done via IP_MUPET_train_script.py

for CIFAR10:
```train
python IP_MUPET_train_script.py --pre_path /path/to/desired/reparametrized.ckpt --saveloc /path/to/desired/reparametrized.ckpt --lossscaling divbynoise --basepath /path/for/storage/of/logsandckpts/ --convmarg 5e-5 --lrate 1e-5 --scaleconvmarg --epoch 100 -ds cifar10 
```

For LSUN bedroom:
```train
python IP_MUPET_train_script.py --pre_path /path/to/desired/reparametrized.ckpt --saveloc /path/to/desired/reparametrized.ckpt --lossscaling divbynoise --basepath /path/for/storage/of/logsandckpts/  --convmarg 5e-5 --lrate 1e-5 --scaleconvmarg -ds bedroom -bs 8 --stopnumb 500 --epochs 100 --maxnoisesig 378 
```


## Evaluation
Exact hyper-parameters and inference algorithm **VE DiffPIR** can be found here: [MUPET Supplemental](./MUPET_supplemental.pdf)

To generate images from the prior, please use

```eval
python ve_img_gen.py --netpath /path/to/net.ckpt --saveall -ds cifar10 -bs 1024 --maximgs 50000 --basepath /path/for/storage/of/images/
```
For evaluation of the FID score, we used pytorch_fid version 0.3.0
```eval
python -m pytorch_fid path/to/images/ path/to/other/images/
```


For the VE version of DiffPIR (evaluation), please use

```eval
python DiffPIR_re.py --basepath /path/for/storage/of/images/ --dspath /path/to/dataset/location/ --netpath /path/to/net.ckpt -ip pix_comp -ipf 0.9 -it 1000 -ms 1.0 -ns 0.01 --mixparam 0.6 -ds bedroom -bs 32 -ox0 --full 
```
