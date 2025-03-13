import torch
import random
from torchvision import transforms
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

    def picksigma(self):
        noise_randvar = random.uniform(0,1)
        sigma = (self.noise_sigmas[1] - self.noise_sigmas[0] + 1) ** noise_randvar - 1 + self.noise_sigmas[0]
        return sigma

    def getnoiseIP(self):
        deg = lambda x: x
        adj = deg
        getx0 = deg
        return deg, adj, getx0

    def getpixelcompIP(self, fac):
        fac = 1 - fac
        deg = ph.Inpainting(tensor_size = (3,self.imsizes[0], self.imsizes[1]), mask = fac, device = self.device)
        getx0 = lambda x: x + 0.5 * (torch.ones_like(x)  - deg(torch.ones_like(x)))
        return deg, deg, getx0

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
        getx0 = lambda x: x + 0.5 * (torch.ones_like(x)  - deg(torch.ones_like(x)))
        return deg, deg, getx0

    def getblurIP(self, blurfac):
        filter = ph.blur.gaussian_blur(sigma=blurfac)
        deg = ph.Blur(filter=filter, device = self.device)
        adj = deg.A_adjoint
        getx0 = lambda x: x
        return deg, adj, getx0
    
    def getSRIP(self, SRfac):
        if SRfac < 1:
            raise Exception('downsampling factor is smaller than 1')
        deg = ph.Downsampling(img_size = (3, self.imsizes[0], self.imsizes[1]), factor = int(SRfac), filter = 'bicubic', device = self.device)
        adj = deg.A_adjoint
        getx0 = transforms.Resize(self.imsizes, interpolation = transforms.InterpolationMode.BICUBIC)
        return deg, adj, getx0

    def getranddeg(self):
        inverse_problem = random.choice(self.ipset)
        if inverse_problem == 'denoising':
            deg, adj, getx0 = self.getnoiseIP()
            return deg, adj, getx0, inverse_problem
        elif inverse_problem == 'pixel_inpaint':
            fac = random.uniform(self.pixel_chances[0], self.pixel_chances[1])
            deg, adj, getx0 = self.getpixelcompIP(fac)
            return deg, adj, getx0, inverse_problem
        elif inverse_problem == 'bw_inpaint':
            blocksizes = [random.randint(self.blocksizes[0], self.blocksizes[1]), random.randint(self.blocksizes[0], self.blocksizes[1])]
            deg, adj, getx0 = self.getblockcompIP(blocksizes)
            return deg, adj, getx0, inverse_problem
        elif inverse_problem == 'blur':
            blurfac = random.uniform(self.blurfacs[0], self.blurfacs[1])
            deg, adj, getx0 = self.getblurIP(blurfac)
            return deg, adj, getx0, inverse_problem
        elif inverse_problem == 'super_resolution':
            SRfac = random.randint(self.srfacs[0],self.srfacs[1])
            deg, adj, getx0 = self.getSRIP(SRfac)
            return deg, adj, getx0, inverse_problem
        else:
            raise Exception(f'inverse problem {inverse_problem} not implemented')
        
    def getIPandsig(self):
        deg, adj, getx0, degtype = self.getranddeg()
        sigma = self.picksigma()
        return deg, adj, getx0, sigma, degtype

class only_denoising():
    def __init__(self,device, noise_sigmas = [0.01, 50], imsizes = [32, 32]):
        self.noise_sigmas = noise_sigmas
        self.imsizes = imsizes
        self.device = device
        self.deg = ph.Inpainting(tensor_size = (3,self.imsizes[0], self.imsizes[1]), mask = 1, device = self.device)
        self.adj = ph.Inpainting(tensor_size = (3,self.imsizes[0], self.imsizes[1]), mask = 1, device = self.device)

    def picksigma(self):
        noise_randvar = random.uniform(0,1)
        sigma = (self.noise_sigmas[1] - self.noise_sigmas[0] + 1) ** noise_randvar - 1 + self.noise_sigmas[0]
        return sigma

    def getnoiseIP(self):
        deg = ph.Inpainting(tensor_size = (3,self.imsizes[0], self.imsizes[1]), mask = 1, device = self.device)
        adj = deg
        return deg, adj
        
    def getIPandsig(self):
        sigma = self.picksigma()
        return self.deg, self.adj, sigma, 'denoising'