import torch
import numpy as np
import random
import os

def set_seed(seed=42, inference = False):
    """Set seed for reproducibility."""
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed) 

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if inference:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False