import torch
import random
import numpy as np

def set_random_seeds():
    '''
    Set the random seeds for deterministic behavior
    '''
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic=True