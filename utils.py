import pickle
import requests
import tarfile

import math
from torch.optim.lr_scheduler import _LRScheduler

from torch import optim
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR, CosineAnnealingLR


class CosineAnnealingWarmUpRestarts(_LRScheduler):

    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def unpickle(file):
    with open(file, 'rb') as f:
        pickle_dict = pickle.load(f, encoding='bytes')
    
    return pickle_dict 

def download_cifar10(url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 
                     target_path='cifar-10-python.tar.gz'):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    if target_path.endswith('tar.gz'):
        tar = tarfile.open(target_path, 'r:gz')
        tar.extractall()
        tar.close()

def write_log(logger, message):
    if logger:
        logger.info(message)

def optimizer_select(model, args):

    if args.optimizer == 'SGD':
        optimizer = SGD(filter(lambda p : p.requires_grad, model.parameters()),
                        args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()),
                        lr=args.lr, eps=1e-8)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), 
                        lr=args.lr, eps=1e-8)
    else:
        raise Exception("Choose optimizer in ['SGD', 'Adam', 'AdamW']")

    return optimizer


def scheduler_select(optimizer, dataloader_dict, args):

    if args.scheduler == 'constant':
        scheduler = StepLR(optimizer, step_size=len(dataloader_dict['train']), gamma=1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs // 3)
    elif args.scheduler == 'cosine_warmup':
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_max=args.num_epochs // 3)
    elif args.scheduler == 'reduce_train':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    elif args.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch : args.lr_lambda ** epoch)
    else:
        raise Exception("Choose scheduler in ['constant', 'cosine', 'cosine_warmup', 'reduce_train', 'lambda']")

    return scheduler 