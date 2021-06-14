import os
import gc
import time
import logging 

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms 
from torch.cuda.amp import GradScaler, autocast

from dataset import CIFAR10Dataset
from model import ResNet


def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, scaler, logger, device):

    start_time_e = time.time()
    model = model.train()

    for i, (data, target) in enumerate(dataloader):

        optimizer.zero_grad()

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast():
            output = model(data)
            loss = F.cross_entropy(output, target)

            pred = output.max(1, keepdim=True)[1]
        
        acc = pred.eq(target.view_as(pred)).sum().item() / len(target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if args.scheduler in ['constant', 'warmup']:
            scheduler.step()
        if args.scheduler == 'reduce_train':
            scheduler.step(loss)

        if i == 0 or freq == args.print_freq or i == len(dataloader):
            batch_log = f"[Epoch:{epoch+1}] \
                          [{i}/{len(dataloader)}] \
                          | train_loss:{loss.item():2.3f} \
                          | train_acc:{acc:02.2f} \
                          | learning_rate:{optimizer.param_groups[0]['lr']:3.6f} \
                          | spend_time:{((time.time()-start_time_e)/60):3.2f}min"

            write_log(logger, batch_log)
            freq = 0
        freq += 1
        
def valid_epoch(args, model, dataloader, device):

    model = model.eval()
    val_loss, val_acc = 0, 0

    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)

                pred = output.max(1, keepdim=True)[1]
            
            acc = pred.eq(target.view_as(pred)).sum().item() / len(target)

            val_loss += loss.item()
            val_acc += (acc * 100)

    return val_loss, val_acc


