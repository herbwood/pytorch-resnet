import os
import gc
import time
import wandb
import logging 

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms 
from torch.cuda.amp import GradScaler, autocast

from dataset import CIFAR10Dataset
from model import ResNet, Bottleneck
from utils import TqdmLoggingHandler, write_log, optimizer_select, scheduler_select, label_smoothing_loss, cutmix_loss


def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, scaler, logger, device):

    start_time_e = time.time()
    model = model.train()

    for i, (data, target) in enumerate(dataloader):

        optimizer.zero_grad()
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        wandb.log({"examples" : [wandb.Image(data[0])]})

        with autocast():
            if args.cutmix:
                output, loss = cutmix_loss(data, target, model, device, args)
            output = model(data)
            loss = label_smoothing_loss(output, target, device=device)
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
            batch_log = "[Epoch:%d][%d/%d] train_loss:%2.3f  | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                    % (epoch+1, i, len(dataloader), 
                    loss.item(), acc * 100, optimizer.param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60)
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


def resnet_training(args):

    wandb.init()
    wandb.run.name = "pytorch-resnet"
    wandb.config.update(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(' %(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #============Data Load==============#
    #===================================#

    write_log(logger, 'Load data...')
    gc.disable()
    transform_dict = {
        'train' : transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        'valid' : transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    } 

    dataset_dict = {
        'train' : CIFAR10Dataset(basepath=args.data_path, phase='train', transform=transform_dict['train'], download=args.download_cifar10),
        'valid' : CIFAR10Dataset(basepath=args.data_path, phase='valid', transform=transform_dict['valid'])
    }

    dataloader_dict = {
        'train' : DataLoader(dataset_dict['train'], drop_last=True, batch_size=args.batch_size, 
                             shuffle=True, pin_memory=True, num_workers=args.num_workers),
        'valid' : DataLoader(dataset_dict['valid'], drop_last=False, batch_size=args.batch_size, 
                             shuffle=False, pin_memory=True, num_workers=args.num_workers)
    }

    gc.enable()
    write_log(logger, f"Total number of training sets and iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    write_log(logger, 'Instantiating models...')
    model = ResNet(block=Bottleneck, layers=[3,4,6,3], num_classes=10)
    model = model.to(device)

    optimizer = optimizer_select(model, args)
    scheduler = scheduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.train()
        model = model.to(device)
        del checkpoint
    
    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    wandb.watch(model)
    write_log(logger, 'Train start!')
    for epoch in range(start_epoch, args.num_epochs):

        train_epoch(args, epoch, model, dataloader_dict['train'], optimizer, scheduler, scaler, logger, device)
        val_loss, val_acc = valid_epoch(args, model, dataloader_dict['valid'], device)

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])

        write_log(logger, f"Validation Loss : {val_loss:3.3f}")
        write_log(logger, f"Validation Accuracy : {val_acc:3.2f}")

        wandb.log({"Validation Loss" : val_loss,
                   "Validation Accuracy" : val_acc})

        if val_acc > best_val_acc:
            write_log(logger, "Checkpoint saving...")
            torch.save({
                'epoch' : epoch,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'scaler' : scaler.state_dict(),
            }, os.path.join(args.save_path, 'checkpoint.pth.tar'))
            best_val_acc = val_acc
            best_epoch = epoch
        
        else:
            else_log = f"Still {best_epoch} epoch Accuracy ({round(best_val_acc, 2)})% is better..."
            write_log(logger, else_log)

        print(f"Best Epoch : {best_epoch+1}")
        print(f"Best Accuracy : {round(best_val_acc, 2)}")