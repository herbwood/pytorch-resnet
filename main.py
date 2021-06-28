import time
import argparse

from train import resnet_training

def main(args):
    
    total_start_time = time.time()

    if args.training:
        resnet_training(args)

    
    print(f"Done! ; {round((time.time() - total_start_time)/60, 3)}min spend")


if __name__  == "__main__":

    parser = argparse.ArgumentParser(description='Parsing Method')

    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--data_path', default='./data', type=str,
                        help='Original data path')
    parser.add_argument('--save_path', default='./save', type=str,
                        help='Model checkpoint file path')
    
    parser.add_argument('--img_size', default=256, type=int,
                        help='Image resize size; Default is 256')
    parser.add_argument('--download_cifar10', default=False, type=bool,
                        help='Whether to download CIFAR-10')

    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'cosine', 'cosine_warmup', 'reduce_train', 'lambda']
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD'; Default is AdamW")
    parser.add_argument('--scheduler', default='constant', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is constant")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")

    parser.add_argument('--num_epochs', default=10, type=int, 
                        help='Training epochs; Default is 10')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size; Default is 16')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')

    parser.add_argument('--test_batch_size', default=32, type=int, 
                        help='Test batch size; Default is 32')

    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)
