# https://github.com/pytorch/examples/blob/main/imagenet/main.py
# adapted for vision transformer and webdataset
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.distributed.optim import ZeroRedundancyOptimizer

import webdataset as wds
import wandb
import transformers
import math

torch.set_float32_matmul_precision('high')

model_names = sorted(name for name in models.__dict__
 
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='ViT')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--d_ff", type=int, default=3072)
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--t_warm_up", type=int, default=10)
parser.add_argument("--t_cos_anneal", type=int, default=120)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--dropout', default=0.0, type=float, metavar='D',
                    help='dropout rate (default: 0.0)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', action='store_true',
                    help='initialize ViT-MAE backbone from Hugging Face pretrained weights')
parser.add_argument('--pretrained-model-name', default='facebook/vit-mae-base', type=str,
                    help='Hugging Face model id used when --pretrained is set')
parser.add_argument('--finetune-checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint to load before training')
parser.add_argument("--mask-ratio", type=float, default=0.75,
                    help='mask ratio used by MAE pretraining')
parser.add_argument('--norm-pix-loss', action='store_true',
                    help='use normalized pixels as MAE reconstruction targets')
parser.add_argument("--decoder-d-model", type=int, default=512)
parser.add_argument("--decoder-num-heads", type=int, default=16)
parser.add_argument("--decoder-d-ff", type=int, default=2048)
parser.add_argument("--decoder-num-layers", type=int, default=8)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--compile', action='store_true', help="use torch.compile to compile the model")
parser.add_argument('--bf16', action='store_true', help="use bfloat16 precision for training")
parser.add_argument('--use_zero', action='store_true', help="use zero optimizer from deepspeed")

best_train_loss = float("inf")
IMAGENET_TRAIN_SAMPLES = 1281167
NUM_CLASSES = 1000


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type =='cuda':
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function

        # args.batch_size is interpreted as per-node batch in this launch setup.
        # Compute scheduler steps from the global batch size across all nodes.
        world_size_for_batch = args.world_size if args.world_size > 0 else 1
        global_batch_size = args.batch_size * (world_size_for_batch if args.distributed else 1)
        args.train_batches = IMAGENET_TRAIN_SAMPLES // global_batch_size
        args.total_steps = args.epochs * args.train_batches

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:

        # args.batch_size is interpreted as per-node batch in this launch setup.
        # Compute scheduler steps from the global batch size across all nodes.
        world_size_for_batch = args.world_size if args.world_size > 0 else 1
        global_batch_size = args.batch_size * (world_size_for_batch if args.distributed else 1)
        args.train_batches = IMAGENET_TRAIN_SAMPLES // global_batch_size
        args.total_steps = args.epochs * args.train_batches
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def label_to_index(label):
    return int(label) 

def main_worker(gpu, ngpus_per_node, args):
    global best_train_loss
    args.gpu = gpu

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    run = None
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        is_main_process = not args.distributed or (args.distributed and args.rank == 0)
    
        if is_main_process:
            run = wandb.init(project="MAE", config=args)
        else:
            run = None
    # create model
    if args.pretrained:
        print(f"=> using pre-trained ViT-MAE pretraining model '{args.pretrained_model_name}'")
        model = transformers.ViTMAEForPreTraining.from_pretrained(args.pretrained_model_name)
    else:
        print("=> creating ViT-MAE pretraining model from scratch (Hugging Face implementation)")
        model_config = transformers.ViTMAEConfig(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_channels=args.in_channels,
            hidden_size=args.d_model,
            num_attention_heads=args.num_heads,
            intermediate_size=args.d_ff,
            num_hidden_layers=args.num_layers,
            mask_ratio=args.mask_ratio,
            norm_pix_loss=args.norm_pix_loss,
            decoder_hidden_size=args.decoder_d_model,
            decoder_num_hidden_layers=args.decoder_num_layers,
            decoder_num_attention_heads=args.decoder_num_heads,
            decoder_intermediate_size=args.decoder_d_ff,
        )
        model = transformers.ViTMAEForPreTraining(model_config)

    if args.compile:
        print("compiling the model with torch.compile...")
        model = torch.compile(model)

    if not use_accel:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device.type == 'cuda':
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(device)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)


    if args.finetune_checkpoint:
        load_finetune_checkpoint(model, args.finetune_checkpoint, reset_head=args.reset_head)

    # define optimizer

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    if args.use_zero:
        optimizer = ZeroRedundancyOptimizer(model.parameters(), 
                    optimizer_class=torch.optim.AdamW, 
                    lr=args.lr, 
                    betas=(0.9, 0.95),
                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), 
                                      weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_train_loss = checkpoint.get('best_train_loss', checkpoint.get('best_val_loss', checkpoint.get('best_acc1', float("inf"))))
            if args.gpu is not None:
                # best metric may be from a checkpoint from a different GPU
                if isinstance(best_train_loss, torch.Tensor):
                    best_train_loss = best_train_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(
            IMAGENET_TRAIN_SAMPLES,
            (3, args.image_size, args.image_size),
            1000,
            transforms.ToTensor(),
        )
    else:

        world_size = args.world_size if args.distributed else 1
        num_workers = max(1, args.workers) 
        train_samples_per_worker = IMAGENET_TRAIN_SAMPLES // (world_size * num_workers)

        # MAE use less data augmentation
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_dataset = wds.WebDataset(args.data + "imagenet1k-train-{0000..1023}.tar",
                                       nodesplitter=wds.split_by_node,
                                       workersplitter=wds.split_by_worker,
                                       shardshuffle=1024).\
            shuffle(1000).decode("pil").to_tuple("jpg", "cls").map_tuple(
             train_transforms, label_to_index
        ).with_epoch(train_samples_per_worker) 

        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    # else:
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True, persistent_workers=False)

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, current_lr = train(
            train_loader, model, optimizer, epoch, device, args
        )
        if run is not None:
            log_data = {
                "train/loss": float(train_loss),
                "train/lr": float(current_lr),
                "epoch": epoch,
            }
            run.log(log_data)

        # scheduler.step()
        
        # remember best train loss and save checkpoint
        is_best = train_loss < best_train_loss
        best_train_loss = min(train_loss, best_train_loss)

        # Fix for ZeroRedundancyOptimizer: consolidate state on all ranks, save only on rank 0
        if args.use_zero and args.distributed:
            optimizer.consolidate_state_dict(to=0)
            if args.rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_train_loss': best_train_loss,
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict()
                }, is_best)
        else:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_train_loss': best_train_loss,
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict()
                }, is_best)


def train(train_loader, model, optimizer, epoch, device, args):
    
    use_accel = not args.no_accel and torch.accelerator.is_available()
    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)

    progress = ProgressMeter(
        args.train_batches,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):

        if not args.finetune_checkpoint:
            current_lr = learning_rate_schedule(
                t=epoch * args.train_batches + i,
                lr_max=args.lr,
                lr_min=args.min_lr,
                t_warm_up=args.t_warm_up,
                total_steps=args.total_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            current_lr = args.lr
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        images = images.to(device, non_blocking=True)

        # compute reconstruction loss
        if args.bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(pixel_values=images)
                loss = outputs.loss
        else:
            outputs = model(pixel_values=images)
            loss = outputs.loss

        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and (not args.distributed or (args.distributed and args.rank == 0)):
            progress.display(i + 1)

    if args.distributed:
        losses.all_reduce()
            
    return losses.avg, current_lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def learning_rate_schedule(t, lr_max, lr_min, t_warm_up, total_steps):
    if total_steps <= 0:
        return lr_max
    if t < t_warm_up:
        return lr_max * float(t + 1) / float(max(1, t_warm_up))
    progress = (t - t_warm_up) / float(max(1, total_steps - t_warm_up))
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):    
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_finetune_checkpoint(model, checkpoint_path, reset_head=False):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if reset_head:
        print("=> --reset-head is ignored for MAE pretraining checkpoints")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(
        f"=> loaded checkpoint '{checkpoint_path}' "
        f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
    )


class MixAugmentCollate:
    def __init__(self, alpha_cutmix, alpha_mixup, num_classes):
        transforms_list = []
        if alpha_mixup > 0:
            transforms_list.append(v2.MixUp(alpha=alpha_mixup, num_classes=num_classes))
        if alpha_cutmix > 0:
            transforms_list.append(v2.CutMix(alpha=alpha_cutmix, num_classes=num_classes))
        self.mix_transform = v2.RandomChoice(transforms_list) if transforms_list else None

    def __call__(self, batch):
        images, targets = default_collate(batch)
        if self.mix_transform is None:
            return images, targets
        return self.mix_transform(images, targets)


if __name__ == '__main__':
    main()
