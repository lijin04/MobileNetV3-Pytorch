import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy

from preprocess import load_data
from model import MobileNetV3

import argparse
from tqdm import tqdm
import time
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser("MobileNetV3 Tiny ImageNet Training")

    parser.add_argument("--dataset-mode", type=str, default="TINY_IMAGENET")
    parser.add_argument("--data", default='data/tiny-imagenet-200')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01, help="base learning rate for RMSprop")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument('--model-mode', type=str, default="LARGE")
    parser.add_argument("--load-pretrained", type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--multiplier', type=float, default=1.0)
    parser.add_argument('--print-interval', type=int, default=30)
    parser.add_argument('--workers', type=int, default=16)

    # 数据增强
    parser.add_argument('--aa', type=str, default='rand-m9-n2-mstd0.5', help="AutoAugment policy")
    parser.add_argument('--reprob', type=float, default=0.1, help="RandomErasing probability")

    # 学习率调度
    parser.add_argument('--lr-step-size', type=int, default=60, help="decrease lr every step-size epochs")
    parser.add_argument('--lr-gamma', type=float, default=0.1, help="decrease lr by a factor")
    parser.add_argument('--warmup-epochs', type=int, default=5, help="number of warmup epochs")

    # EMA
    parser.add_argument('--ema-decay', type=float, default=0.9999, help="EMA decay")

    # 分布式
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def init_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        if args.rank == 0:
            print(f"分布式训练: world_size={args.world_size}, local_rank={args.local_rank}")
    else:
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EMA:
    """Exponential Moving Average - 参考参考脚本实现"""
    def __init__(self, model, decay=0.9999):
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = decay

    def update_parameters(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)


def add_weight_decay(model, weight_decay=1e-5):
    """将 weight decay 只应用到 Conv 和 Linear 的 weight，不应用到 bias 和 bn"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class CosineLR:
    """Cosine Annealing LR with warmup"""
    def __init__(self, optimizer, total_epochs=200, warmup_epochs=5, warmup_lr_init=1e-6, min_lr=1e-5):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        """epoch 从 1 开始"""
        if epoch <= self.warmup_epochs:
            # warmup: 从 warmup_lr_init 线性上升到 base_lr
            lr = self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * epoch / self.warmup_epochs
        else:
            # cosine annealing: 从 base_lr 平滑下降到 min_lr
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def reduce_tensor(tensor, world_size):
    """多卡同步 tensor"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train(train_loader, model, criterion, optimizer, epoch, args, ema=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(args.local_rank, non_blocking=True), target.cuda(args.local_rank, non_blocking=True)

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update_parameters(model)

        # 多卡同步
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            acc1 = reduce_tensor(acc1, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)
        else:
            reduced_loss = loss.data

        batch_size = data.size(0)
        losses.update(reduced_loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0 and args.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f'Train: [{epoch:>3d}][{i:>4d}/{len(train_loader)}]  '
                f'Loss: {losses.val:.4f} ({losses.avg:.4f})  '
                f'Acc@1: {top1.val:.2f} ({top1.avg:.2f})  '
                f'Acc@5: {top5.val:.2f} ({top5.avg:.2f})  '
                f'LR: {lr:.6f}'
            )

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args, log_suffix=""):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.cuda(args.local_rank, non_blocking=True), target.cuda(args.local_rank, non_blocking=True)

            output = model(data)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            batch_size = data.size(0)
            losses.update(reduced_loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_interval == 0 and args.rank == 0:
                print(
                    f'Test_{log_suffix}: [{i:>4d}/{len(val_loader)}]  '
                    f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'Loss: {losses.val:.4f} ({losses.avg:.4f})  '
                    f'Acc@1: {top1.val:.2f} ({top1.avg:.2f})  '
                    f'Acc@5: {top5.val:.2f} ({top5.avg:.2f})'
                )

    if args.rank == 0:
        print(f' * Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%')

    return losses.avg, top1.avg, top5.avg


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        n_classes = x.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def plot_curves(train_losses, val_losses, train_acc1, val_acc1, args):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{args.dataset_mode} - Loss Curve ({args.model_mode})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc1, 'b-', label='Train Acc@1')
    ax2.plot(epochs, val_acc1, 'r-', label='Val Acc@1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{args.dataset_mode} - Accuracy Curve ({args.model_mode})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if not os.path.isdir("reporting"):
        os.mkdir("reporting")
    filename = f"best_model_{args.model_mode}_{args.dataset_mode}"
    plt.savefig(f'./reporting/{filename}_curves.png', dpi=150)
    print(f"曲线图已保存: ./reporting/{filename}_curves.png")
    plt.close()


def main():
    args = get_args()
    init_distributed(args)

    if args.distributed:
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1

    train_loader, test_loader = load_data(args)

    if args.dataset_mode == "CIFAR10":
        num_classes = 10
    elif args.dataset_mode == "CIFAR100":
        num_classes = 100
    elif args.dataset_mode == "IMAGENET":
        num_classes = 1000
    elif args.dataset_mode == "FLOWER102":
        num_classes = 102
    elif args.dataset_mode == "TINY_IMAGENET":
        num_classes = 200

    total_batch = args.batch_size * args.world_size
    if args.rank == 0:
        print(f'num_classes: {num_classes}')
        print(f"使用 GPU 数: {args.world_size}")
        print(f"Total batch size: {total_batch}")

    model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes,
                        multiplier=args.multiplier, dropout_rate=args.dropout)

    if args.distributed:
        model = model.cuda(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.to(device)
        if torch.cuda.device_count() >= 1:
            if args.rank == 0:
                print("num GPUs: ", torch.cuda.device_count())
            model = nn.DataParallel(model).to(device)

    # EMA
    ema = EMA(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
    if args.rank == 0:
        print(f"EMA decay: {args.ema_decay}")

    # 加载 checkpoint
    if args.load_pretrained or args.evaluate:
        filename = "best_model_" + str(args.model_mode) + "_" + str(args.dataset_mode)
        ckpt_path = f'./checkpoint/{filename}_ckpt.t7'
        if args.rank == 0:
            print(f"加载 checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc1 = checkpoint['best_acc1']
        acc5 = checkpoint['best_acc5']
        best_acc1 = acc1
        if args.rank == 0:
            print(f"Load Model Acc@1: {acc1:.2f}%, Acc@5: {acc5:.2f}%, epoch: {epoch}")
    else:
        if args.rank == 0:
            print("init model load ...")
        epoch = 1
        best_acc1 = 0

    # RMSprop 优化器（参考 MobileNetV3 论文和参考脚本）
    parameters = add_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(parameters, lr=args.learning_rate, alpha=0.9, eps=1e-3, weight_decay=0, momentum=args.momentum)

    # Cosine Annealing LR with warmup
    scheduler = CosineLR(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )

    # 损失函数
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(args.local_rank)

    if args.evaluate:
        val_loss, acc1, acc5 = validate(test_loader, model, criterion, args)
        if args.rank == 0:
            print(f"Acc@1: {acc1:.2f}%, Acc@5: {acc5:.2f}%")
        return

    if args.rank == 0:
        if not os.path.isdir("reporting"):
            os.mkdir("reporting")

    train_loss_history = []
    val_loss_history = []
    train_acc1_history = []
    val_acc1_history = []

    start_time = time.time()
    report_filename = f"best_model_{args.model_mode}_{args.dataset_mode}"

    if args.rank == 0:
        log_file = open(f"./reporting/{report_filename}.txt", "w")
    else:
        log_file = None

    for epoch in range(epoch, args.epochs + 1):
        # StepLR step（传入当前 epoch）
        scheduler.step(epoch)

        train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args, ema=ema)

        # 用真实模型验证（EMA 在前期步数不够，不用于验证）
        val_loss, acc1, acc5 = validate(test_loader, model, criterion, args, log_suffix='Raw')

        if args.rank == 0:
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc1_history.append(train_acc1)
            val_acc1_history.append(acc1)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                print('Saving..')
                best_acc5 = acc5
                state = {
                    'model': model.state_dict(),
                    'model_ema': ema.model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'epoch': epoch,
                }
                # 保存 EMA 半精度模型用于推理
                state_ema = {'model': deepcopy(ema.model).half()}

                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, f'./checkpoint/{report_filename}_ckpt.t7')
                torch.save(state_ema, f'./checkpoint/{report_filename}_ema.pth')

            time_interval = time.time() - start_time
            time_split = time.gmtime(time_interval)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Training time: {time_interval:.1f}s ({time_split.tm_hour}h {time_split.tm_min}m {time_split.tm_sec}s)", end='')
            print(f" Test best acc@1: {best_acc1:.2f}% acc@1: {acc1:.2f}% acc@5: {acc5:.2f}% lr: {current_lr:.6f}")

            log_file.write(f"Epoch: {epoch} Best acc: {best_acc1:.2f} Test acc: {acc1:.2f} acc5: {acc5:.2f} lr: {current_lr:.6f}\n")
            log_file.write(f"Training time: {time_interval:.1f}s\n")

    if args.rank == 0:
        log_file.close()
        plot_curves(train_loss_history, val_loss_history, train_acc1_history, val_acc1_history, args)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
