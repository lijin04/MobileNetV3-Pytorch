import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import load_data
from model import MobileNetV3

import argparse
from tqdm import tqdm
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--dataset-mode", type=str, default="IMAGENET", help="(example: CIFAR10, CIFAR100, IMAGENET, FLOWER102, TINY_IMAGENET), (default: IMAGENET)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=512, help="number of batch size, (default, 512)")
    parser.add_argument("--learning-rate", type=float, default=1e-1, help="learning_rate, (default: 1e-1)")
    parser.add_argument("--dropout", type=float, default=0.8, help="dropout rate, not implemented yet, (default: 0.8)")
    parser.add_argument('--model-mode', type=str, default="LARGE", help="(example: LARGE, SMALL), (default: LARGE)")
    parser.add_argument("--load-pretrained", type=bool, default=False, help="(default: False)")
    parser.add_argument('--evaluate', type=bool, default=False, help="Testing time: True, (default: False)")
    parser.add_argument('--multiplier', type=float, default=1.0, help="(default: 1.0)")
    parser.add_argument('--print-interval', type=int, default=5, help="training information and evaluation information output frequency, (default: 5)")
    parser.add_argument('--data', default='data/tiny-imagenet-200', help="dataset root path, (default: data/tiny-imagenet-200)")
    parser.add_argument('--workers', type=int, default=0, help="data loading workers, (default: 0, set 0 on Windows)")
    parser.add_argument('--distributed', type=bool, default=False)

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# reference,
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Thank you.
class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0:
            progress.print(i)

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_interval == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

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
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
    """Label Smoothing Loss - 防止模型过度自信，提升泛化能力"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        n_classes = x.size(-1)
        # 构造平滑后的标签分布
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def plot_curves(train_losses, val_losses, train_acc1, val_acc1, args):
    """保存训练损失和准确率曲线图"""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{args.dataset_mode} - Loss Curve ({args.model_mode})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy曲线
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
    print('num_classes: ', num_classes)

    model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes, multiplier=args.multiplier, dropout_rate=args.dropout).to(device)
    if torch.cuda.device_count() >= 1:
        print("num GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)

    if args.load_pretrained or args.evaluate:
        filename = "best_model_" + str(args.model_mode) + "_" + str(args.dataset_mode)
        checkpoint = torch.load('./checkpoint/' + filename + '_ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc1 = checkpoint['best_acc1']
        acc5 = checkpoint['best_acc5']
        best_acc1 = acc1
        print("Load Model Accuracy1: ", acc1, " acc5: ", acc5, "Load Model end epoch: ", epoch)
    else:
        print("init model load ...")
        epoch = 1
        best_acc1 = 0

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=4e-5, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    # 使用 Label Smoothing 防止过拟合（对 Tiny ImageNet 等小数据集尤其有效）
    if args.dataset_mode == "TINY_IMAGENET":
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
        # 余弦退火学习率调度，比阶梯衰减更平滑
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = None

    if args.evaluate:
        val_loss, acc1, acc5 = validate(test_loader, model, criterion, args)
        print("Acc1: ", acc1, "Acc5: ", acc5)
        return

    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    # 记录训练过程的指标
    train_loss_history = []
    val_loss_history = []
    train_acc1_history = []
    val_acc1_history = []

    start_time = time.time()
    report_filename = f"best_model_{args.model_mode}_{args.dataset_mode}"
    with open(f"./reporting/{report_filename}.txt", "w") as f:
        for epoch in range(epoch, args.epochs):
            if args.dataset_mode != "TINY_IMAGENET":
                adjust_learning_rate(optimizer, epoch, args)

            train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args)
            val_loss, acc1, acc5 = validate(test_loader, model, criterion, args)

            if scheduler:
                scheduler.step()

            # 记录历史
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc1_history.append(train_acc1.item())
            val_acc1_history.append(acc1.item())

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                print('Saving..')
                best_acc5 = acc5
                state = {
                    'model': model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, f'./checkpoint/{report_filename}_ckpt.t7')

            time_interval = time.time() - start_time
            time_split = time.gmtime(time_interval)
            print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ", time_split.tm_sec, end='')
            print(" Test best acc1:", best_acc1, " acc1: ", acc1, " acc5: ", acc5)

            f.write("Epoch: " + str(epoch) + " " + " Best acc: " + str(best_acc1) + " Test acc: " + str(acc1) + "\n")
            f.write("Training time: " + str(time_interval) + " Hour: " + str(time_split.tm_hour) + " Minute: " + str(
                time_split.tm_min) + " Second: " + str(time_split.tm_sec))
            f.write("\n")

    # 训练结束后绘制曲线
    plot_curves(train_loss_history, val_loss_history, train_acc1_history, val_acc1_history, args)


if __name__ == "__main__":
    main()
