import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from myModels import TSN
from transforms import *
from opts import parser

best_prec1 = 0
fisrt_log = True

def main():

    global args, best_prec1,log_txt_name
    args = parser.parse_args()

    log_txt_name = 'log_no_clip_grad_ALL.txt'
    args.snapshot_pref = 'ucf101_ALL'
    args.eval_freq = 5
    # args.lr_steps = [116,125,186]
    # args.epochs = 206
    # args.lr_steps = [60, 120]
    args.lr_steps = [60, 90]
    args.epochs = 120
    save_epochs = 3

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # args.resume = 'ucf101_base_bnatt90838_onlyRes3D_125_96.56_train_.pth.tar'
    args.resume = 'ucf101_ALL_89_82.82_.pth.tar'

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            # model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            # model.load_state_dict(model_dict)
            # save_checkpoint({
            #     'epoch': args.start_epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            # }, True)

            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {}) (best_prec1{})"
                  .format(args.evaluate, checkpoint['epoch'],best_prec1)))

        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="iamge_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=6, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    # for i, (input, target) in enumerate(train_loader):
    #     if i >2 :
    #         exit()
    #     print(input.size())

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="iamge_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=3, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    # print(policies)
    # # for m in model.state_dict():
    # #     print(policies)
    # exit()
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate :
    # if True:
        validate(val_loader, model, criterion, 0)
        return



    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train_prec1 = train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % save_epochs == 0 and (epoch + 1) % args.eval_freq != 0:
            train_prec1 = ('{:.2f}'.format(train_prec1.cpu().item()))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, False, epoch, train_prec1,True)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            prec1 = ('{:.2f}'.format(prec1.cpu().item()))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best,epoch,prec1,False)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # print(input.size())
        # print(target.size())
        # if i <1900:
        #     print(i)
        #     continue

        # print(input.size())  #torch.Size([10, 126, 224, 224])
        # exit(0)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # RGB_output,output = model(input_var)
        # print(output.size())
        # print(target_var.size())
        # exit()
        # loss = criterion(output, target_var)
        loss = criterion(output, target_var) #+ criterion(RGB_output, target_var)

        # print(output)
        # print(output.size())
        # print(output.data)
        # print(output.data.size())
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
        #     if total_norm > args.clip_gradient:
        #         print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            p = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(p)
            if i % 50 == 0:
                with open(log_txt_name,'a')as txt:
                    # if(fisrt_log):
                    #     fisrt_log =False
                    #     txt.write('=====================================================================' + '\n')UnboundLocalError: local variable 'fisrt_log' referenced before assignment
                    txt.write(time.asctime( time.localtime(time.time()) )+ '    '+p + '\n')
    return top1.avg



def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):


        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        # print(input.size())
        # print(target.size())
        # print(output.size())
        # print(prec1.size())
        # print(prec5.size())
        # exit()

        # class_prec1 = prec1.cpu().item()
        # if class_prec1 < 0.1:
        #     with open('spcific_error_class.txt','a') as f:
        #         f.write(str(class_prec1) + '=======>' + str(target.cpu().item())+'=======>' + str(i) + '\n')

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            p = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(p)
            if i % 50 == 0:
                with open(log_txt_name,'a')as txt:
                    txt.write(time.asctime( time.localtime(time.time()) )+ '    '+p + '\n')

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))
    with open(log_txt_name, 'a')as txt:
        txt.write('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +'\n')
        txt.write(time.asctime(time.localtime(time.time())) + '    ' + ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)) + '\n')

    return top1.avg


def save_checkpoint(state, is_best,epoch,prec1,istrain,filename='.pth.tar'):

    if istrain:
        filename = '_'.join((args.snapshot_pref, str(epoch), str(prec1),'train', filename))
        torch.save(state, filename)
        return

    # filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    filename = '_'.join((args.snapshot_pref, str(epoch),str(prec1), filename))
    torch.save(state, filename)
    if is_best:
        # best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        best_name = '_'.join((args.snapshot_pref, str(epoch),str(prec1), 'best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
