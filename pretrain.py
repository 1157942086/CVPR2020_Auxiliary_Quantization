import os
import sys
import math
import time
import logging
import sys
import argparse
import glob
import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from model import resnet50
import utils
from utils import adjust_learning_rate, save_checkpoint
import numpy as np
from random import shuffle


parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--resume_train', action='store_true', default=False, help='resume training')
parser.add_argument('--resume_dir', type=str, default='./weights/checkpoint.pth.tar', help='save weights directory')
parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
parser.add_argument('--bitW', type=int, default=32, help='weight precision')
parser.add_argument('--bitA', type=int, default=32, help='activation precision')


args = parser.parse_args()
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.exists(args.save):
    os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


# Image Preprocessing 
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])


    num_epochs = args.epochs
    batch_size = args.batch_size


    train_dataset = datasets.folder.ImageFolder(root='/data/train/', transform=train_transform)
    test_dataset = datasets.folder.ImageFolder(root='/data/val/', transform=test_transform)

 

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True, num_workers=10, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64, 
                                          shuffle=False, num_workers=10, pin_memory=True)

    

    num_train = train_dataset.__len__()
    n_train_batches = math.floor(num_train / batch_size)

    criterion = nn.CrossEntropyLoss().cuda()
    model = resnet50(args, pretrained=True)
    model = utils.dataparallel(model, 4)


    test_record = []
    train_record = []
    epoch = 0
    best_top1 = 0

    optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,15,20])

    # define the binarization operator

    while epoch < num_epochs:

        epoch = epoch + 1
    # resume training    
        if (args.resume_train) and (epoch == 1):   
            checkpoint = torch.load(args.resume_dir)
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            test_record = list(
                np.load(args.weights_dir + 'test_record.npy'))
            train_record = list(
                np.load(args.weights_dir + 'train_record.npy'))

            for i in range(epoch):
                scheduler.step()

        logging.info('epoch %d', epoch)

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)


    # training
        
        F_train_acc_top1, F_train_acc_top5, H_train_acc_top1, H_train_acc_top5, train_obj = train(train_loader, model, criterion, optimizer)

        logging.info('train_acc %f %f', F_train_acc_top1, H_train_acc_top1)
        train_record.append([F_train_acc_top1, F_train_acc_top5, H_train_acc_top1, H_train_acc_top5])
        np.save(args.weights_dir + 'train_record.npy', train_record)   

    # test
        F_test_acc_top1, F_test_acc_top5, H_test_acc_top1, H_test_acc_top5, test_obj = infer(test_loader, model, criterion)
        is_best = F_test_acc_top1 > best_top1
        if is_best:
            best_top1 = F_test_acc_top1

        logging.info('test_acc %f %f', F_test_acc_top1, H_test_acc_top1)
        test_record.append([F_test_acc_top1, F_test_acc_top5, H_test_acc_top1, H_test_acc_top5])
        np.save(args.weights_dir + 'test_record.npy', test_record)


        scheduler.step()

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_top1': best_top1,
                }, args, is_best)




def train(train_queue, model, criterion, optimizer):

    F_objs = utils.AvgrageMeter()
    F_top1 = utils.AvgrageMeter()
    F_top5 = utils.AvgrageMeter()

    H_objs = utils.AvgrageMeter()
    H_top1 = utils.AvgrageMeter()
    H_top5 = utils.AvgrageMeter()

    model.train()
 
    for step, (input, target) in enumerate(train_queue):
   
        n = input.size(0)
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        F_out, H_out = model(input)
        loss_1 = criterion(F_out, target)
        loss_2 = criterion(H_out, target)
        loss = loss_1 + loss_2

        loss.backward()

        optimizer.step()

        prec1, prec5 = utils.accuracy(F_out, target, topk=(1, 5))
        F_objs.update(loss_1.item(), n)
        F_top1.update(prec1.item(), n)
        F_top5.update(prec5.item(), n)

        prec1, prec5 = utils.accuracy(H_out, target, topk=(1, 5))
        H_objs.update(loss_2.item(), n)
        H_top1.update(prec1.item(), n)
        H_top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, F_objs.avg, F_top1.avg, F_top5.avg)
            logging.info('train %03d %e %f %f', step, H_objs.avg, H_top1.avg, H_top5.avg)
             
    return F_top1.avg, F_top5.avg, H_top1.avg, H_top5.avg, F_objs.avg



def infer(valid_queue, model, criterion):

    
    F_objs = utils.AvgrageMeter()
    F_top1 = utils.AvgrageMeter()
    F_top5 = utils.AvgrageMeter()

    H_objs = utils.AvgrageMeter()
    H_top1 = utils.AvgrageMeter()
    H_top5 = utils.AvgrageMeter()

    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            F_out, H_out = model(input)
            loss_1 = criterion(F_out, target)
            loss_2 = criterion(H_out, target)
 
            n = input.size(0)
            prec1, prec5 = utils.accuracy(F_out, target, topk=(1, 5))
            F_objs.update(loss_1.item(), n)
            F_top1.update(prec1.item(), n)
            F_top5.update(prec5.item(), n)

            prec1, prec5 = utils.accuracy(H_out, target, topk=(1, 5))
            H_objs.update(loss_2.item(), n)
            H_top1.update(prec1.item(), n)
            H_top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, F_objs.avg, F_top1.avg, F_top5.avg)
                logging.info('valid %03d %e %f %f', step, H_objs.avg, H_top1.avg, H_top5.avg)

    return F_top1.avg, F_top5.avg, H_top1.avg, H_top5.avg, F_objs.avg
 

if __name__ == '__main__':
    utils.create_folder(args)       
    main()
