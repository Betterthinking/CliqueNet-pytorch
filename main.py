# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import os

from net import CliqueNet
from utils import get_args, get_dataloader

if __name__ == "__main__":

    args = get_args()
    train_loader, test_loader = get_dataloader(args)
    use_cuda = args.use_cuda
    num_classes = 10
    dropout_prob = 0.2
    #hyper-parameters

#    A,B,C,D,E,r = 32,32,32,32,10,args.r # a classic CapsNet
    model = CliqueNet(3, num_classes, 5, 40, attention=True, compression=True, dropout_prob=dropout_prob)
    criterion = CrossEntropyLoss()
    #closs = CrossEntropyLoss()

    with torch.cuda.device(args.gpu):
#        print(args.gpu, type(args.gpu))
        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained))
        if use_cuda:
            print("activating cuda")
            model = model.cuda()
        
        total_epochs = args.num_epochs
        milestones = [int(total_epochs*0.5), int(total_epochs*0.75)]
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        best_acc = 0.0
        for epoch in range(args.num_epochs):
            #Train
            print("Epoch {}".format(epoch))
            b = 0
            correct = 0
            model.train()
            for data in train_loader:
                b += 1

                optimizer.zero_grad()
                imgs,labels = data #n,1,28,28;
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                
                score = model(imgs)
                loss = criterion(score, labels)
                loss.backward()
                optimizer.step()
                #stats
                pred = score.max(1)[1] #b
                acc = pred.eq(labels).cpu().sum().data.item()
                correct += acc
                if b % args.print_freq == 0:                          
                    print("batch:{}".format(b))
                    print("total loss: {:.4f},  acc: {:}/{}".format(loss.data.item(), acc, args.batch_size))
            acc = float(correct)/len(train_loader.dataset)
            print("Epoch{} Train acc:{:4}".format(epoch, acc))
            scheduler.step(acc)
            #Test
            print('Testing...')
            model.eval()
            correct = 0
            for data in test_loader:
                imgs,labels = data #b,1,28,28
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                score = model(imgs) 
                #stats
                pred = score.max(1)[1]
                acc = pred.eq(labels).cpu().sum().data.item()
                correct += acc
            acc = float(correct)/len(test_loader.dataset)
            print("Epoch{} Test acc:{:4}".format(epoch, acc))

            if acc >= best_acc:
                best_acc = acc
                if not os.path.exists('./model'):
                    os.makedirs('./model')
                print("Writing checkpoint to: model/model_{}.pth".format(epoch))
                torch.save(model.state_dict(), "model/model_{}.pth".format(epoch))
            
            
            
            

        
        
