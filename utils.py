# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:25:51 2018

@author: Yuxi Li
"""

import argparse
import torch
from torchvision import datasets, transforms

def get_dataloader(args):

    transform = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.augmentation:
      train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    else:
      train_transform = transform
    

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False)


    return train_loader, test_loader


def get_args():
    parser = argparse.ArgumentParser(description='CliqueNet')

    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_epochs', type=int, default=1)
    parser.add_argument('-lr', type=float, default=2e-2, help="Initial learning rate")
    parser.add_argument('-disable_cuda', action='store_true',
                    help='Disable CUDA')
    parser.add_argument('-augmentation', action='store_true',
                    help='Apply data augmentation')
    parser.add_argument('-print_freq', type=int, default=10, help="Log print frequency")
    parser.add_argument('-pretrained', type=str, default="Start from a pretrained model")
    parser.add_argument('-gpu', type=int, default=0, help = "Which gpu to use") 
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    return args


if __name__ == "__main__":
    args = get_args()
    loader,_ = get_dataloader(args)
    print(len(loader.dataset))
    for data in loader:
        x,y = data
        print(x[0,0,:,:])
        break
