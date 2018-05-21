This repository provide the pytorch re-implementation of CliqueNet, the site of original paper is [here](https://arxiv.org/abs/1802.10419). In this implementation, the test is done on CIFAR-10 dataset.

There are several different from the implementation of the statement in paper.
- we use a post activation of conv-bn-relu instead of pre-activation bn-relu-conv
- we adopt the strategy of attention transition and compression, but we didn't adopt the bottleneck inside clique blocks
- we offer a simple data augmentation option for random flip

## Requirement
Our code is based on the latest version of pytorch, please visit the [official site](https://pytorch.org) to install the latest version.

## Usage

To train a cliquenet on CIFAR-10, please refer the following command:
```Shell
python main.py [-h] [-batch_size BATCH_SIZE] [-num_epochs NUM_EPOCHS] [-lr LR]
               [-clip CLIP] [-disable_cuda] [-augmentation]
               [-print_freq PRINT_FREQ] [-pretrained PRETRAINED] [-gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  -batch_size BATCH_SIZE
  -num_epochs NUM_EPOCHS
  -lr LR                Initial learning rate
  -disable_cuda         Disable CUDA
  -augmentation         Apply data augmentation
  -print_freq PRINT_FREQ
                        Log print frequency
  -pretrained PRETRAINED
  -gpu GPU              Which gpu to use


```

you can also modify the hyperparameters in `main.py` to change the net configuration

## Results

We conduct a simple version of experiment, the dropout ratio of our network is 0.1, we train the network for 200 epochs without data augmentation. The current result on CIFAR-10 test set is only at most 92.23, there is still a large margin between ours and the paper results. We will try to fix it later.