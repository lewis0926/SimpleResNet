The code is implemented by Python language with Jupyter notebook. 
It requires numpy, matplotlib, torch, torchvision, random and time lib for general purpose.
Many functions from torch and torchvision, such as DataLoader, random_split, lr_scheduler, are called.
Resnet inherited from torch.nn.Module is used as model.

Main code can be found in main.ipynb.
genetic_algorithm.py includes some GA functions for training.
utility.py includes general functions of defining device using.
model.py includes the details of ResNet model.
train.py includes the detail code of training.
