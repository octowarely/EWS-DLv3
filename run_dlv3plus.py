import os
import torch
from methods.dlv3plus import DLv3Plus

# Ensure working directory is the script's directory (needed for relative data paths)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

pretrained = True  # Set to False to train from scratch

print('DeepLab v3+ ({})'.format('pretrained on ImageNet' if pretrained else 'from scratch'))
print('-' * 60)

model = DLv3Plus(pretrained=pretrained)

print('Training...')
train_metrics, val_metrics = model.train()
print('Train metrics:')
for k, v in train_metrics.items():
    print(f'  {k}: {v:.4f}')
print('Val metrics:')
for k, v in val_metrics.items():
    print(f'  {k}: {v:.4f}')

print('-' * 60)
print('Testing...')
test_metrics, preds, labels = model.test()
print('Test metrics:')
for k, v in test_metrics.items():
    print(f'  {k}: {v:.4f}')
