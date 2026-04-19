import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
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

# Per-class average summary
plant_keys = [k for k in test_metrics if 'plants' in k]
soil_keys  = [k for k in test_metrics if 'soil' in k]
print('\nVegetation (plants) average: {:.4f}'.format(np.mean([test_metrics[k] for k in plant_keys])))
print('Soil average:                {:.4f}'.format(np.mean([test_metrics[k] for k in soil_keys])))
print('  (averaged over: f1, precision, recall, IoU)')

# Visualization: Original / Ground Truth / Prediction
test_paths = sorted(glob.glob('data/test/*6.png'))
num_samples = min(4, len(test_paths))

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
fig.suptitle('DeepLab v3+ — Test Results', fontsize=14)
col_titles = ['Original', 'Ground Truth', 'Prediction']

for i in range(num_samples):
    img = np.array(PILImage.open(test_paths[i]).convert('RGB'))
    gt  = labels[i].reshape(350, 350).numpy()
    pr  = preds[i].reshape(350, 350).numpy()

    for j, (data, cmap) in enumerate([(img, None), (gt, 'gray'), (pr, 'gray')]):
        axes[i, j].imshow(data, cmap=cmap)
        axes[i, j].set_title(col_titles[j])
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
print('\nVisualization saved to test_visualization.png')
