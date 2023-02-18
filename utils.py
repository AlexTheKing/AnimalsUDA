import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


def visualize_samples(dataset, indices, title=None, count=10, columns=5):
    size_coeff = count / 10
    plt.figure(figsize=(20, 5 * size_coeff))
    display_indices = indices[:count]
    if title:
        plt.suptitle(f'{title} {count}/{len(indices)}')
    rows = np.ceil(count / columns).astype(int)
    for index, sample_index in enumerate(display_indices):
        image, target = dataset[sample_index]
        plt.subplot(rows, columns, index + 1)
        plt.title(f'Label: {dataset.label_target(target)}')
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        plt.imshow(image)
        plt.grid(False)
        plt.axis('off')
