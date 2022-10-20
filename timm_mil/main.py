from glob import glob
import numpy as np
import cv2
import torch

from mil import ConvolutionalMultiInstanceLearningModel


if __name__ == '__main__':

    image_paths = glob('../data/*.jpg')
    image_paths = sorted(image_paths, key=lambda path: int(path.split('.')[-2].split('_')[-1]))

    x = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), (224, 224)) for image_path in image_paths])
    x = np.moveaxis(x, -1, 1)
    x = torch.as_tensor(x, dtype=torch.float)
    x = torch.unsqueeze(x, dim=0)
    x /= 255.

    print(f'Input Shape: {x.shape} - Mean: {torch.mean(x):.4f} Std: {torch.std(x):.4f} Min: {torch.min(x):.4f} Max: {torch.max(x):.4f}')

    model = ConvolutionalMultiInstanceLearningModel(
        n_instances=16,
        model_name='efficientnet_b0',
        pretrained=False,
        freeze_parameters=None,
        aggregation='max',
        head_class='ConvolutionalHead',
        head_args={
            'intermediate_dimensions': 512,
            'output_dimensions': 1,
            'pooling_type': 'attention',
            'activation': 'ReLU',
            'activation_args': {},
            'dropout_probability': 0.,
            'batch_normalization': False
        }
    )

    y = model(x)
    print(f'Output Shape: {y.shape} - Mean: {torch.mean(y):.4f} Std: {torch.std(y):.4f} Min: {torch.min(y):.4f} Max: {torch.max(y):.4f}')
