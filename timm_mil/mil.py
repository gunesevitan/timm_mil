import torch
import torch.nn as nn
import timm

import heads


class ConvolutionalMultiInstanceLearningModel(nn.Module):

    def __init__(self, n_instances, model_name, pretrained, freeze_parameters, aggregation, head_class, head_args):

        super(ConvolutionalMultiInstanceLearningModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=head_args['output_dimensions']
        )

        if freeze_parameters is not None:
            # Freeze all parameters in backbone
            if freeze_parameters == 'all':
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            else:
                # Freeze specified parameters in backbone
                for group in freeze_parameters:
                    if isinstance(self.backbone, timm.models.DenseNet):
                        for parameter in self.backbone.features[group].parameters():
                            parameter.requires_grad = False
                    elif isinstance(self.backbone, timm.models.EfficientNet):
                        for parameter in self.backbone.blocks[group].parameters():
                            parameter.requires_grad = False

        self.aggregation = aggregation
        n_classifier_features = self.backbone.get_classifier().in_features
        input_features = (n_classifier_features * n_instances) if self.aggregation == 'concat' else n_classifier_features
        self.head = getattr(heads, head_class)(input_dimensions=input_features, **head_args)

    def forward(self, x):

        # Stack instances on batch dimension before passing input to feature extractor
        input_batch_size, input_instance, input_channel, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_instance, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)
        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.aggregation == 'avg':
            # Average feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.mean(x, dim=1)
        elif self.aggregation == 'max':
            # Max feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.max(x, dim=1)[0]
        elif self.aggregation == 'logsumexp':
            # LogSumExp feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.logsumexp(x, dim=1)
        elif self.aggregation == 'concat':
            # Stack feature maps on channel dimension
            x = x.contiguous().view(input_batch_size, input_instance * feature_channel, feature_height, feature_width)

        output = self.head(x)
        return output
