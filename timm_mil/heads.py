import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, sequence_length, dimensions, bias=True):

        super(Attention, self).__init__()

        weight = torch.zeros(dimensions, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(sequence_length))

    def forward(self, x):

        input_batch_size, input_sequence_length, input_dimensions = x.shape

        eij = torch.mm(
            x.contiguous().view(-1, input_dimensions),
            self.weight
        ).view(-1, input_sequence_length)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        output = torch.sum(weighted_input, 1)

        return output


class ConvolutionalHead(nn.Module):

    def __init__(
            self,
            input_dimensions, intermediate_dimensions, output_dimensions, pooling_type,
            activation, activation_args, dropout_probability=0., batch_normalization=False
    ):

        super(ConvolutionalHead, self).__init__()

        self.pooling_type = pooling_type
        self.attention = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_dimensions),
            Attention(sequence_length=49, dimensions=input_dimensions)
        ) if self.pooling_type == 'attention' else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(input_dimensions * 2 if pooling_type == 'concat' else input_dimensions, intermediate_dimensions, bias=True),
            getattr(nn, activation)(**activation_args),
            nn.BatchNorm1d(num_features=intermediate_dimensions) if batch_normalization else nn.Identity(),
            nn.Dropout(p=dropout_probability) if dropout_probability >= 0. else nn.Identity(),
            nn.Linear(intermediate_dimensions, output_dimensions, bias=True)
        )

    def forward(self, x):

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)
        elif self.pooling_type == 'attention':
            input_batch_size, feature_channel = x.shape[:2]
            x = x.contiguous().view(input_batch_size, feature_channel, -1).permute(0, 2, 1)
            x = self.attention(x)

        output = self.head(x)

        return output
