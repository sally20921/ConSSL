import torch
import torch.nn as nn

class MLP(nn.Module):
    '''logistic regression. same as linear regression without further processing'''
    def __init__(self, args, pt_model, num_classes):
        super(MLP, self).__init__()
        n_channels = pt_model.output_dim
        self.classifier = nn.Sequential
        self.classifier.add_module('W1', nn.Linear(n_channels, num_classes))

    def forward(self, x):
        return self.classifier(x)

class LinearLayer(nn.Module):
    '''freeze the encoder and train the supervised classification head with a cross entropy loss'''
    def __init__(self, args, pt_model, num_classes):
        super(LinearLayer, self).__init__()
        self.encoder = pt_model.encoder
        self.mlp = MLP(args, pt_model, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        output = self.mlp(h)
        return output



