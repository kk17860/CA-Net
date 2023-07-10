from torch import nn
import torch
from functools import reduce
import torch.nn.functional as F
import math

class attention_resnet(nn.Module):
    def __init__(self, pre_channels, cur_channel, reduction=16):  
        super(attention_resnet, self).__init__()
        self.pre_fusions = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(pre_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = nn.Sequential(
                nn.Linear(cur_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )

        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(cur_channel // reduction, cur_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _, _ = cur_layer.size()

        pre_fusions = [self.pre_fusions[i](pre_layers[i].view(b, -1)) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer.view(b, -1))
        fusion = cur_fusion + sum(pre_fusions)

        att_weights = self.generation(fusion).view(b, cur_c, 1, 1, 1)

        return att_weights