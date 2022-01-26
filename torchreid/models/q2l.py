import torch
import torch.nn as nn
import math

from .transformer import build_position_encoding
from .common import ModelInterface
from torch.cuda.amp import autocast

__all__ = ['build_q2l']

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, use_bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        for i in range(self.num_class):
            self.weight[0][i].data.uniform_(-stdv, stdv)
        if self.use_bias:
            for i in range(self.num_class):
                self.bias[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.weight * x).sum(-1)
        if self.use_bias:
            x = x + self.bias
        return x


class BackboneWrapper(nn.Module):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.get_num_features()

    def forward(self, input):
        out = self.backbone(input, return_featuremaps=True)
        pos = self.position_embedding(out).to(out.dtype)
        return [out], [pos]


class Query2Label(ModelInterface):
    def __init__(self,
                backbone,
                transfomer,
                num_classes=80,
                pretrain=False,
                **kwargs):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_classes ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_classes
        assert self.loss in ['asl', 'bce', 'am_binary'], "Q2L supports only ASL, BCE pr AM Binary losses"

        hidden_dim = transfomer.get_hidden_dim()
        backbone_features = backbone.num_channels
        self.input_proj = nn.Conv2d(backbone_features, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.fc = GroupWiseLinear(num_classes, hidden_dim, use_bias=True)

    def forward(self, input):
        with autocast(enabled=self.mix_precision):
            src, pos = self.backbone(input)
            src, pos = src[-1], pos[-1]

            query_input = self.query_embed.weight
            hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
            logits = self.fc(hs[-1])
            if self.similarity_adjustment:
                logits = self.sym_adjust(logits, self.amb_t)

            if not self.training:
                return [logits]

            elif self.loss in ['asl', 'bce', 'am_binary']:
                    out_data = [logits]
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

            return tuple(out_data)

    def get_config_optim(self, lrs):
        parameters = [
            {'params': self.backbone.named_parameters()},
            {'params': self.fc.named_parameters()},
            {'params': self.input_proj.parameters(), 'weight_decay': 0.},
            {'params': self.query_embed.parameters(), 'weight_decay': 0.}
        ]
        if isinstance(lrs, list):
            assert len(lrs) == len(parameters)
            for lr, param_dict in zip(lrs, parameters):
                param_dict['lr'] = lr
        else:
            assert isinstance(lrs, float)
            for param_dict in parameters:
                param_dict['lr'] = lrs

        return parameters


def build_q2l(backbone, transformer, hidden_dim=2048, pretrain=False, input_size=448, **kwargs):
    position_emb = build_position_encoding(hidden_dim=hidden_dim, img_size=input_size)
    wrapped_model = BackboneWrapper(backbone, position_emb)
    model = Query2Label(
        backbone=wrapped_model,
        transfomer=transformer,
        pretrain=pretrain,
        **kwargs
    )

    return model
