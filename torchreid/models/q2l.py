import torch
import torch.nn as nn
import math

from .transformer import build_position_encoding
from .common import ModelInterface

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
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
        self.num_channels = backbone.num_features
        # self.args = args

    def forward(self, input):
        out = self.backbone(input, return_featuremaps=True)
        pos = self.position_embedding(out).to(out.dtype)
        return [out], [pos]


class BackboneWrapper(nn.Module):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_features
        # self.args = args

    def forward(self, input):
        out = self.backbone(input, return_featuremaps=True)
        pos = self.position_embedding(out).to(out.dtype)
        return [out], [pos]


class Qeruy2Label(ModelInterface):
    def __init__(self,
                backbone,
                transfomer,
                lr_finder=None,
                num_classes=80,
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
        self.lr_finder = lr_finder
        assert self.loss in ['softmax', 'asl', 'bce'], "Q2L supports only softmax or ASL losses"

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."

        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.fc = GroupWiseLinear(num_classes, hidden_dim, use_bias=True)


    def forward(self, input):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
        logits = self.fc(hs[-1])
        # import ipdb; ipdb.set_trace()
        if not self.training and self.is_classification():
            return [logits]

        elif self.loss in ['softmax', 'am_softmax', 'asl']:
                out_data = [logits]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        if self.lr_finder.enable and self.lr_finder.lr_find_mode == 'automatic':
            return out_data
        return tuple(out_data)

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())


def build_q2l(backbone, transformer, hidden_dim=2048, input_size=448, **kwargs):
    position_emb = build_position_encoding(hidden_dim=hidden_dim, img_size=input_size)
    wrapped_model = BackboneWrapper(backbone, position_emb)
    model = Qeruy2Label(
        backbone=wrapped_model,
        transfomer=transformer,
        **kwargs
    )

    return model
