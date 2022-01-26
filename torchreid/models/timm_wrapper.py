import timm

from torchreid.losses import AngleSimpleLinear
from .common import ModelInterface
from torchreid.ops import Dropout
from torch import nn
from torch.cuda.amp import autocast

__all__ = ["timm_wrapped_models", "TimmModelsWrapper"]
AVAI_MODELS = {
                'mobilenetv3_large_21k' : 'mobilenetv3_large_100_miil_in21k',
                'mobilenetv3_large_1k' : 'mobilenetv3_large_100_miil',
                'tresnet' : 'tresnet_m',
                'efficientnetv2_s_21k': 'tf_efficientnetv2_s_in21k',
                'efficientnetv2_s_1k': 'tf_efficientnetv2_s_in21ft1k',
                'efficientnetv2_m_21k': 'tf_efficientnetv2_m_in21k',
                'efficientnetv2_m_1k': 'tf_efficientnetv2_m_in21ft1k',
                'efficientnetv2_b0' : 'tf_efficientnetv2_b0',
                'resnet101' : "tv_resnet101"
              }

class TimmModelsWrapper(ModelInterface):
    def __init__(self,
                 model_name,
                 pretrained=False,
                 dropout_cls = None,
                 pooling_type='avg',
                 **kwargs):
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.is_mobilenet = True if model_name in ["mobilenetv3_large_100_miil_in21k", "mobilenetv3_large_100_miil"] else False
        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       num_classes=self.num_classes)
        self.num_head_features = self.model.num_features
        self.num_features = (self.model.conv_head.in_channels if self.is_mobilenet
                             else self.model.num_features)
        self.dropout = Dropout(**dropout_cls)
        self.pooling_type = pooling_type
        if self.loss in ["am_softmax", "am_binary"]:
            self.model.act2 = nn.PReLU()
            self.model.classifier = AngleSimpleLinear(self.num_head_features, self.num_classes)
        else:
            assert self.loss in ["softmax", "asl", "bce"]
            self.model.classifier = self.model.get_classifier()

    def forward(self, x, return_featuremaps=False, return_all=False, **kwargs):
        with autocast(enabled=self.mix_precision):
            y = self.extract_features(x)
            if return_featuremaps:
                return y

            glob_features = self._glob_feature_vector(y, self.pooling_type, reduce_dims=False)
            logits = self.infer_head(glob_features)
            if self.similarity_adjustment:
                logits = self.sym_adjust(logits, self.amb_t)

            if return_all:
                return [(logits, y, glob_features)]
            if not self.training:
                return [logits]
            return tuple([logits])

    def extract_features(self, x):
        if self.is_mobilenet:
            x = self.model.conv_stem(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            y = self.model.blocks(x)
            return y
        return self.model.forward_features(x)

    def infer_head(self, x):
        if self.is_mobilenet:
            x  = self.model.act2(self.model.conv_head(x))
        self.dropout(x)
        return self.model.classifier(x.view(x.shape[0], -1))

    def get_config_optim(self, lrs):
        parameters = [
            {'params': self.model.named_parameters()},
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

class ModelFactory:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def __call__(self, **kwargs):
        """
        Constructs a timm model
        """
        net = TimmModelsWrapper(self.model_name, **kwargs)
        return net

timm_wrapped_models = {dor_name : ModelFactory(model_name=timm_name) for dor_name, timm_name in AVAI_MODELS.items()}
